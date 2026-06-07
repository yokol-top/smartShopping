"""
主Agent Orchestrator（编排器）

实现主Agent的三重身份：
  a) 执行者  —— 简单任务自己通过TaskPlanner完成
  b) 指挥官  —— 中等/复杂任务分解、动态创建子Agent委派、汇总
  c) 对话者  —— 唯一与用户直接交互的角色

一次复杂任务的完整生命周期：

  Phase 1: 理解（Understand）
  ├── 解析用户意图（可能需要追问）
  ├── 自己先做初步探索（读关键文件、了解项目结构）  ← 自己做！
  └── 判断任务复杂度

  Phase 2: 规划（Plan）
  ├── 分解任务为子步骤
  ├── 决定哪些自己做，哪些委派                     ← 决策
  ├── 确定子任务之间的依赖关系
  └── 制定执行顺序（串行 vs 并行）

  Phase 3: 执行（Execute）
  ├── 自己直接完成简单子任务（串行）                ← 自己做！
  ├── 为子Agent构造精确的task prompt              ← 信息压缩
  ├── 并行启动委派子Agent（asyncio.gather）
  └── 等待所有委派任务完成后，再串行执行主Agent任务  ← 实现顺序

  Phase 4: 整合（Integrate）
  ├── 接收子Agent结果（只看"总结邮件"）
  ├── 验证结果质量（子Agent可能做错）              ← 质量把关
  ├── 如果不满意 → 重试子Agent 或自己修正
  ├── 整合多个子Agent的结果
  └── 可能需要额外的收尾工作                      ← 自己做！

  Phase 5: 交付（Deliver）
  ├── 总结做了什么，告知用户
  └── 处理用户的后续问题

委派子Agent的判断条件（满足任一）：
1. 可以拆成独立子任务且并行能显著加速
2. 上下文快满了或子任务会产生大量中间结果
3. 与当前主线任务无关（用户突然岔开话题）
"""

import json
import logging
import re
import time
from typing import Dict, Any, Optional

from utils import LLMClient
from .exceptions import NeedUserInputException
from .intent_recognizer import IntentResult, IntentType, TaskComplexity
from .orchestrator_memory import OrchestratorMemory
from .sub_agent_factory import SubAgentFactory
from .task_state import (
    TaskState, TaskPhase, SubTask, SubTaskStatus,
    SubAgentResult, DelegationReason,
)


class Orchestrator:
    """主Agent编排器，实现完整的5阶段任务生命周期。

    - 子Agent动态创建（角色、工具、上下文按需配置）
    - 主Agent只看子Agent的结果摘要
    - 任务状态可序列化，支持会话恢复
    - 失败处理：重试→降级→熔断
    """

    def __init__(
            self,
            llm_client: LLMClient,
            mcp_manager=None,
            rag_engine=None,
            task_planner=None,
            tool_manager=None,
            context_manager=None,
            config: Dict[str, Any] = None,
            logger: logging.Logger = None,
    ):
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.rag_engine = rag_engine
        self.task_planner = task_planner
        self.tool_manager = tool_manager
        self.context_manager = context_manager
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # 编排器配置
        orch_config = self.config.get('orchestrator', {})
        self.context_overflow_threshold = orch_config.get('context_overflow_threshold', 0.75)
        self.max_subtask_retries = orch_config.get('max_subtask_retries', 2)
        self.parallel_threshold = orch_config.get('parallel_threshold', 2)

        # 子Agent工厂
        self.factory = SubAgentFactory(
            llm_client=llm_client,
            mcp_manager=mcp_manager,
            config=self.config,
            logger=logger,
            context_manager=context_manager,
            task_planner=task_planner,
        )

        # Orchestrator结构化任务记忆
        self.memory = OrchestratorMemory(
            llm_client=llm_client, logger=logger,
        )

        # 当前活跃的TaskState
        self._current_task: Optional[TaskState] = None

        self._last_request_failed: bool = False

        self.logger.info("[Orchestrator] 初始化完成")

    # ================================================================
    # 公开接口
    # ================================================================

    def handle_request(
            self,
            user_query: str,
            intent_result: IntentResult,
            context: str = "",
            long_term_context: str = "",
            user_id: str = "",
            username: str = "",
            verbose: bool = True,
    ) -> str:
        """处理用户请求的主入口

        路由矩阵（intent_type × complexity）：

                       simple          medium              complex
        ─────────────────────────────────────────────────────────────
        RAG_SIMPLE   直接 RAG          直接 RAG            P&E
        RAG_ADVANCED 直接 RAG          P&E（并行步骤）     分层规划（DAG并行阶段）
        MCP_EXECUTE  ReAct             P&E（并行步骤）     5阶段生命周期
        CHAT/其他    直接回答          直接回答            直接回答

        覆盖规则（优先级高于矩阵）：
          is_plannable=False → 强制走 ReAct（步骤未知，需边做边看）
        """
        self._last_request_failed = False

        intent_type = intent_result.intent_type
        complexity  = intent_result.complexity

        self.logger.info(
            f"[Orchestrator] 路由 | intent={intent_type.value} "
            f"complexity={complexity.value} is_plannable={intent_result.is_plannable}"
        )

        try:
            # ── 覆盖规则：is_plannable=False → ReAct ─────────────────────
            if not intent_result.is_plannable:
                return self._execute_simple(
                    user_query, intent_result, context, long_term_context, verbose
                )

            # ── Greeting / MCP_ASK_INFO / SIMPLE_CHAT → 直接回答 ─────────
            if intent_type in (IntentType.GREETING, IntentType.MCP_ASK_INFO,
                               IntentType.SIMPLE_CHAT):
                return self._execute_simple(
                    user_query, intent_result, context, long_term_context, verbose
                )

            # ── RAG_SIMPLE → 直接 RAG（复杂度不影响，单问题快速回答）────
            if intent_type == IntentType.RAG_SIMPLE:
                return self._execute_simple(
                    user_query, intent_result, context, long_term_context, verbose
                )

            # ── RAG_ADVANCED × complexity ─────────────────────────────────
            if intent_type == IntentType.RAG_ADVANCED:
                if complexity == TaskComplexity.COMPLEX:
                    # 跨类目/多阶段分析 → 分层规划（DAG 并行阶段）
                    return self._execute_direct(
                        user_query, intent_result, context, long_term_context,
                        TaskComplexity.COMPLEX, verbose,
                    )
                elif complexity == TaskComplexity.MEDIUM:
                    # 多步骤分析 → P&E（步骤内 DAG 并行）
                    return self._execute_direct(
                        user_query, intent_result, context, long_term_context,
                        TaskComplexity.MEDIUM, verbose,
                    )
                else:
                    # simple → 直接 RAG
                    return self._execute_simple(
                        user_query, intent_result, context, long_term_context, verbose
                    )

            # ── MCP_EXECUTE × complexity ──────────────────────────────────
            if intent_type == IntentType.MCP_EXECUTE:
                if complexity == TaskComplexity.COMPLEX:
                    # 多阶段操作（推荐→下单、跨类目采购）→ 5阶段生命周期
                    # 需要：SubAgent 隔离、熔断器、任务状态持久化
                    return self._execute_lifecycle(
                        user_query, intent_result, context, long_term_context,
                        user_id, username, verbose,
                    )
                elif complexity == TaskComplexity.MEDIUM:
                    # 多步骤操作（查用户→搜商品→下单）→ P&E（步骤内 DAG 并行）
                    return self._execute_direct(
                        user_query, intent_result, context, long_term_context,
                        TaskComplexity.MEDIUM, verbose,
                    )
                else:
                    # simple（查订单/查用户）→ ReAct 单步探索
                    return self._execute_simple(
                        user_query, intent_result, context, long_term_context, verbose
                    )

            # ── 兜底：未知意图类型 → 直接回答 ────────────────────────────
            return self._execute_simple(
                user_query, intent_result, context, long_term_context, verbose
            )

        except NeedUserInputException as e:
            self.logger.info(f"[Orchestrator] 执行中途需要用户输入: {e.question[:80]}")
            # 用统一前缀标记，由 agent.py 的 chat() 识别并返回给用户
            return f"__NEED_INPUT__:{e.question}"

    def get_current_task_state(self) -> Optional[TaskState]:
        """获取当前任务状态（供持久化）"""
        return self._current_task

    def restore_task_state(self, state_data: Dict[str, Any]):
        """从持久化数据恢复任务状态"""
        try:
            self._current_task = TaskState.deserialize(state_data)
            self.logger.info(
                f"[Orchestrator] 恢复任务状态: {self._current_task.summary()}"
            )
        except Exception as e:
            self.logger.error(f"[Orchestrator] 恢复任务状态失败: {e}")
            self._current_task = None

    def enrich_context(self, user_query: str, context: str) -> str:
        """用orchestrator记忆enrich上下文（保持向后兼容）"""
        self.memory.extract_entities_from_context(context)
        resolved = self.memory.resolve_reference(user_query, context)
        if resolved:
            self.logger.info(f"[Orchestrator] 引用解析完成: {resolved}")
            # B3 fix: 将解析结果注入上下文，否则 LLM 无法感知"方案一"对应的实际商品
            return context + f"\n[引用解析]\n{resolved}"
        return context

    # ================================================================
    # Phase 0: 简单任务直接执行
    # ================================================================

    def _execute_simple(
            self,
            user_query: str,
            intent_result: IntentResult,
            context: str,
            long_term_context: str,
            verbose: bool,
    ) -> str:
        """简单任务：主Agent直接通过TaskPlanner执行"""
        if self.task_planner:
            # 注入orchestrator记忆
            orchestrator_ctx = self.memory.get_context_for_sub_agent()
            if orchestrator_ctx:
                context += f"\n[Orchestrator已知信息]\n{orchestrator_ctx}"

            response = self.task_planner.execute(
                user_query=user_query,
                intent=intent_result,
                context=context,
                long_term_context=long_term_context,
                verbose=verbose,
                on_step_complete=self.memory.extract_entities_from_step_result,
            )

            # 更新orchestrator记忆
            self.memory.update_from_sub_agent_result(
                route="main", query=user_query, response=response,
            )
            return response

        return self.llm_client.generate(
            prompt=f"用户问题：{user_query}\n上下文：{context}\n请回答：",
        ).strip()

    # ================================================================
    # Phase 0b: 直通 TaskPlanner（跳过5阶段开销）
    # ================================================================

    def _execute_direct(
            self,
            user_query: str,
            intent_result: IntentResult,
            context: str,
            long_term_context: str,
            forced_complexity: TaskComplexity,
            verbose: bool,
    ) -> str:
        """可规划的串行任务：跳过5阶段编排，直接调用 TaskPlanner。

        - forced_complexity=MEDIUM  → Plan-and-Execute
        - forced_complexity=COMPLEX → 分层规划（Hierarchical）

        相比 _execute_simple，区别在于会强制覆盖 complexity，
        让 TaskPlanner 走到对应的规划路径，而不是 ReAct。
        """
        if verbose:
            label = "Plan-and-Execute" if forced_complexity == TaskComplexity.MEDIUM else "分层规划"
            print(f"\n📋 直通 {label}（跳过5阶段编排）\n")

        if not self.task_planner:
            return self.llm_client.generate(
                prompt=f"用户问题：{user_query}\n上下文：{context}\n请回答："
            ).strip()

        # 注入 Orchestrator 记忆
        orchestrator_ctx = self.memory.get_context_for_sub_agent()
        enriched_context = context
        if orchestrator_ctx:
            enriched_context += f"\n[Orchestrator已知信息]\n{orchestrator_ctx}"

        # 用指定复杂度覆盖，让 TaskPlanner 选择对应策略
        # L3 fix: 同步携带意图识别阶段已提取的 _goal/_constraints，
        # 避免 Phase1/TaskPlanner 降级触发多余的 LLM 调用
        directed_intent = IntentResult(
            intent_type=intent_result.intent_type,
            complexity=forced_complexity,
            tool_name=intent_result.tool_name,
            confidence=intent_result.confidence,
            requires_rag=intent_result.requires_rag,
            requires_mcp=intent_result.requires_mcp,
        )
        directed_intent._goal = getattr(intent_result, '_goal', '')
        directed_intent._constraints = getattr(intent_result, '_constraints', [])

        response = self.task_planner.execute(
            user_query=user_query,
            intent=directed_intent,
            context=enriched_context,
            long_term_context=long_term_context,
            verbose=verbose,
            on_step_complete=self.memory.extract_entities_from_step_result,
        )

        self.memory.update_from_sub_agent_result(
            route="main", query=user_query, response=response,
        )
        return response

    # ================================================================
    # 完整5阶段生命周期
    # ================================================================

    def _execute_lifecycle(
            self,
            user_query: str,
            intent_result: IntentResult,
            context: str,
            long_term_context: str,
            user_id: str,
            username: str,
            verbose: bool,
    ) -> str:
        """MCP_EXECUTE + complex：完整5阶段生命周期执行。

        适用场景：多阶段操作类任务（推荐→下单、跨类目采购等），需要：
        - SubAgent 隔离（避免操作副作用扩散）
        - 熔断器保护（防止连续失败）
        - TaskState 持久化（支持会话恢复）
        """

        # 创建任务状态
        state = TaskState(user_query=user_query)
        self._current_task = state

        try:
            # Phase 1: 理解
            state.advance_phase(TaskPhase.UNDERSTAND)
            if verbose:
                print("\n🧠 Phase 1: 理解任务...")
            self._phase_understand(state, intent_result, context, long_term_context)

            # Phase 2: 规划
            state.advance_phase(TaskPhase.PLAN)
            if verbose:
                print("📋 Phase 2: 制定计划...")
            self._phase_plan(state, context, user_id, username)

            if verbose:
                self._print_plan(state)

            # Phase 3: 执行
            state.advance_phase(TaskPhase.EXECUTE)
            if verbose:
                print("\n⚡ Phase 3: 执行任务...")
            self._phase_execute(state, context, long_term_context, user_id, username, verbose)

            # Phase 4: 整合
            state.advance_phase(TaskPhase.INTEGRATE)
            if verbose:
                print("\n🔗 Phase 4: 整合结果...")
            self._phase_integrate(state, user_query, context, verbose)

            # Phase 5: 交付
            state.advance_phase(TaskPhase.DELIVER)
            if verbose:
                print("📦 Phase 5: 生成回复...\n")
            self._phase_deliver(state, user_query, context, long_term_context)

            state.advance_phase(TaskPhase.COMPLETED)
            return state.final_response

        except NeedUserInputException:
            raise  # 由 handle_request() 统一捕获转换为 __NEED_INPUT__ 前缀
        except Exception as e:
            state.advance_phase(TaskPhase.FAILED)
            self.logger.error(f"[Orchestrator] 任务执行失败: {e}")
            # 降级：尝试直接回答
            return self._fallback_response(user_query, context, long_term_context, str(e))

    # ================================================================
    # Phase 1: 理解
    # ================================================================

    def _phase_understand(
            self,
            state: TaskState,
            intent_result: IntentResult,
            context: str,
            long_term_context: str,
    ):
        """Phase 1: 理解用户意图，提取目标和约束"""
        state.complexity = intent_result.complexity.value

        # 优先复用意图识别阶段已提取的目标（避免重复 LLM 调用）
        llm_goal = getattr(intent_result, '_goal', '')
        llm_constraints = getattr(intent_result, '_constraints', [])

        if llm_goal:
            state.user_goal = llm_goal
            state.constraints = list(llm_constraints) if isinstance(llm_constraints, list) else []
            # intent_result.reason 也追加到约束
            if hasattr(intent_result, 'reason') and intent_result.reason:
                state.constraints.append(intent_result.reason)
            self.logger.info(
                f"[Orchestrator] Phase1：复用意图识别结果 | "
                f"目标={state.user_goal[:60]} | 约束数={len(state.constraints)}"
            )
            return   # 不再调用 LLM

        # 降级：_goal 为空时（如使用启发式识别）才调用 LLM
        self.logger.info("[Orchestrator] Phase1：_goal 为空，降级调用 LLM 提取目标")
        try:
            prompt = f"""分析以下用户的购物请求，提取关键信息。

用户输入：{state.user_query}
对话上下文：{context[-600:] if context else '无'}

请以JSON格式返回：
{{"goal": "用户的核心目标（一句话概括）", "constraints": ["约束1", "约束2"], "implicit_needs": "用户可能隐含的需求"}}

goal 示例：
- "推荐5000元以内的游戏手机并下单"
- "查询最近的订单状态"
- "比较三款耳机然后买最好的"

constraints 只包含明确提到的限制（如预算、品牌偏好、数量要求），没有则返回空列表。
只返回JSON。"""

            resp = self.llm_client.generate(prompt=prompt, temperature=0.1).strip()
            # 提取JSON
            code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', resp, re.DOTALL)
            if code_block:
                resp = code_block.group(1).strip()
            brace_start = resp.find('{')
            brace_end = resp.rfind('}')
            if brace_start >= 0 and brace_end > brace_start:
                resp = resp[brace_start:brace_end + 1]
            parsed = _json.loads(resp)

            state.user_goal = parsed.get('goal', state.user_query) or state.user_query
            raw_constraints = parsed.get('constraints', [])
            if raw_constraints and isinstance(raw_constraints, list):
                state.constraints = [str(c) for c in raw_constraints if c]
            # 将隐含需求也加入约束
            implicit = parsed.get('implicit_needs', '')
            if implicit and isinstance(implicit, str) and implicit.strip():
                state.constraints.append(f"隐含需求: {implicit.strip()}")

        except Exception as e:
            # 降级：直接使用原始输入作为目标
            self.logger.debug(f"[Orchestrator] Phase1 LLM目标提取失败，使用原始输入: {e}")
            state.user_goal = state.user_query

        # intent_result 中的 reason 也作为约束补充
        if hasattr(intent_result, 'reason') and intent_result.reason:
            state.constraints.append(intent_result.reason)

        self.logger.info(
            f"[Orchestrator] Phase1完成 | 目标={state.user_goal[:60]} | "
            f"约束={state.constraints} | 复杂度={state.complexity}"
        )

    # ================================================================
    # Phase 2: 规划
    # ================================================================

    def _phase_plan(
            self,
            state: TaskState,
            context: str,
            user_id: str,
            username: str,
    ):
        """Phase 2: 分解任务，决定委派策略"""

        # 让LLM分解任务（按用户查询过滤工具，只传入相关工具）
        available_tools = self._get_available_tools_summary(query=state.user_query)
        orchestrator_ctx = self.memory.get_context_for_sub_agent()

        prompt = f"""你是智能购物平台的任务规划专家。请将用户的购物任务分解为可执行的子任务。

用户任务：{state.user_query}
用户目标：{state.user_goal}
任务复杂度：{state.complexity}

对话上下文（最近）：{context[-800:] if context else '无'}
{f'已知信息：{orchestrator_ctx[:500]}' if orchestrator_ctx else ''}
当前用户：user_id={user_id}, username={username}

可用工具：
{available_tools}

请以JSON格式返回子任务列表：
```json
{{
  "sub_tasks": [
    {{
      "id": "t1",
      "description": "子任务描述",
      "type": "main|delegate",
      "delegation_reason": "parallelizable|context_overflow|off_topic|specialized|main_agent",
      "tools": ["工具名1"],
      "depends_on": [],
      "role": "子Agent角色描述（仅delegate类型需要）"
    }}
  ],
  "execution_order": [["t1", "t2"], ["t3"]]
}}
```

购物标准流程（规划时参考）：
1. 了解需求/偏好 → 2. 搜索/推荐商品 → 3. 用户选择 → 4. 确认下单信息（地址、支付） → 5. 创建订单

决策规则：
1. 简单的直接操作（单工具调用） → type="main"（自己做）
2. 可以并行且相互独立的子任务（如同时搜索不同类目商品） → type="delegate"
3. 会产生大量中间结果的子任务 → type="delegate"
4. 大部分任务应该type="main"（自己做更可靠）
5. execution_order：同一层内的任务可并行，层间串行
6. 下单操作必须由主Agent执行（type="main"），不可委派

只返回JSON。"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.3).strip()
            plan = self._parse_plan_response(resp)

            # 从配置读取下单工具集合（代码层保障，不依赖 LLM 遵守 Prompt 规则）
            order_creation_tools = set(
                self.config.get('security', {}).get('order_creation_tools', ['create_complex_order'])
            )

            for task_data in plan.get("sub_tasks", []):
                is_delegate = task_data.get("type") == "delegate"
                # L5 fix: delegate 类型的默认 delegation_reason 应为 PARALLELIZABLE，
                # 而非 MAIN_AGENT（语义矛盾）
                default_reason = (
                    DelegationReason.PARALLELIZABLE.value if is_delegate
                    else DelegationReason.MAIN_AGENT.value
                )
                sub_task = SubTask(
                    id=task_data.get("id", f"t{len(state.sub_tasks) + 1}"),
                    description=task_data.get("description", ""),
                    assigned_to="sub_agent" if is_delegate else "main",
                    delegation_reason=task_data.get("delegation_reason", default_reason),
                    depends_on=task_data.get("depends_on", []),
                    agent_role=task_data.get("role", ""),
                    agent_tools=task_data.get("tools", []),
                    max_retries=self.max_subtask_retries,
                    timeout=60.0,
                )
                # L2 fix: 代码层强制下单工具只能由主Agent执行，
                # Prompt 约束无法保证 LLM 100% 遵守，必须在代码层兜底
                if sub_task.assigned_to == "sub_agent" and set(sub_task.agent_tools) & order_creation_tools:
                    self.logger.warning(
                        f"[Orchestrator] 子任务 {sub_task.id} 包含下单工具 "
                        f"{set(sub_task.agent_tools) & order_creation_tools}，"
                        f"强制改为主Agent执行（不允许委派下单操作）"
                    )
                    sub_task.assigned_to = "main"
                    sub_task.delegation_reason = DelegationReason.MAIN_AGENT.value

                state.sub_tasks.append(sub_task)

            state.execution_order = plan.get(
                "execution_order",
                [[st.id for st in state.sub_tasks]]
            )

        except Exception as e:
            self.logger.error(f"[Orchestrator] 规划失败: {e}")
            # 降级：创建单个主Agent子任务
            state.sub_tasks = [SubTask(
                id="t1",
                description=state.user_query,
                assigned_to="main",
            )]
            state.execution_order = [["t1"]]

        self.logger.info(
            f"[Orchestrator] Phase2完成 | 子任务数={len(state.sub_tasks)} | "
            f"委派数={sum(1 for st in state.sub_tasks if st.assigned_to == 'sub_agent')}"
        )

    # ================================================================
    # Phase 3: 执行
    # ================================================================

    def _phase_execute(
            self,
            state: TaskState,
            context: str,
            long_term_context: str,
            user_id: str,
            username: str,
            verbose: bool,
    ):
        """Phase 3: 按执行顺序执行子任务"""

        for layer_idx, layer_task_ids in enumerate(state.execution_order):
            state.current_layer = layer_idx
            layer_tasks = [
                state.get_subtask(tid) for tid in layer_task_ids
                if state.get_subtask(tid)
            ]

            if not layer_tasks:
                continue

            # 分离主Agent任务和委派任务
            main_tasks = [t for t in layer_tasks if t.assigned_to == "main"]
            delegate_tasks = [t for t in layer_tasks if t.assigned_to == "sub_agent"]

            if verbose and layer_tasks:
                print(f"\n  📌 执行层 {layer_idx + 1}/{len(state.execution_order)} "
                      f"（{len(main_tasks)} 个自己做, {len(delegate_tasks)} 个委派）")

            # 先启动委派任务（并行）
            delegate_results = {}
            if delegate_tasks:
                # 为委派任务构建精简上下文
                for dt in delegate_tasks:
                    dt.agent_context = self._build_subtask_context(
                        dt, state, context, user_id, username
                    )
                    dt.mark_running()

                if verbose:
                    for dt in delegate_tasks:
                        print(f"    🤖 委派子Agent: {dt.description[:50]}")

                delegate_results = self.factory.execute_subtasks_parallel(delegate_tasks)

            # 然后执行主Agent任务（串行）
            for mt in main_tasks:
                mt.mark_running()
                if verbose:
                    print(f"    🔧 主Agent执行: {mt.description[:50]}")

                result = self._execute_main_task(
                    mt, state, context, long_term_context, user_id, username, verbose
                )
                mt.mark_completed(result)

                if verbose:
                    status = "✅" if result.success else "❌"
                    print(f"    {status} 完成: {result.summary[:60]}")

            # 处理委派结果
            for dt in delegate_tasks:
                if dt.id in delegate_results:
                    sub_result = delegate_results[dt.id]
                    # 检测 NEED_INPUT 标记
                    if (not sub_result.success
                            and sub_result.error
                            and sub_result.error.startswith("__NEED_INPUT__:")):
                        question = sub_result.error[len("__NEED_INPUT__:"):]
                        state.paused_subtasks[dt.id] = question
                        dt.mark_paused()
                        if verbose:
                            print(f"    \u23f8 子任务 {dt.id} 需要用户输入: {question[:60]}")
                    elif sub_result.success:
                        dt.mark_completed(sub_result)
                        if verbose:
                            print(f"    ✅ 子Agent完成: {sub_result.summary[:60]}")
                    else:
                        # 失败处理：重试 或 降级到主Agent
                        self._handle_subtask_failure(
                            dt, sub_result, state, context,
                            long_term_context, user_id, username, verbose
                        )

            # 若有 PAUSED 子任务，取第一个问题抛出给用户
            if state.paused_subtasks:
                first_question = next(iter(state.paused_subtasks.values()))
                raise NeedUserInputException(first_question)

            # 更新orchestrator记忆
            for t in layer_tasks:
                if t.result and t.result.success:
                    self.memory.extract_entities_from_step_result(t.result.summary)

    def _infer_subtask_intent(self, sub_task) -> tuple:
        """根据子任务描述推断意图类型、工具名和复杂度

        Returns:
            (IntentType, tool_name_or_None, TaskComplexity)
        """
        from .intent_utils import infer_intent_from_desc
        return infer_intent_from_desc(
            desc=sub_task.description.lower(),
            allowed_tools=sub_task.agent_tools or None,
        )

    def _execute_main_task(
            self,
            sub_task: SubTask,
            state: TaskState,
            context: str,
            long_term_context: str,
            user_id: str,
            username: str,
            verbose: bool,
    ) -> SubAgentResult:
        """主Agent直接执行子任务"""
        start_time = time.time()

        try:
            # 构建包含前置结果的上下文
            enriched_context = self._build_execution_context(state, context, user_id, username)

            if self.task_planner:
                inferred_type, inferred_tool, inferred_complexity = self._infer_subtask_intent(sub_task)
                sub_intent = IntentResult(
                    intent_type=inferred_type,
                    complexity=inferred_complexity,
                    tool_name=inferred_tool,
                )

                result_text = self.task_planner.execute(
                    user_query=sub_task.description,
                    intent=sub_intent,
                    context=enriched_context,
                    long_term_context=long_term_context,
                    verbose=False,  # 子任务不打印详细过程
                    on_step_complete=self.memory.extract_entities_from_step_result,
                )
            else:
                result_text = self.llm_client.generate(
                    prompt=f"任务：{sub_task.description}\n上下文：{enriched_context}\n请完成任务：",
                ).strip()

            return SubAgentResult(
                task_id=sub_task.id,
                success=True,
                summary=result_text,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return SubAgentResult(
                task_id=sub_task.id,
                success=False,
                summary="",
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _handle_subtask_failure(
            self,
            sub_task: SubTask,
            failed_result: SubAgentResult,
            state: TaskState,
            context: str,
            long_term_context: str,
            user_id: str,
            username: str,
            verbose: bool,
    ):
        """处理子任务失败

        企业级处理策略：
        1. 可重试 → 通过子Agent工厂重试
        2. 不可重试 → 降级到主Agent直接执行
        3. 主Agent也失败 → 标记失败，取消下游依赖
        """
        sub_task.mark_failed(failed_result.error or "子Agent执行失败", failed_result)

        # 策略1：重试（如果还有重试次数）
        if sub_task.can_retry and self.factory.circuit_breaker.allow_request():
            if verbose:
                print(f"    🔄 子Agent失败，重试中... (第{failed_result.retry_count + 1}次)")

            sub_task.mark_retrying()
            retry_result = self.factory.execute_subtask(sub_task)

            if retry_result.success:
                sub_task.mark_completed(retry_result)
                if verbose:
                    print(f"    ✅ 重试成功: {retry_result.summary[:60]}")
                return
            else:
                sub_task.mark_failed(retry_result.error or "重试失败", retry_result)

        # 策略2：降级到主Agent
        if verbose:
            print(f"    ⬇️ 降级到主Agent直接执行...")

        self.logger.info(
            f"[Orchestrator] 子任务 {sub_task.id} 降级到主Agent执行"
        )

        # 改为主Agent执行
        sub_task.assigned_to = "main"
        fallback_result = self._execute_main_task(
            sub_task, state, context, long_term_context,
            user_id, username, verbose,
        )

        if fallback_result.success:
            sub_task.mark_completed(fallback_result)
            if verbose:
                print(f"    ✅ 降级执行成功: {fallback_result.summary[:60]}")
        else:
            sub_task.mark_failed(fallback_result.error or "降级执行也失败")
            # 取消依赖此任务的后续子任务
            state.cancel_dependents(sub_task.id)
            if verbose:
                print(f"    ❌ 降级执行失败，已取消下游任务")

    # ================================================================
    # Phase 4: 整合
    # ================================================================

    def _phase_integrate(
            self,
            state: TaskState,
            user_query: str,
            context: str,
            verbose: bool,
    ):
        """Phase 4: 整合所有子任务结果"""

        results = state.get_completed_results()
        failed = state.get_failed_subtasks()

        if not results:
            state.integrated_result = "所有子任务执行失败"
            return

        # 汇总所有结果
        result_parts = []
        for st in state.sub_tasks:
            if st.result and st.result.success:
                result_parts.append(f"[{st.id}] {st.description}: {st.result.summary}")
            elif st.status == SubTaskStatus.FAILED:
                result_parts.append(
                    f"[{st.id}] {st.description}: ❌ 失败 - {st.result.error if st.result else '未知错误'}")
            elif st.status == SubTaskStatus.CANCELLED:
                result_parts.append(f"[{st.id}] {st.description}: ⏭ 已跳过")

        state.integrated_result = "\n".join(result_parts)

        # 质量验证：仅对明确调用下单工具的子任务检查是否返回了订单号
        # 通过 agent_tools 判断，避免把"下单前确认"类描述文字误判为"已执行下单"
        quality_warnings = []
        # 从配置读取下单工具集合，兜底使用实际存在的工具名
        order_creation_tools = set(
            self.config.get('security', {}).get(
                'order_creation_tools', ['create_complex_order']
            )
        )
        for st in state.sub_tasks:
            if st.result and st.result.success:
                result_text = st.result.summary
                is_order_task = bool(set(st.agent_tools or []) & order_creation_tools)
                if is_order_task:
                    has_order_id = bool(re.search(r'ORD-\d+', result_text))
                    if not has_order_id:
                        quality_warnings.append(
                            f"⚠️ 子任务[{st.id}]调用了下单工具，但响应中未检测到订单号(ORD-XXXX格式)"
                        )

        if quality_warnings:
            state.integrated_result += "\n\n[质量校验警告]\n" + "\n".join(quality_warnings)
            self.logger.warning(f"[Orchestrator] Phase4质量警告: {quality_warnings}")

        # 验证结果质量
        if failed and verbose:
            print(f"    ⚠️ {len(failed)} 个子任务失败，已用可用结果整合")

        self.logger.info(
            f"[Orchestrator] Phase4完成 | "
            f"成功={len(results)} 失败={len(failed)}"
        )

    # ================================================================
    # Phase 5: 交付
    # ================================================================

    def _phase_deliver(
            self,
            state: TaskState,
            user_query: str,
            context: str,
            long_term_context: str,
    ):
        """Phase 5: 生成最终回复"""

        # 检测执行结果中是否包含已创建的订单号
        order_ids = re.findall(r'ORD-\d+', state.integrated_result)

        if order_ids:
            # 订单已创建：不注入旧对话上下文，避免"请确认"类旧消息干扰 LLM
            context_section = ""
        else:
            context_section = f"\n对话上下文（最近）：{context[-500:]}" if context else ""

        prompt = f"""你是智能购物平台的购物助手。以下是你为用户完成购物任务的结果汇总。
请基于这些结果，用自然、清晰的语言回复用户。

用户原始请求：{user_query}

任务执行结果：
{state.integrated_result}
{context_section}

要求：
1. 用面向用户的自然语言回答，不要暴露内部任务编号（t1, t2等）
2. 推荐商品时标注商品ID和价格，格式：[P001] 商品名 ¥价格
3. 下单结果突出订单号、商品、金额、收货地址等关键信息
4. 如果有失败的子任务，诚实告知用户哪些没有完成
5. 简洁友好，避免过度营销话术"""

        try:
            state.final_response = self.llm_client.generate(
                prompt=prompt, temperature=0.5
            ).strip()
        except Exception as e:
            # 降级：直接用整合结果
            state.final_response = state.integrated_result

        # 更新orchestrator记忆
        self.memory.update_from_sub_agent_result(
            route="orchestrator",
            query=user_query,
            response=state.final_response,
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    def _build_subtask_context(
            self,
            sub_task: SubTask,
            state: TaskState,
            context: str,
            user_id: str,
            username: str,
    ) -> str:
        """为委派的子Agent构建精简上下文

        主Agent对子Agent的上下文进行信息压缩——
        只给子Agent完成任务所需的最少信息。
        """
        parts = []

        # 用户信息
        if user_id:
            parts.append(f"当前用户: user_id={user_id}, username={username}")

        # 前置子任务结果（只给结论）
        for dep_id in sub_task.depends_on:
            dep_task = state.get_subtask(dep_id)
            if dep_task and dep_task.result and dep_task.result.success:
                parts.append(f"前置任务[{dep_id}]结论: {dep_task.result.summary[:300]}")

        # Orchestrator记忆中的关键实体
        orchestrator_ctx = self.memory.get_context_for_sub_agent()
        if orchestrator_ctx:
            parts.append(f"已知信息:\n{orchestrator_ctx[:500]}")

        # 从主Agent上下文中截取最相关的部分
        if context:
            parts.append(f"对话上下文（最近）:\n{context[-500:]}")

        return "\n\n".join(parts)

    def _build_execution_context(
            self,
            state: TaskState,
            context: str,
            user_id: str,
            username: str,
    ) -> str:
        """构建主Agent执行子任务时的上下文"""
        parts = [context]

        # 注入已完成子任务的结果
        for st in state.sub_tasks:
            if st.result and st.result.success:
                parts.append(f"[已完成-{st.id}] {st.description}: {st.result.summary[:300]}")

        # 注入orchestrator记忆
        orchestrator_ctx = self.memory.get_context_for_sub_agent()
        if orchestrator_ctx:
            parts.append(f"[Orchestrator已知信息]\n{orchestrator_ctx}")

        return "\n".join(parts)

    def _get_available_tools_summary(self, query: str = "") -> str:
        """获取可用工具的简要列表（按查询上下文过滤，减少无关工具噪音）"""
        if self.mcp_manager:
            tools = self.mcp_manager.get_tools_for_context(
                intent_type='mcp_execute', query=query
            )
            lines = []
            for t in tools[:20]:
                name = t.get("name", "")
                desc = t.get("description", "")
                lines.append(f"- {name}: {desc}")
            return "\n".join(lines) if lines else "无可用工具"
        return "无可用工具"

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的规划JSON"""
        response = response.strip()

        # 提取JSON代码块
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_block:
            response = code_block.group(1).strip()

        # 提取第一个JSON对象
        brace_start = response.find('{')
        if brace_start >= 0:
            depth = 0
            in_string = False
            escape = False
            for i in range(brace_start, len(response)):
                c = response[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        response = response[brace_start:i + 1]
                        break

        return json.loads(response)

    def _fallback_response(
            self,
            user_query: str,
            context: str,
            long_term_context: str,
            error: str,
    ) -> str:
        """降级响应：当编排流程失败时直接回答"""
        self._last_request_failed = True
        self.logger.warning(f"[Orchestrator] 降级到直接回答: {error}")

        if self.task_planner:
            try:
                fallback_intent = IntentResult(
                    intent_type=IntentType.SIMPLE_CHAT,
                    complexity=TaskComplexity.SIMPLE,
                )
                return self.task_planner.execute(
                    user_query=user_query,
                    intent=fallback_intent,
                    context=context,
                    long_term_context=long_term_context,
                    verbose=False,
                )
            except Exception:
                pass

        return self.llm_client.generate(
            prompt=f"用户问题：{user_query}\n上下文：{context[-500:]}\n请回答：",
        ).strip()

    def _print_plan(self, state: TaskState):
        """打印执行计划"""
        main_count = sum(1 for st in state.sub_tasks if st.assigned_to == "main")
        delegate_count = sum(1 for st in state.sub_tasks if st.assigned_to == "sub_agent")
        print(f"\n📝 执行计划（{len(state.sub_tasks)} 个子任务: "
              f"{main_count} 自己做, {delegate_count} 委派）:")
        for st in state.sub_tasks:
            icon = "🔧" if st.assigned_to == "main" else "🤖"
            tools = f" [{','.join(st.agent_tools)}]" if st.agent_tools else ""
            deps = f" (依赖:{','.join(st.depends_on)})" if st.depends_on else ""
            print(f"  {icon} {st.id}: {st.description}{tools}{deps}")

        print(f"\n  执行顺序: {' → '.join([str(layer) for layer in state.execution_order])}")
