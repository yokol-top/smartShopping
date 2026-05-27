"""
任务规划模块 (Task Planner)

根据意图识别的复杂度，采用不同的规划策略：
- 简单任务 (SIMPLE): 单意图且步骤在4步以内，使用ReAct循环执行
- 中等任务 (MEDIUM): 单意图4-7步 / ≤3个意图且4-7步骤，使用Plan and Execute规划和执行
- 复杂任务 (COMPLEX): 使用分层规划（先规划几个大的阶段，然后每个阶段按照中等任务处理）

规划器接收评估器的反馈，根据反馈进行重新规划或终止。
"""
import json
import logging
import random
import re
from typing import Dict, Any, List, Optional

from agent.intent_recognizer import IntentResult, IntentType, TaskComplexity
from observability import get_tracer
from .task_evaluator import TaskEvaluator, EvalSeverity


class PlanStep:
    """执行计划中的单个步骤"""

    def __init__(self, step_id: int, description: str, action_type: str,
                 action_input: Dict[str, Any] = None, depends_on: List[int] = None):
        self.step_id = step_id
        self.description = description
        self.action_type = action_type
        self.action_input = action_input or {}
        self.depends_on = depends_on or []
        self.result: Optional[str] = None
        self.status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action_type": self.action_type,
            "action_input": self.action_input,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
        }


class TaskPlanner:
    """
    任务规划器

    职责：
    1. 根据意图识别结果选择规划策略
    2. 生成执行计划（步骤列表）
    3. 执行计划（每步使用ReAct）
    4. 接收评估器反馈，决定是否重新规划
    """

    def __init__(self, config: Dict[str, Any], llm_client=None, mcp_manager=None,
                 rag_engine=None, evaluator: TaskEvaluator = None,
                 context_manager=None, tool_manager=None,
                 logger: logging.Logger = None):
        self.config = config
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.rag_engine = rag_engine
        self.evaluator = evaluator
        self.context_manager = context_manager
        self.tool_manager = tool_manager  # 统一工具管理器（本地+MCP）
        self.logger = logger or logging.getLogger(__name__)

        react_config = config.get('react', {})
        self.react_max_iterations = react_config.get('max_iterations', 5)
        self.react_temperature = react_config.get('temperature', 0.3)

        planner_config = config.get('planner', {})
        self.max_replan_attempts = planner_config.get('max_replan_attempts', 2)

        # 评估阶段开关（从配置读取，允许关闭不必要的LLM评估调用）
        eval_config = config.get('evaluator', {})
        self.eval_pre_enabled = eval_config.get('pre_evaluation', True)
        self.eval_mid_enabled = eval_config.get('mid_evaluation', True)
        self.eval_post_enabled = eval_config.get('post_evaluation', True)

        self._tool_executor = None  # 可选的子Agent工具执行回调
        self._on_step_complete = None  # 可选的步骤完成回调
        self.logger.info(
            f"TaskPlanner 初始化完成 | 评估开关: pre={self.eval_pre_enabled} "
            f"mid={self.eval_mid_enabled} post={self.eval_post_enabled}"
        )

    # ================================================================
    # 主入口
    # ================================================================
    def execute(self, user_query: str, intent: IntentResult, context: str = "",
                long_term_context: str = "", verbose: bool = True,
                tool_executor=None, on_step_complete=None) -> str:
        """根据意图识别结果执行任务

        Args:
            tool_executor: 可选的工具执行回调，签名为
                (tool_name, step_desc, user_query, context, history) -> Optional[str]
                返回None表示该工具不由外部执行，回退到内置MCP调用。
                用于将子Agent作为任务执行节点集成到规划流程中。
            on_step_complete: 可选的步骤完成回调，签名为 (step_result: str) -> None
                每个步骤执行完成后调用，用于从结果中提取实体信息（ID等）
                存入orchestrator记忆，避免后续步骤重复查询。
        """
        self._tool_executor = tool_executor
        self._on_step_complete = on_step_complete
        tracer = get_tracer()

        with tracer.start_span("task.execute", {
            "task.intent_type": intent.intent_type.value,
            "task.complexity": intent.complexity.value,
            "task.user_query": user_query,
        }):
            intent_type = intent.intent_type
            complexity = intent.complexity

            if intent_type == IntentType.GREETING:
                return self._handle_greeting()

            if intent_type == IntentType.MCP_ASK_INFO:
                tool_name = intent.tool_name or self._infer_tool_name(user_query, context)
                return self._handle_ask_tool_info(tool_name, verbose)

            if complexity == TaskComplexity.SIMPLE:
                return self._execute_simple(user_query, intent, context, long_term_context, verbose)
            elif complexity == TaskComplexity.MEDIUM:
                return self._execute_plan_and_execute(user_query, intent, context, long_term_context, verbose, "medium")
            else:
                # 复杂任务使用分层规划
                return self._execute_hierarchical(user_query, intent, context, long_term_context, verbose)

    # ================================================================
    # 简单任务：直接执行
    # ================================================================
    def _execute_simple(self, user_query, intent, context, long_term_context, verbose):
        """简单任务：根据意图类型直接执行"""
        self.logger.info(f"简单任务执行 - 意图: {intent.intent_type.value}")
        if verbose:
            print(f"\n⚡ 简单任务 - 直接执行\n")

        if intent.intent_type == IntentType.SIMPLE_CHAT:
            return self._direct_chat(user_query, context, long_term_context)
        elif intent.intent_type == IntentType.RAG_SIMPLE:
            return self._rag_query(user_query, context, long_term_context, advanced=False, verbose=verbose)
        elif intent.intent_type == IntentType.RAG_ADVANCED:
            return self._rag_query(user_query, context, long_term_context, advanced=True, verbose=verbose)
        elif intent.intent_type == IntentType.MCP_EXECUTE:
            return self._react_loop(user_query, intent, context, long_term_context, verbose)
        return self._direct_chat(user_query, context, long_term_context)

    # ================================================================
    # 中等/复杂任务：Plan and Execute
    # ================================================================
    def _execute_plan_and_execute(self, user_query, intent, context, long_term_context, verbose, detail_level):
        """Plan and Execute 流程（中等/复杂任务共用）"""
        tracer = get_tracer()
        self.logger.info(f"{detail_level}任务 - Plan and Execute 模式")
        if verbose:
            label = "📋 中等任务" if detail_level == "medium" else "🧠 复杂任务"
            print(f"\n{label} - Plan and Execute 模式\n")

        replan_count = 0
        execution_history = []  # 累计所有已完成步骤（跨重规划保留）

        while replan_count <= self.max_replan_attempts:
            # Step 1: 生成计划（携带已完成步骤上下文）
            with tracer.start_span("task.generate_plan", {"task.detail_level": detail_level}):
                plan = self._generate_plan(
                    user_query, intent, context, long_term_context,
                    detail_level, completed_history=execution_history,
                )
                tracer.set_span_attributes({"task.plan_steps": len(plan)})
            if verbose:
                print(f"📝 生成执行计划（{len(plan)} 步）:")
                for s in plan:
                    print(f"   {s.step_id}. [{s.action_type}] {s.description}")
                print()

            # Step 2: 执行前评估（可通过配置关闭以节省LLM调用）
            if self.evaluator and self.eval_pre_enabled:
                # 将已完成步骤传给评估器，避免评估器认为缺少已完成的步骤
                pre_eval = self.evaluator.pre_evaluate(
                    user_query, [s.to_dict() for s in plan], context,
                    completed_history=execution_history,
                )
                if verbose:
                    self._print_eval("执行前", pre_eval)
                if pre_eval.should_replan:
                    replan_count += 1
                    if replan_count > self.max_replan_attempts:
                        break
                    context += f"\n[评估反馈] {self._sanitize_feedback(pre_eval.message, pre_eval.suggestions)}"
                    if verbose:
                        print(f"🔄 重新规划 ({replan_count}/{self.max_replan_attempts})...\n")
                    continue

            # Step 3: 逐步执行
            completed_steps = []
            should_replan = False

            for i, step in enumerate(plan):
                step.status = "running"
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"🔄 执行步骤 {step.step_id}/{len(plan)}: {step.description}")
                    print(f"{'='*50}\n")

                with tracer.start_span(f"task.execute_step.{step.step_id}", {
                    "task.step_id": step.step_id,
                    "task.step_description": step.description,
                    "task.step_action_type": step.action_type,
                }):
                    step_result = self._execute_step(step, user_query, context, long_term_context, execution_history, verbose)
                    step.result = step_result
                    step.status = "completed"
                    tracer.set_span_attributes({
                        "task.step_result_preview": step_result[:300] if step_result else "",
                        "task.step_status": "completed",
                    })

                record = {"step_id": step.step_id, "description": step.description,
                          "action_type": step.action_type, "result": step_result}
                execution_history.append(record)
                completed_steps.append(record)

                # 从步骤结果中提取实体信息（ID等）存入orchestrator记忆
                if self._on_step_complete and step_result:
                    try:
                        self._on_step_complete(step_result)
                    except Exception as e:
                        self.logger.debug(f"步骤完成回调异常: {e}")

                if verbose:
                    self.logger.debug(f"   ✅ 结果: {step_result}\n")

                # 执行中评估（可通过配置关闭以节省LLM调用）
                if self.evaluator and self.eval_mid_enabled:
                    remaining = [s.to_dict() for s in plan[i+1:]]
                    mid_eval = self.evaluator.mid_evaluate(user_query, record, step_result, completed_steps, remaining)
                    if verbose:
                        self._print_eval("执行中", mid_eval)
                    if mid_eval.should_replan:
                        # 检查是否是缺少用户输入导致的失败（用户未提供必需参数）
                        if self._is_missing_user_input_error(step_result):
                            self.logger.info("检测到缺少用户必需参数，向用户请求补充信息")
                            if verbose:
                                print("\n❓ 检测到缺少必要信息，需要您补充...\n")
                            return self._generate_ask_user_response(
                                user_query, execution_history, step_result,
                                step.description, context, long_term_context,
                            )
                        replan_count += 1
                        if replan_count > self.max_replan_attempts:
                            break
                        context += f"\n[执行中反馈] 步骤{step.step_id}: {self._sanitize_feedback(mid_eval.message, mid_eval.suggestions)}"
                        should_replan = True
                        break

            if should_replan:
                if verbose:
                    print(f"\n🔄 根据评估反馈重新规划...\n")
                continue

            # Step 4: 生成最终答案
            with tracer.start_span("task.generate_final_answer"):
                final_answer = self._generate_final_answer(user_query, execution_history, context, long_term_context)

            # Step 5: 执行后评估（可通过配置关闭以节省LLM调用）
            if self.evaluator and self.eval_post_enabled:
                post_eval = self.evaluator.post_evaluate(user_query, final_answer, execution_history, context)
                if verbose:
                    self._print_eval("执行后", post_eval)
                if post_eval.should_replan:
                    replan_count += 1
                    if replan_count <= self.max_replan_attempts:
                        context += f"\n[执行后反馈] {self._sanitize_feedback(post_eval.message, post_eval.suggestions)}"
                        continue
                if post_eval.severity == EvalSeverity.REMINDER and post_eval.suggestions:
                    final_answer += "\n\n💡 " + "；".join(post_eval.suggestions)

            return final_answer

        # 超过最大重试
        self.logger.warning("超过最大重新规划次数")
        return self._generate_final_answer(user_query, execution_history, context, long_term_context)

    # ================================================================
    # 复杂任务：分层规划
    # ================================================================
    def _execute_hierarchical(self, user_query, intent, context, long_term_context, verbose):
        """复杂任务的分层规划执行

        流程：
        1. 先规划几个大的阶段（高层计划）
        2. 每个阶段按照中等任务的Plan and Execute处理
        3. 汇总所有阶段结果生成最终答案
        """
        tracer = get_tracer()
        self.logger.info("复杂任务 - 分层规划模式")
        if verbose:
            print(f"\n🧠 复杂任务 - 分层规划模式\n")

        # Step 1: 生成高层阶段计划
        with tracer.start_span("task.hierarchical.generate_phases"):
            phases = self._generate_high_level_phases(user_query, intent, context, long_term_context)
            tracer.set_span_attributes({"task.phases_count": len(phases)})

        if verbose:
            print(f"📋 分层计划（{len(phases)} 个阶段）:")
            for i, phase in enumerate(phases, 1):
                print(f"   阶段 {i}: {phase['description']}")
            print()

        # Step 2: 逐阶段执行（每个阶段视为中等任务）
        all_phase_results = []
        accumulated_context = context

        for i, phase in enumerate(phases):
            phase_desc = phase['description']
            if verbose:
                print(f"\n{'='*60}")
                print(f"🔷 阶段 {i+1}/{len(phases)}: {phase_desc}")
                print(f"{'='*60}\n")

            with tracer.start_span(f"task.hierarchical.phase.{i+1}", {
                "task.phase_id": i + 1,
                "task.phase_description": phase_desc,
            }):
                # 构建阶段查询，包含已完成阶段的摘要，避免重复执行
                completed_summary = ""
                if all_phase_results:
                    completed_parts = ["\n[已完成的阶段（不要重复执行）]"]
                    for pr in all_phase_results:
                        completed_parts.append(
                            f"- 阶段{pr['phase_id']}({pr['description']}): {pr['result'][:300]}"
                        )
                    completed_summary = "\n".join(completed_parts)

                phase_query = (
                    f"[阶段 {i+1}] {phase_desc}\n\n"
                    f"（原始用户请求：{user_query}）"
                    f"{completed_summary}"
                )
                phase_result = self._execute_plan_and_execute(
                    phase_query, intent, accumulated_context, long_term_context,
                    verbose, "medium"
                )

                all_phase_results.append({
                    "phase_id": i + 1,
                    "description": phase_desc,
                    "result": phase_result,
                })

                # 将阶段结果追加到上下文，供后续阶段参考
                accumulated_context += f"\n[阶段{i+1}结果] {phase_desc}: {phase_result[:500]}"

                tracer.set_span_attributes({
                    "task.phase_result_preview": phase_result[:300],
                })

        # Step 3: 汇总所有阶段结果
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 汇总所有阶段结果")
            print(f"{'='*60}\n")

        with tracer.start_span("task.hierarchical.summarize"):
            final_answer = self._summarize_phases(user_query, all_phase_results, context, long_term_context)

        return final_answer

    def _generate_high_level_phases(self, user_query, intent, context, long_term_context):
        """生成高层阶段计划"""
        prompt_parts = []
        if long_term_context:
            prompt_parts.append(long_term_context)
        prompt_parts.append(f"""你是一个任务规划专家。用户的请求比较复杂，请将其拆分为2-4个高层阶段。

用户请求: {user_query}

对话上下文:
{context[:500]}

要求：
1. 每个阶段是一个独立的子目标
2. 阶段之间有清晰的先后顺序
3. 每个阶段应该是一个可以用4-7步完成的中等任务

返回JSON数组格式:
[{{"phase_id": 1, "description": "阶段描述"}}]

只返回JSON数组。""")

        try:
            response = self.llm_client.generate(
                prompt="\n".join(prompt_parts), temperature=self.react_temperature
            )
            phases = self._parse_json_safe(response, context_hint="high_level_phases")
            return phases if phases else [{"phase_id": 1, "description": user_query}]
        except Exception as e:
            self.logger.error(f"生成高层阶段失败: {e}")
            return [{"phase_id": 1, "description": user_query}]

    def _summarize_phases(self, user_query, phase_results, context, long_term_context):
        """汇总所有阶段结果生成最终答案"""
        phases_summary = ""
        for pr in phase_results:
            result_preview = pr['result'][:500] if pr['result'] else '未完成'
            phases_summary += f"阶段{pr['phase_id']}（{pr['description']}）:\n{result_preview}\n\n"

        parts = []
        if long_term_context:
            parts.append(long_term_context)
        parts.append(f"""用户请求: {user_query}

各阶段执行结果:
{phases_summary}

请基于以上所有阶段的执行结果，为用户提供完整、准确、有条理的最终回答。""")

        try:
            return self.llm_client.generate(prompt="\n".join(parts), temperature=0.5).strip()
        except Exception as e:
            self.logger.error(f"汇总阶段结果失败: {e}")
            return "\n\n".join(f"阶段{pr['phase_id']}: {pr['result']}" for pr in phase_results)

    # ================================================================
    # 计划生成
    # ================================================================
    def _generate_plan(self, user_query, intent, context, long_term_context,
                        detail_level, completed_history: list = None):
        """使用LLM生成执行计划

        Args:
            completed_history: 已完成的步骤历史（重规划时传入，避免重复执行）
        """
        # 上下文工程：懒加载工具描述（仅名称+描述，不含schema）
        tools_desc = self._get_tools_desc(brief=True)
        # 上下文工程：动态预算分配 + 分区预算管理
        if self.context_manager:
            active = ['short_term_memory', 'tools', 'planning_steps', 'tool_results']
            if long_term_context:
                active.append('long_term_memory')
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section('short_term_memory', context)
            long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)
            tools_desc = self.context_manager.manage_section('tools', tools_desc)
        detail_map = {
            "medium": "将任务拆分为4-7个关键步骤，每步目标明确。注意步骤间的依赖关系。",
            "complex": "将任务精细化拆分为多个原子步骤（7步以上），每步只做一件事，步骤间需注明依赖关系。"
        }

        # 构建已完成步骤上下文（重规划时用）——区分成功和失败
        completed_context = ""
        if completed_history:
            succeeded = []
            failed_info = []
            for h in completed_history:
                result_preview = h['result'][:200] if h.get('result') else ''
                if '失败' in result_preview or '错误' in result_preview or 'error' in result_preview.lower() or 'validation error' in result_preview.lower():
                    failed_info.append(
                        f"- 步骤{h['step_id']} [{h['action_type']}] {h['description']} => 失败: {result_preview}"
                    )
                else:
                    succeeded.append(
                        f"- 步骤{h['step_id']} [{h['action_type']}] {h['description']} => 结果: {result_preview}"
                    )
            completed_context = "\n".join(succeeded) if succeeded else ""

        prompt_parts = []
        if long_term_context:
            prompt_parts.append(long_term_context)
            prompt_parts.append("")

        # 已完成步骤提示
        replan_instruction = ""
        failed_info_ref = failed_info if completed_history else []
        if completed_context:
            replan_instruction = f"""\n【重要】以下步骤已经成功执行，绝对不要重复规划这些步骤：
{completed_context}

请只规划剩余未完成的步骤。必须直接引用上述已完成步骤的结果（如用户ID、商品ID、地址ID等）。
"""
        if failed_info_ref:
            replan_instruction += f"""\n以下步骤之前执行失败，请在新计划中修正：
{chr(10).join(failed_info_ref)}
"""

        prompt_parts.append(f"""你是一个任务规划专家。请根据用户请求生成执行计划。

用户请求: {user_query}

对话上下文:
{context}
{replan_instruction}
可用工具:
{tools_desc}

拆分要求: {detail_map.get(detail_level, detail_map['medium'])}

请生成JSON格式的执行计划:
[
  {{"step_id": 1, "description": "步骤描述", "action_type": "search_knowledge/call_mcp_tool/generate_answer", "action_input": {{}}, "depends_on": []}}
]

action_type说明:
- search_knowledge: 从知识库检索公司内部文档/知识（注意：不是用来查询工具执行结果的，前置步骤的结果会自动传递到后续步骤）
  action_input: {{"query": "搜索关键词"}}
- call_mcp_tool: 调用MCP工具（必须使用上方“可用工具”列表中的工具名）
  action_input: {{"tool_name": "工具名", "parameters": {{...}}}}
- generate_answer: 基于已有信息生成回答
  action_input: {{"prompt_hint": "回答要点"}}

规则:
1. 最后一步应该是generate_answer，用于整合结果
2. 每步的action_input必须包含足够参数
3. 依赖前置步骤结果的在depends_on中标注
4. 前置步骤返回的ID、数据等会自动传递给后续步骤，不需要单独规划“获取ID”的步骤
5. 只使用可用工具列表中存在的工具

只返回JSON数组。""")

        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=self.react_temperature)
            plan_data = self._parse_json_safe(response, context_hint="generate_plan")

            # 获取所有可用工具名集合，用于验证
            available_tools = set()
            if self.mcp_manager:
                for tool in self.mcp_manager.get_available_tools(use_cache=True):
                    available_tools.add(tool['name'])

            steps = []
            for item in plan_data:
                action_type = item.get('action_type', 'generate_answer')
                tool_name = item.get('action_input', {}).get('tool_name', '')

                # 过滤掉使用不存在工具的步骤
                if action_type == 'call_mcp_tool' and tool_name and available_tools and tool_name not in available_tools:
                    self.logger.warning(f"计划中引用了不存在的工具 '{tool_name}'，跳过该步骤: {item.get('description', '')}")
                    continue

                steps.append(PlanStep(
                    step_id=item.get('step_id', len(steps) + 1),
                    description=item.get('description', ''),
                    action_type=action_type,
                    action_input=item.get('action_input', {}),
                    depends_on=item.get('depends_on', []),
                ))

            # 重新编号步骤ID（过滤后可能不连续）
            for i, step in enumerate(steps, 1):
                step.step_id = i

            return steps
        except Exception as e:
            self.logger.error(f"生成计划失败: {e}")
            return [PlanStep(step_id=1, description="直接回答用户问题", action_type="generate_answer",
                             action_input={"prompt_hint": "直接回答"})]

    # ================================================================
    # ReAct循环（简单任务使用）
    # ================================================================
    def _react_loop(self, user_query, intent, context, long_term_context, verbose):
        """简单任务的ReAct循环执行"""
        tracer = get_tracer()
        self.logger.info("ReAct单步执行")
        # 选择性懒加载：用 brief 模式列出所有工具，仅为意图识别出的目标工具加载完整 schema
        # 这样既能让 LLM 知道所有可用工具，又能正确填充目标工具的参数，同时大幅节省上下文空间
        target_tool = intent.tool_name if intent else None
        tools_desc = self._get_tools_desc(brief=True, target_tool=target_tool)
        # 上下文工程：动态预算分配 + 分区预算管理
        if self.context_manager:
            active = ['short_term_memory', 'tools', 'tool_results']
            if long_term_context:
                active.append('long_term_memory')
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section('short_term_memory', context)
            long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)
            tools_desc = self.context_manager.manage_section('tools', tools_desc)
        thoughts_actions = []

        for iteration in range(self.react_max_iterations):
            if verbose:
                print(f"\n🔄 第 {iteration + 1} 轮思考\n")

            with tracer.start_span(f"task.react_iteration.{iteration + 1}", {
                "react.iteration": iteration + 1,
            }):
                history = self._format_react_history(thoughts_actions)
                # 上下文工程：管理工具执行结果预算
                if self.context_manager and history:
                    history = self.context_manager.manage_section('tool_results', history)
                prompt_parts = []
                if long_term_context:
                    prompt_parts.append(long_term_context)
                    prompt_parts.append("")
                prompt_parts.append(f"""你是一个能够使用工具完成任务的AI助手。

用户任务: {user_query}

对话历史:
{context}

可用工具:
{tools_desc}

已完成的步骤:
{history if history else "（尚未开始）"}

**⚠️ 关键规则（必须严格遵守）**:
1. 先仔细检查"已完成的步骤"中的观察结果。如果观察结果已包含"成功"、"完成"等信息，说明任务已完成，必须立即使用 Action: finish 返回结果。
2. 绝对不要重复执行已经成功的操作。如果某个工具调用已经成功返回结果，不要再次调用同一工具。
3. 每一轮的Thought中必须先总结已完成步骤的结果，再决定下一步。
4. 如果缺少必要参数，请不要自行补充，应该立刻终止执行，然后提示用户补充。

请严格按照以下格式回复：

Thought: [先总结已完成步骤的结果，再分析是否需要继续]
Action: [search_knowledge / call_mcp_tool / finish]
Action Input: [JSON格式参数]

Action Input格式：
- search_knowledge: {{"query": "搜索关键词"}}
- call_mcp_tool: {{"tool_name": "工具名", "parameters": {{具体参数}}}}
- finish: {{"answer": "最终答案"}}

如果任务已完成、缺少必要参数，使用 Action: finish。""")

                prompt = "\n".join(prompt_parts)
                react_response = self.llm_client.generate(prompt=prompt, temperature=self.react_temperature)
                self.logger.info(f"[ReAct] LLM原始响应 (前500字符):\n{react_response[:500] if react_response else '(空)'}")
                thought, action, action_input = self._parse_react_response(react_response)

                tracer.set_span_attributes({
                    "react.thought": thought,
                    "react.action": action,
                })

                if verbose:
                    print(f"💭 思考: {thought}")
                    print(f"⚡ 行动: {action}")
                    print(f"📥 输入: {action_input}\n")

                thoughts_actions.append({'thought': thought, 'action': action, 'action_input': action_input})

                if action == "finish":
                    return action_input.get('answer', '任务已完成')

                # 防重复保护：检测是否与之前已成功的步骤完全相同
                if self._is_duplicate_action(action, action_input, thoughts_actions[:-1]):
                    self.logger.warning("检测到重复操作，强制结束")
                    if verbose:
                        print("⚠️ 检测到重复操作，自动结束任务\n")
                    return self._summarize_history(user_query, thoughts_actions[:-1])

                if action == "search_knowledge":
                    observation = self._exec_rag_search(action_input.get('query', user_query), context)
                elif action == "call_mcp_tool":
                    observation = self._exec_mcp_call(action_input.get('tool_name', ''),
                                                       action_input.get('parameters', {}))
                else:
                    observation = f"未知动作: {action}"
                    # 连续未知动作计数，超过2次强制终止
                    consecutive_unknown = sum(
                        1 for ta in thoughts_actions
                        if ta.get('observation', '').startswith('未知动作')
                    )
                    if consecutive_unknown >= 2:
                        self.logger.warning(f"连续 {consecutive_unknown} 次未知动作，强制终止 ReAct")
                        if verbose:
                            print("⚠️ 多次调用了不存在的工具，自动结束任务\n")
                        return self._summarize_history(user_query, thoughts_actions)

                thoughts_actions[-1]['observation'] = observation
                tracer.set_span_attributes({"react.observation_preview": observation[:300]})

                if verbose:
                    preview = observation[:200] + ('...' if len(observation) > 200 else '')
                    print(f"👀 观察: {preview}\n")

        # 达到最大迭代，总结返回
        return self._summarize_history(user_query, thoughts_actions)

    # ================================================================
    # 单步执行
    # ================================================================
    def _execute_step(self, step, user_query, context, long_term_context, execution_history, verbose):
        """执行单个计划步骤"""
        action_type = step.action_type
        action_input = step.action_input

        if action_type == "search_knowledge":
            return self._exec_rag_search(action_input.get('query', user_query), context)
        elif action_type == "call_mcp_tool":
            tool_name = action_input.get('tool_name', '')

            # 优先尝试委托子Agent执行（子Agent作为任务执行节点）
            if self._tool_executor:
                delegated = self._tool_executor(
                    tool_name, step.description, user_query, context, execution_history
                )
                if delegated is not None:
                    self.logger.info(f"[TaskPlanner] 步骤已委托子Agent执行: {tool_name}")
                    return delegated

            parameters = action_input.get('parameters', {})
            # 必须动态提取参数的情况：
            # 1. 参数为空或含占位符
            # 2. 有前置步骤结果（参数可能依赖动态值如user_id）
            # 3. 懒加载模式：计划生成时LLM未见schema，参数不可靠
            has_prior_results = bool(execution_history)
            lazy_loading = bool(self.context_manager)
            if not parameters or self._has_placeholder(parameters) or has_prior_results or lazy_loading:
                return self._extract_and_call_mcp(tool_name, step.description, user_query, context, long_term_context, execution_history, verbose)
            return self._exec_mcp_call(tool_name, parameters)
        elif action_type == "generate_answer":
            return self._gen_step_answer(user_query, step.description, execution_history, context, long_term_context)
        else:
            return f"未知的action_type: {action_type}"

    def _extract_and_call_mcp(self, tool_name, step_description, user_query, context, long_term_context, execution_history, verbose):
        """使用LLM提取参数并调用MCP工具（懒加载：此时才加载完整schema）"""
        # 上下文工程：懒加载第二阶段 - 仅在需要填充参数时才加载完整 Input Schema
        schema = self._get_tool_schema(tool_name)
        if not schema:
            return f"未找到工具 {tool_name} 的schema"

        schema_json = json.dumps(self._compress_schema(schema), ensure_ascii=False)
        hist = "\n".join(f"- {h['description']}: {h.get('result', '')[:500]}" for h in execution_history)

        # 上下文工程：动态预算分配 + 分区预算管理
        if self.context_manager:
            active = ['short_term_memory', 'tool_results']
            if long_term_context:
                active.append('long_term_memory')
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section('short_term_memory', context)
            long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)
            if hist:
                hist = self.context_manager.manage_section('tool_results', hist)

        prompt_parts = []
        if long_term_context:
            prompt_parts.append(long_term_context)
        prompt_parts.append(f"""你是一个工具调用助手。请根据当前步骤的目标，从上下文中提取工具参数。

当前步骤目标: {step_description}

用户请求: {user_query}
对话历史: {context}
前置步骤结果:
{hist if hist else "（无）"}

工具名: {tool_name}
工具参数结构（JSON Schema）:
{schema_json}

**关键规则**:
1. 只提取与"当前步骤目标"相关的参数。用户请求可能包含多个操作，但本次调用只负责当前步骤描述的那一个操作，与当前步骤无关的字段设为null或不填。
2. 严格按照schema结构提取参数，必需参数必须提供。
3. 如果前置步骤结果中包含了本次调用所需的动态值（如用户ID、订单号等），必须使用前置步骤返回的实际值，不要编造或使用默认值。
4. 参数优先级：前置步骤结果 > 用户请求中的信息 > 默认值。
5. 根据上下文智能推断操作类型：步骤描述仅供参考。如果schema中包含操作类型字段（如action: add/update/delete），需根据实际情况选择——前置步骤刚创建了新实体时，后续为该实体新增数据应使用"add"而非"update"。
6. 如果某个必需的ID字段在上下文中不存在（如刚创建的实体还没有子资源ID），说明应使用创建/新增操作而非更新操作。

只返回JSON。""")

        try:
            param_json = self.llm_client.generate(prompt="\n".join(prompt_parts), temperature=0.3)
            parameters = json.loads(self._clean_json(param_json))
            if verbose:
                print(f"📋 提取参数: {json.dumps(parameters, ensure_ascii=False, indent=2)}\n")
            return self._exec_mcp_call(tool_name, parameters)
        except Exception as e:
            return f"参数提取失败: {e}"

    # ================================================================
    # 基础执行方法
    # ================================================================
    def _direct_chat(self, user_query, context, long_term_context=""):
        """直接对话"""
        # 上下文工程：动态预算分配 + 分区预算管理（无RAG/工具，预算释放给对话分区）
        if self.context_manager:
            active = ['short_term_memory']
            if long_term_context:
                active.append('long_term_memory')
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section('short_term_memory', context)
            long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)
        parts = []
        if long_term_context:
            parts.append(long_term_context)
            parts.append("")
        parts.append(f"对话历史：\n{context}\n用户输入: {user_query}\n\n请根据提供的上下文，结合用户最新输入的信息，推断用户的意图，然后简洁、准确地回答用户的最新问题")
        try:
            return self.llm_client.generate(prompt="\n".join(parts)).strip()
        except Exception as e:
            self.logger.error(f"直接对话失败: {e}")
            return "抱歉，我遇到了一些问题。请稍后再试。"

    def _rag_query(self, user_query, context, long_term_context, advanced=False, verbose=True):
        """RAG检索并回答"""
        if verbose:
            print("🚀 使用高级RAG技术处理查询...\n" if advanced else "🔎 正在搜索知识库...\n")

        try:
            results = self.rag_engine.retrieve(query=user_query, context=context,
                                                top_k=5 if advanced else 3, use_advanced=advanced)
            if not results:
                if verbose:
                    print("⚠️  知识库中未找到相关信息\n")
                return self._direct_chat(user_query, context, long_term_context)

            if verbose:
                print(f"✅ 找到 {len(results)} 个相关文档\n")

            retrieved_ctx = "\n\n".join([
                f"[文档 {i+1}]{' (相关度: ' + str(round(r.get('score', 0), 2)) + ')' if advanced else ''}\n{r['document']}"
                for i, r in enumerate(results)
            ])

            # 上下文工程：动态预算分配 + 分区预算管理（无工具，预算释放给RAG和对话分区）
            if self.context_manager:
                active = ['rag_results', 'short_term_memory']
                if long_term_context:
                    active.append('long_term_memory')
                self.context_manager.set_active_sections(active)
                retrieved_ctx = self.context_manager.manage_section('rag_results', retrieved_ctx)
                context = self.context_manager.manage_section('short_term_memory', context)
                long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)

            parts = []
            if long_term_context:
                parts.append(long_term_context)
                parts.append("")
            instruction = '请综合多个文档信息，提供详细、准确、有条理的回答。' if advanced else '请基于上述知识库信息回答用户的问题。如果信息不足，请明确说明。'
            parts.append(f"检索到的知识库信息：\n{retrieved_ctx}\n\n对话历史：\n{context}\n\n{instruction}")

            answer = self.llm_client.generate(prompt="\n".join(parts)).strip()

            # Self-fix验证（仅高级模式）
            if advanced and self.config.get('../rag', {}).get('self_fix', {}).get('enabled', True):
                if verbose:
                    print("🔍 正在验证答案质量...\n")
                contexts = [r['document'] for r in results]
                fix_result = self.rag_engine.self_fix.verify_and_fix(query=user_query, answer=answer, context=contexts)
                if fix_result['fixed']:
                    if verbose:
                        print(f"✨ 答案已优化（迭代 {fix_result['iterations']} 次）\n")
                    answer = fix_result['answer']
                elif verbose:
                    print("✅ 答案质量验证通过\n")

            return answer
        except Exception as e:
            self.logger.error(f"RAG处理失败: {e}")
            return "抱歉，搜索知识库时遇到了问题。"

    def _exec_rag_search(self, query, context):
        """执行RAG搜索"""
        try:
            results = self.rag_engine.retrieve(query=query, context=context, top_k=3, use_advanced=False)
            if not results:
                return "知识库中未找到相关信息"
            formatted = "\n\n".join([f"相关文档 {i+1}: {r['document'][:200]}..." for i, r in enumerate(results)])
            return f"检索到 {len(results)} 个相关文档:\n{formatted}"
        except Exception as e:
            return f"RAG检索失败: {str(e)}"

    def _exec_mcp_call(self, tool_name, parameters):
        """执行工具调用（MCP 或本地工具）

        Returns:
            统一格式字符串：成功 "工具执行成功: {JSON结果}" / 失败 "工具执行失败: {原因}"
        """
        # 频率限制检查
        if self.tool_manager and self.tool_manager.rate_limiter:
            check = self.tool_manager.rate_limiter.check(tool_name)
            if not check.allowed:
                self.logger.warning(f"工具调用被频率限制: {check.reason}")
                return f"工具执行失败: {check.reason}"

        try:
            # 优先检查是否为本地工具
            if self.tool_manager:
                tool_info = self.tool_manager.get_tool(tool_name)
                if tool_info and tool_info.source == "local":
                    self.logger.info(f"调用本地工具: {tool_name}")
                    result = self.tool_manager.call_local_tool(tool_name, parameters)
                    return f"工具执行成功: {json.dumps(result, ensure_ascii=False)}"

            # MCP 工具调用
            server_name = self._find_tool_server(tool_name)
            if not server_name:
                return f"工具执行失败: 找不到工具 {tool_name} 所属的服务"
            self.logger.info(f"调用MCP - 服务: {server_name}, 工具: {tool_name}")
            result = self.mcp_manager.call_tool(server_name=server_name, tool_name=tool_name, parameters=parameters)
            if isinstance(result, dict) and 'error' in result:
                return f"工具执行失败: {result['error']}"

            # 记录调用
            if self.tool_manager:
                self.tool_manager.record_call(tool_name)

            return f"工具执行成功: {json.dumps(result, ensure_ascii=False)}"
        except Exception as e:
            return f"工具执行失败: {str(e)}"

    def _infer_tool_name(self, user_query: str, context: str) -> Optional[str]:
        """从用户查询中推断目标工具名（意图识别不再提供tool_name时使用）"""
        if not self.llm_client or not self.mcp_manager:
            return None

        tools_desc = self._get_tools_desc(brief=True)
        prompt = f"""根据用户请求，从工具列表中选择最匹配的工具。

用户请求: {user_query}
对话上下文: {context[-300:] if context else '(无)'}

可用工具:
{tools_desc}

只返回工具名称（如 create_complex_order），不要其他内容。如果没有匹配的工具，返回"无"。"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.1).strip()
            if resp and resp != "无" and resp != "finish" and resp != "search_knowledge":
                self.logger.info(f"[TaskPlanner] 推断工具名: {resp}")
                return resp
        except Exception as e:
            self.logger.warning(f"[TaskPlanner] 工具名推断失败: {e}")
        return None

    def _handle_greeting(self):
        """处理问候"""
        greetings = [
            "你好！我是你的智能助手。有什么我可以帮你的吗？",
            "嗨！很高兴见到你。我可以回答问题、搜索知识，还能调用各种工具来帮助你。",
            "你好！我已准备好帮助你了。请告诉我你想了解什么。"
        ]
        return random.choice(greetings)

    def _handle_ask_tool_info(self, tool_name, verbose):
        """处理询问工具信息"""
        if verbose:
            print(f"📖 查询工具信息: {tool_name}\n")
        try:
            available_tools = self.mcp_manager.get_available_tools(use_cache=True)
            tool_info = next((t for t in available_tools if t['name'] == tool_name), None)
            if not tool_info:
                return f"抱歉，未找到工具 {tool_name} 的信息。"

            input_schema = tool_info.get('inputSchema', {})
            properties = input_schema.get('properties', {})
            required_params = input_schema.get('required', [])
            description = tool_info.get('description', '无描述')

            if not properties:
                return f"✅ **{tool_name}** 不需要提供任何额外信息，直接告诉我就可以执行。\n\n📝 功能说明：{description}"

            param_lines = []
            for pname, pschema in properties.items():
                req_mark = " ⚠️" if pname in required_params else ""
                param_lines.extend(self._format_param_info(f"{pname}{req_mark}", pschema))

            parts = [f"📋 **{tool_name}** 需要的信息：\n", f"💡 {description}\n", "---\n",
                     "**请提供以下信息**：", "（带 ⚠️ 的是必须提供的）\n"]
            parts.extend(param_lines)
            parts.append("\n---")
            parts.append("💬 提示：直接告诉我这些信息，我会帮您完成操作。")
            return "\n".join(parts)
        except Exception as e:
            return f"抱歉，获取工具信息时遇到错误: {str(e)}"

    # ================================================================
    # 答案生成
    # ================================================================
    def _generate_final_answer(self, user_query, execution_history, context, long_term_context):
        """基于执行历史生成最终回答"""
        hist = ""
        for s in execution_history:
            r = s.get('result', '')
            if len(r) > 500:
                r = r[:500] + "...(已截断)"
            hist += f"步骤{s['step_id']}({s['description']}): {r}\n\n"

        # 上下文工程：动态预算分配 + 分区预算管理（无RAG/工具，预算释放给执行结果分区）
        if self.context_manager:
            active = ['short_term_memory', 'tool_results']
            if long_term_context:
                active.append('long_term_memory')
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section('short_term_memory', context)
            long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)
            hist = self.context_manager.manage_section('tool_results', hist)

        parts = []
        if long_term_context:
            parts.append(long_term_context)
        parts.append(f"""用户请求: {user_query}

对话上下文:
{context}

执行过程和结果:
{hist}

请基于以上执行结果，为用户提供完整、准确、友好的回答。
如果某些操作失败了，请说明情况并给出建议。""")

        try:
            return self.llm_client.generate(prompt="\n".join(parts), temperature=0.5).strip()
        except Exception as e:
            self.logger.error(f"生成最终答案失败: {e}")
            return "抱歉，在整理结果时遇到了问题。"

    def _gen_step_answer(self, user_query, step_desc, execution_history, context, long_term_context):
        """为generate_answer类型的步骤生成答案"""
        hist = "\n".join(f"- {h['description']}: {h.get('result', '')[:300]}" for h in execution_history)
        parts = []
        if long_term_context:
            parts.append(long_term_context)
        parts.append(f"用户请求: {user_query}\n当前步骤: {step_desc}\n\n前置步骤结果:\n{hist if hist else '（无）'}\n\n对话上下文: {context}\n\n请基于前置步骤的结果，完成当前步骤的任务。")
        try:
            return self.llm_client.generate(prompt="\n".join(parts), temperature=0.5).strip()
        except Exception as e:
            return f"生成答案失败: {e}"

    @staticmethod
    def _is_missing_user_input_error(step_result: str) -> bool:
        """检查步骤失败是否因为用户未提供必需参数

        匹配模式：
        - Pydantic validation error with input_value=None
        - 工具明确要求缺失的用户输入参数
        """
        if not step_result:
            return False
        result_lower = step_result.lower()
        # Pydantic校验：必需字段为 None
        if 'validation error' in result_lower and 'input_value=none' in result_lower:
            return True
        # 工具返回"缺少XX参数"
        if 'input should be a valid' in result_lower and 'none' in result_lower:
            return True
        return False

    def _generate_ask_user_response(
        self, user_query, execution_history, failed_result,
        failed_step_desc, context, long_term_context,
    ) -> str:
        """生成向用户请求补充信息的回复

        当步骤因缺少用户必需参数失败时，总结已完成的进度并询问用户缺失的信息。
        """
        # 整理已完成步骤摘要
        succeeded_parts = []
        for h in execution_history:
            result_preview = h['result'][:200] if h.get('result') else ''
            if '失败' not in result_preview and 'error' not in result_preview.lower():
                succeeded_parts.append(f"- {h['description']}: {result_preview}")
        succeeded_summary = "\n".join(succeeded_parts) if succeeded_parts else "（无）"

        parts = []
        if long_term_context:
            parts.append(long_term_context)
        parts.append(f"""用户请求: {user_query}

已完成的操作:
{succeeded_summary}

当前失败的步骤: {failed_step_desc}
失败原因: {failed_result[:300]}

请为用户生成一个友好的回复，内容包括：
1. 简要说明哪些操作已经成功完成（包括关键ID等信息）
2. 说明当前步骤因为缺少必要信息而无法继续
3. 明确告诉用户需要补充哪些信息（从错误信息中提取缺少的参数名称）
4. 告诉用户补充信息后可以继续完成剩余操作

语气友好，简洁明了。""")

        try:
            return self.llm_client.generate(prompt="\n".join(parts), temperature=0.5).strip()
        except Exception as e:
            self.logger.error(f"生成用户补充信息请求失败: {e}")
            return f"部分操作已完成，但在执行「{failed_step_desc}」时发现缺少必要信息。请补充相关参数后重试。\n错误详情: {failed_result[:200]}"

    def _summarize_history(self, user_query, thoughts_actions):
        """总结ReAct历史"""
        history = self._format_react_history(thoughts_actions)
        prompt = f"任务: {user_query}\n\n已执行的步骤:\n{history}\n\n请基于以上执行历史，为用户提供一个总结性的回答。"
        try:
            return self.llm_client.generate(prompt=prompt, temperature=0.5).strip()
        except Exception as e:
            return f"总结失败: {e}"

    # ================================================================
    # 辅助方法
    # ================================================================
    def _get_tools_desc(self, brief=True, target_tool: str = None, query: str = ""):
        """获取可用工具列表描述

        懒加载策略：
        - brief=True（默认）：仅返回名称+描述，用于工具选择阶段，节省上下文token
        - brief=False：包含完整Input Schema，仅在需要填充参数时使用
        - target_tool：指定需要加载完整 schema 的工具名
        - query：用户查询，用于关键词/分类过滤（仅 tool_manager 可用时生效）

        当 tool_manager 可用时，通过多层过滤（关键词→频率→top_k）缩小候选集；
        否则回退到直接从 mcp_manager 获取全量工具。
        """
        # 优先使用 tool_manager（多层过滤 + 本地工具支持）
        if self.tool_manager:
            return self.tool_manager.get_tools_desc(query=query, target_tool=target_tool)

        # 回退：直接从 mcp_manager 获取
        desc = ["1. search_knowledge - 从知识库中搜索相关信息"]
        if not brief:
            desc.append('   参数: {"query": "搜索关键词"}')
        idx = 2
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_available_tools(use_cache=True)
            for tool in mcp_tools:
                name = tool.get('name', '未知工具')
                d = tool.get('description', '无描述')
                td = f"{idx}. {name} - {d}"
                should_load_schema = (not brief) or (target_tool and name == target_tool)
                if should_load_schema:
                    schema = tool.get('inputSchema', {})
                    if schema:
                        sj = json.dumps(schema, indent=2, ensure_ascii=False)
                        td += f"\n   Input Schema:\n" + '\n'.join('   ' + l for l in sj.split('\n'))
                desc.append(td)
                idx += 1
        desc.append(f"\n{idx}. finish - 完成任务并返回最终答案")
        if not brief:
            desc.append('   参数: {"answer": "最终答案内容"}')
        return "\n".join(desc)

    def _find_tool_server(self, tool_name):
        """查找工具所属的服务器"""
        if self.tool_manager:
            return self.tool_manager.find_tool_server(tool_name)
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_available_tools(use_cache=True):
            if tool['name'] == tool_name:
                return tool.get('server')
        servers = self.mcp_manager.get_enabled_servers()
        if servers:
            return servers[0].get('name') if isinstance(servers, list) else list(servers.keys())[0]
        return None

    def _get_tool_schema(self, tool_name):
        """获取工具的输入schema"""
        if self.tool_manager:
            return self.tool_manager.get_tool_schema(tool_name)
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_available_tools(use_cache=True):
            if tool['name'] == tool_name:
                return tool.get('inputSchema', {})
        return None

    @staticmethod
    def _compress_schema(schema: dict) -> dict:
        """压缩JSON Schema，减少上下文窗口占用

        主要优化：
        1. anyOf[{type:X},{type:null}] → {type:X, optional:true}（Pydantic Optional模式）
        2. 移除 additionalProperties、title 等对参数提取无用的字段
        3. 保留 description、default、enum 等语义信息
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            # 跳过对参数提取无用的元数据字段
            if key in ('additionalProperties', 'title', '$defs'):
                continue

            if key == 'properties' and isinstance(value, dict):
                result[key] = {
                    pname: TaskPlanner._compress_property(pschema)
                    for pname, pschema in value.items()
                }
            else:
                result[key] = value

        return result

    @staticmethod
    def _compress_property(prop: dict) -> dict:
        """压缩单个属性的schema"""
        if not isinstance(prop, dict):
            return prop

        # 处理 anyOf: [{type:X}, {type:null}] → {type:X, optional:true}
        if 'anyOf' in prop:
            non_null = [s for s in prop['anyOf'] if s.get('type') != 'null']
            if len(non_null) == 1:
                compressed = dict(non_null[0])
                compressed['optional'] = True
                # 保留外层的 default、description
                if 'default' in prop:
                    compressed['default'] = prop['default']
                if 'description' in prop:
                    compressed['description'] = prop['description']
                return compressed

        # 递归处理嵌套对象
        if prop.get('type') == 'object' and 'properties' in prop:
            result = dict(prop)
            result['properties'] = {
                k: TaskPlanner._compress_property(v)
                for k, v in prop['properties'].items()
            }
            result.pop('additionalProperties', None)
            result.pop('title', None)
            return result

        # 移除无用字段
        result = {k: v for k, v in prop.items() if k not in ('title',)}
        return result

    def _format_react_history(self, thoughts_actions):
        """格式化ReAct历史"""
        if not thoughts_actions:
            return ""
        lines = []
        for i, step in enumerate(thoughts_actions, 1):
            lines.append(f"步骤 {i}:")
            lines.append(f"  思考: {step['thought']}")
            lines.append(f"  行动: {step['action']}")
            lines.append(f"  输入: {step['action_input']}")
            if 'observation' in step:
                obs = step['observation']
                if len(obs) > 800:
                    obs = obs[:800] + "...(已截断)"
                lines.append(f"  观察: {obs}")
        return "\n".join(lines)

    def _parse_react_response(self, response):
        """解析ReAct响应

        支持两种 LLM 输出格式：
        1. 标准文本格式:
           Thought: ...
           Action: ...
           Action Input: {...}
        2. JSON 格式（LLM 有时会输出单行 JSON）:
           {"Thought": "...", "Action": "...", "Action Input": {...}}
        """
        lines = response.strip().split('\n')
        current_section = None
        content_lines = []
        sections = {}
        for line in lines:
            line = line.strip()
            if line.startswith('Thought:'):
                if current_section and content_lines:
                    sections[current_section] = ' '.join(content_lines).strip()
                current_section = 'thought'
                content_lines = [line[len('Thought:'):].strip()]
            elif line.startswith('Action Input:'):
                if current_section and content_lines:
                    sections[current_section] = ' '.join(content_lines).strip()
                current_section = 'action_input'
                content_lines = [line[len('Action Input:'):].strip()]
            elif line.startswith('Action:'):
                if current_section and content_lines:
                    sections[current_section] = ' '.join(content_lines).strip()
                current_section = 'action'
                content_lines = [line[len('Action:'):].strip()]
            elif line and current_section:
                content_lines.append(line)
        if current_section and content_lines:
            sections[current_section] = ' '.join(content_lines).strip()

        # 如果标准格式未解析到结果，尝试 JSON 格式解析
        if not sections.get('action'):
            sections = self._try_parse_json_react(response, sections)

        thought = sections.get('thought', '')
        action = sections.get('action', '').strip()
        action_input_str = sections.get('action_input', '{}')

        # 如果 action_input 已经是 dict（从 JSON 解析得到），直接使用
        if isinstance(action_input_str, dict):
            action_input = action_input_str
        else:
            action_input = self._parse_action_input(action_input_str)

        # LLM 有时直接把工具名作为 Action（如 "get_user_detail"）
        # 而非规定的 "call_mcp_tool"，这里自动修正
        valid_actions = {'search_knowledge', 'call_mcp_tool', 'finish'}
        if action not in valid_actions and action:
            # 检查是否是一个已知的 MCP 工具名
            if self.mcp_manager:
                try:
                    known_tools = {t['name'] for t in self.mcp_manager.get_available_tools(use_cache=True)}
                    if action in known_tools:
                        self.logger.info(f"[ReAct] 自动修正 Action: '{action}' → 'call_mcp_tool'")
                        # 把工具名放入 action_input
                        if 'tool_name' not in action_input:
                            action_input = {'tool_name': action, 'parameters': action_input}
                        action = 'call_mcp_tool'
                except Exception:
                    pass

        return thought, action, action_input

    def _try_parse_json_react(self, response, existing_sections):
        """尝试将 LLM 响应解析为 JSON 格式的 ReAct 输出

        处理 LLM 返回如下格式的情况：
        {"Thought": "...", "Action": "...", "Action Input": {...}}
        """
        text = response.strip()
        # 尝试提取第一个完整的 JSON 对象
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if not m:
                return existing_sections
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                return existing_sections

        if not isinstance(obj, dict):
            return existing_sections

        # 支持大小写和多种 key 名
        key_map = {
            'thought': 'thought', 'Thought': 'thought',
            'action': 'action', 'Action': 'action',
            'action_input': 'action_input', 'Action Input': 'action_input',
            'Action_Input': 'action_input', 'actionInput': 'action_input',
        }

        sections = dict(existing_sections)
        for raw_key, norm_key in key_map.items():
            if raw_key in obj and obj[raw_key]:
                sections[norm_key] = obj[raw_key]

        # 处理 LLM 将 answer 直接放在顶层的情况：
        # {"action": "finish", "answer": "..."} → action_input = {"answer": "..."}
        if sections.get('action') == 'finish' and 'action_input' not in sections:
            answer = obj.get('answer') or obj.get('Answer', '')
            if answer:
                sections['action_input'] = {'answer': answer}

        if sections.get('action'):
            self.logger.info(f"[ReAct] 使用 JSON 格式解析成功: action={sections.get('action')}")
        return sections

    @staticmethod
    def _parse_action_input(action_input_str):
        """解析 action_input 字符串为 dict"""
        cleaned = action_input_str.strip()
        # 清理 markdown 代码块包裹
        if cleaned.startswith('```'):
            parts = cleaned.split('```')
            inner = parts[1] if len(parts) >= 2 else cleaned
            if inner.startswith('json'):
                inner = inner[4:]
            cleaned = inner.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试提取第一个平衡的 {...} JSON 块（非贪婪）
            # 使用平衡括号匹配，避免贪婪匹配跨越多个 JSON 对象
            depth = 0
            start = -1
            for i, ch in enumerate(cleaned):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break
            return {'raw': action_input_str}

    def _format_param_info(self, param_name, param_schema, indent=0):
        """递归格式化参数信息"""
        lines = []
        indent_str = "  " * indent
        actual = param_schema
        if 'anyOf' in param_schema:
            for s in param_schema['anyOf']:
                if s.get('type') != 'null':
                    actual = s
                    break
        desc = actual.get('description', '')
        enum_hint = ""
        if 'enum' in actual:
            enum_hint = f"（可选：{' / '.join(map(str, actual['enum']))}）"
        if desc:
            lines.append(f"{indent_str}• **{param_name}**{enum_hint}：{desc}")
        else:
            lines.append(f"{indent_str}• **{param_name}**{enum_hint}")
        if actual.get('type') == 'object' and 'properties' in actual:
            sub_req = actual.get('required', [])
            for sn, ss in actual['properties'].items():
                rm = " ⚠️" if sn in sub_req else ""
                lines.extend(self._format_param_info(f"{sn}{rm}", ss, indent + 1))
        return lines

    @staticmethod
    def _is_duplicate_action(action, action_input, previous_steps):
        """检测当前动作是否与之前已成功执行的步骤重复"""
        for prev in previous_steps:
            if prev.get('action') != action:
                continue
            # 检查观察结果是否表明成功
            obs = prev.get('observation', '')
            if '成功' not in obs and '完成' not in obs:
                continue
            # 比较action_input是否相同
            prev_input = prev.get('action_input', {})
            if action == 'call_mcp_tool':
                if (prev_input.get('tool_name') == action_input.get('tool_name')
                        and prev_input.get('parameters') == action_input.get('parameters')):
                    return True
            elif action == 'search_knowledge':
                if prev_input.get('query') == action_input.get('query'):
                    return True
        return False

    @staticmethod
    def _sanitize_feedback(message: str, suggestions: list) -> str:
        """清理评估反馈，移除具体示例值避免污染规划上下文"""
        import re
        text = message
        if suggestions:
            text += "。建议: " + "; ".join(suggestions)
        # 移除括号中的示例值，如"（如UID-1234）"、"（例如order-001）"
        text = re.sub(r'[（(]\s*(?:如|例如|比如|e\.?g\.?)\s*[^)）]*[)）]', '', text)
        # 移除引号中可能的示例ID/值
        text = re.sub(r'[""「」『』]\s*(?:UID|uid|ID|id|USER|user)[-_]?\d+\s*[""「」『』]', '"<动态参数>"', text)
        return text

    @staticmethod
    def _has_placeholder(parameters):
        """检查参数是否包含占位符"""
        for v in parameters.values():
            if isinstance(v, str) and (v.startswith('{') or v.startswith('<') or v == 'null' or v == ''):
                return True
        return False

    @staticmethod
    def _clean_json(response):
        """清理LLM返回的JSON，处理常见的格式问题"""
        response = response.strip()

        # 1. 移除 markdown 代码块
        if '```' in response:
            parts = response.split('```')
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
                if cleaned.startswith('[') or cleaned.startswith('{'):
                    response = cleaned
                    break

        # 2. 提取第一个完整的 JSON 数组或对象
        #    LLM 可能在 JSON 前后输出解释文字
        match = re.search(r'(\[.*\])', response, re.DOTALL)
        if match:
            response = match.group(1)
        else:
            match = re.search(r'(\{.*\})', response, re.DOTALL)
            if match:
                response = match.group(1)

        # 3. 移除尾部逗号（JSON 不允许 trailing comma）
        #    例如: {"a": 1,} -> {"a": 1}
        response = re.sub(r',\s*([}\]])', r'\1', response)

        # 4. 移除 JSON 中的行注释 (// ...)
        response = re.sub(r'//[^\n]*', '', response)

        # 5. 移除控制字符（保留换行、tab）
        response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response)

        return response.strip()

    def _parse_json_safe(self, response: str, context_hint: str = "plan") -> list:
        """安全解析 JSON，失败时尝试 LLM 修复"""
        cleaned = self._clean_json(response)

        # 首次尝试
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_err:
            self.logger.warning(f"JSON 首次解析失败({context_hint}): {first_err}")

        # 用 LLM 修复损坏的 JSON
        try:
            repair_prompt = f"""以下JSON格式有误，请修复并只返回修正后的JSON数组，不要有任何其他文字：

{cleaned[:1500]}

修正后的JSON："""
            repaired = self.llm_client.generate(prompt=repair_prompt, temperature=0.1, max_tokens=1500).strip()
            repaired = self._clean_json(repaired)
            return json.loads(repaired)
        except Exception as repair_err:
            self.logger.error(f"JSON 修复也失败({context_hint}): {repair_err}")
            raise

    @staticmethod
    def _print_eval(phase_name, eval_result):
        """打印评估结果"""
        icons = {EvalSeverity.MUST_FIX: "🚫", EvalSeverity.ACCEPTABLE: "✅", EvalSeverity.REMINDER: "💡"}
        icon = icons.get(eval_result.severity, "❓")
        status = "通过" if eval_result.passed else "未通过"
        print(f"   {icon} [{phase_name}评估] {status} - {eval_result.message}")
        for s in eval_result.suggestions:
            print(f"      → {s}")
