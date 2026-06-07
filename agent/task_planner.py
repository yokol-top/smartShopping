"""
任务规划模块 (Task Planner)

根据意图识别的复杂度，采用不同的规划策略：
- 简单任务 (SIMPLE): 单意图且步骤在4步以内，使用ReAct循环执行
- 中等任务 (MEDIUM): 单意图4-7步 / ≤3个意图且4-7步骤，使用Plan and Execute规划和执行
- 复杂任务 (COMPLEX): 使用分层规划（先规划几个大的阶段，然后每个阶段按照中等任务处理）

规划器接收评估器的反馈，根据反馈进行重新规划或终止。
"""
import concurrent.futures
import json
import logging
import random
import re
from typing import Dict, Any, List, Optional

from agent.intent_recognizer import IntentResult, IntentType, TaskComplexity
from observability import get_tracer
from .exceptions import NeedUserInputException
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

        self._on_step_complete = None  # 步骤完成回调（供 orchestrator 记忆提取用）
        self.factory = None  # 由 agent.py 在 Orchestrator 创建后注入，避免循环依赖

        # 工具调用器（MCP/本地工具执行逻辑独立维护）
        from .tool_caller import ToolCaller
        self._tool_caller = ToolCaller(
            llm_client=llm_client,
            mcp_manager=mcp_manager,
            tool_manager=tool_manager,
            context_manager=context_manager,
            config=config,
            logger=logger,
        )
        self.logger.info(
            f"TaskPlanner 初始化完成 | 评估开关: pre={self.eval_pre_enabled} "
            f"mid={self.eval_mid_enabled} post={self.eval_post_enabled}"
        )

    # ================================================================
    # 主入口
    # ================================================================
    def execute(self, user_query: str, intent: IntentResult, context: str = "",
                long_term_context: str = "", verbose: bool = True,
                on_step_complete=None) -> str:
        """根据意图识别结果执行任务

        Args:
            on_step_complete: 可选的步骤完成回调 (step_result: str) -> None
                每步完成后调用，用于从结果中提取实体信息存入 orchestrator 记忆。
        """
        self._on_step_complete = on_step_complete
        tracer = get_tracer()

        with tracer.start_span("task.execute", {
            "task.intent_type": intent.intent_type.value,
            "task.complexity": intent.complexity.value,
            "task.user_query": user_query,
        }):
            intent_type = intent.intent_type
            complexity = intent.complexity

            try:
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
            except NeedUserInputException:
                raise  # 向上传播，由 Orchestrator 统一捕获

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

            # Step 3: 按 DAG 层执行步骤（同层步骤无依赖，可并行）
            completed_steps = []
            should_replan = False
            executed_ids: set = set()  # 已执行的 step_id，用于 remaining 计算

            step_layers = self._compute_step_dag_layers(plan)

            for layer in step_layers:
                if should_replan:
                    break

                if len(layer) == 1:
                    # ── 单步串行（原有逻辑）──────────────────────────────
                    step = layer[0]
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
                        step_result = self._execute_step(
                            step, user_query, context, long_term_context, execution_history, verbose
                        )
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
                    executed_ids.add(step.step_id)

                    if self._on_step_complete and step_result:
                        try:
                            self._on_step_complete(step_result)
                        except Exception as e:
                            self.logger.debug(f"步骤完成回调异常: {e}")

                    if verbose:
                        self.logger.debug(f"   ✅ 结果: {step_result}\n")

                    # 执行中评估
                    if self.evaluator and self.eval_mid_enabled:
                        remaining = [s.to_dict() for s in plan if s.step_id not in executed_ids]
                        mid_eval = self.evaluator.mid_evaluate(
                            user_query, record, step_result, completed_steps, remaining
                        )
                        if verbose:
                            self._print_eval("执行中", mid_eval)
                        if mid_eval.should_replan:
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
                            context += (
                                f"\n[执行中反馈] 步骤{step.step_id}: "
                                f"{self._sanitize_feedback(mid_eval.message, mid_eval.suggestions)}"
                            )
                            should_replan = True
                            break

                else:
                    # ── 多步并行（同波次内无依赖关系）────────────────────
                    ids = [s.step_id for s in layer]
                    if verbose:
                        print(f"\n{'='*50}")
                        print(f"⚡ 并行执行步骤 {ids}")
                        print(f"{'='*50}\n")

                    # 波次开始时的历史快照（只读，不含本波次结果）
                    history_snapshot = list(execution_history)

                    def _run_step(s, _snap=history_snapshot):
                        return self._execute_step(
                            s, user_query, context, long_term_context, _snap, False
                        )

                    with tracer.start_span(f"task.execute_wave.parallel",
                                           {"task.wave_step_ids": str(ids)}):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(layer)) as pool:
                            futures = [pool.submit(_run_step, s) for s in layer]
                            wave_results = []
                            for step, fut in zip(layer, futures):
                                try:
                                    result = fut.result(timeout=90)
                                except Exception as e:
                                    self.logger.error(f"步骤 {step.step_id} 并行执行异常: {e}")
                                    result = f"步骤执行失败: {e}"
                                step.result = result
                                step.status = "completed"
                                wave_results.append((step, result))

                    # 批量追加结果（保持 step_id 升序）
                    wave_results.sort(key=lambda x: x[0].step_id)
                    for step, result in wave_results:
                        record = {"step_id": step.step_id, "description": step.description,
                                  "action_type": step.action_type, "result": result}
                        execution_history.append(record)
                        completed_steps.append(record)
                        executed_ids.add(step.step_id)
                        if verbose:
                            self.logger.debug(f"   ✅ 步骤{step.step_id}: {result[:200]}\n")
                        if self._on_step_complete and result:
                            try:
                                self._on_step_complete(result)
                            except Exception as e:
                                self.logger.debug(f"步骤完成回调异常: {e}")

                    # 波次结束后统一执行中评估（对最后一个结果评估，代表整体波次）
                    if self.evaluator and self.eval_mid_enabled and wave_results:
                        last_step, last_result = wave_results[-1]
                        remaining = [s.to_dict() for s in plan if s.step_id not in executed_ids]
                        mid_eval = self.evaluator.mid_evaluate(
                            user_query, completed_steps[-1], last_result, completed_steps, remaining
                        )
                        if verbose:
                            self._print_eval("执行中（波次结束）", mid_eval)
                        if mid_eval.should_replan:
                            replan_count += 1
                            if replan_count > self.max_replan_attempts:
                                break
                            context += (
                                f"\n[执行中反馈] 步骤{last_step.step_id}: "
                                f"{self._sanitize_feedback(mid_eval.message, mid_eval.suggestions)}"
                            )
                            should_replan = True

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
        """复杂任务的分层规划执行（DAG 并行版）

        流程：
        1. 生成高层阶段计划（含 depends_on 依赖声明）
        2. 拓扑排序得到执行波次（DAG layer）
        3. 每个波次内的独立阶段通过 SubAgentFactory 并行执行
           无 factory 或单阶段波次则直接串行执行
        4. 汇总所有阶段结果生成最终答案
        """
        from .task_state import SubTask, DelegationReason

        tracer = get_tracer()
        self.logger.info("复杂任务 - 分层规划模式（DAG 并行）")
        if verbose:
            print(f"\n🧠 复杂任务 - 分层规划模式\n")

        # Step 1: 生成高层阶段计划（含依赖关系）
        with tracer.start_span("task.hierarchical.generate_phases"):
            phases = self._generate_high_level_phases(user_query, intent, context, long_term_context)
            tracer.set_span_attributes({"task.phases_count": len(phases)})

        # Step 2: 计算 DAG 执行层
        dag_layers = self._compute_dag_layers(phases)
        parallel_count = sum(1 for layer in dag_layers if len(layer) > 1)

        if verbose:
            print(f"📋 分层计划（{len(phases)} 个阶段，{len(dag_layers)} 波次，{parallel_count} 波次可并行）:")
            for wave_idx, layer in enumerate(dag_layers, 1):
                for phase in layer:
                    deps = phase.get("depends_on", [])
                    tag = " [并行]" if len(layer) > 1 else ""
                    dep_str = f" ← 依赖{deps}" if deps else ""
                    print(f"   波次{wave_idx}{tag} 阶段{phase['phase_id']}: {phase['description']}{dep_str}")
            print()

        all_phase_results: List[Dict] = []
        accumulated_context = context
        all_tool_names = self._get_all_tool_names()

        # Step 3: 按 DAG 层执行
        for wave_idx, layer in enumerate(dag_layers):
            # 构建前置阶段摘要（供本波次所有阶段共用）
            completed_summary = ""
            if all_phase_results:
                parts = ["\n[已完成的阶段（不要重复执行）]"]
                for pr in all_phase_results:
                    parts.append(f"- 阶段{pr['phase_id']}({pr['description']}): {pr['result'][:300]}")
                completed_summary = "\n".join(parts)

            can_parallel = len(layer) > 1 and self.factory is not None

            if can_parallel:
                # ── 并行波次：通过 SubAgentFactory 并发执行 ──────────────
                if verbose:
                    ids = [p["phase_id"] for p in layer]
                    print(f"\n{'='*60}")
                    print(f"⚡ 波次 {wave_idx+1}（并行）: 阶段 {ids} 同时启动")
                    print(f"{'='*60}\n")

                sub_tasks = []
                for phase in layer:
                    phase_query = (
                        f"[阶段 {phase['phase_id']}] {phase['description']}\n\n"
                        f"（原始用户请求：{user_query}）"
                        f"{completed_summary}"
                    )
                    st = SubTask(
                        id=f"phase_{phase['phase_id']}",
                        description=phase_query,
                        assigned_to="sub_agent",
                        delegation_reason=DelegationReason.PARALLELIZABLE.value,
                        depends_on=[],  # 层内已无依赖
                        agent_role="智能购物规划助手，负责完成分配的阶段任务",
                        agent_tools=all_tool_names,
                        agent_context=accumulated_context,
                        timeout=90.0,
                    )
                    sub_tasks.append((phase, st))

                with tracer.start_span(f"task.hierarchical.wave.{wave_idx+1}.parallel",
                                       {"task.wave_size": len(sub_tasks)}):
                    results_map = self.factory.execute_subtasks_parallel(
                        [st for _, st in sub_tasks]
                    )

                for phase, st in sub_tasks:
                    r = results_map.get(st.id)
                    result_text = (r.summary if r and r.success
                                   else (r.error if r else "子Agent执行失败"))
                    all_phase_results.append({
                        "phase_id": phase["phase_id"],
                        "description": phase["description"],
                        "result": result_text,
                    })
                    accumulated_context += (
                        f"\n[阶段{phase['phase_id']}结果] {phase['description']}: {result_text[:500]}"
                    )
                    tracer.set_span_attributes({
                        f"task.phase_{phase['phase_id']}_result_preview": result_text[:200],
                    })

            else:
                # ── 串行波次：主 Agent 直接执行（单阶段 or 无 factory 兜底）──
                for phase in layer:
                    phase_desc = phase["description"]
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"🔷 阶段 {phase['phase_id']}/{len(phases)}: {phase_desc}")
                        print(f"{'='*60}\n")

                    with tracer.start_span(f"task.hierarchical.phase.{phase['phase_id']}", {
                        "task.phase_id": phase["phase_id"],
                        "task.phase_description": phase_desc,
                    }):
                        phase_query = (
                            f"[阶段 {phase['phase_id']}] {phase_desc}\n\n"
                            f"（原始用户请求：{user_query}）"
                            f"{completed_summary}"
                        )
                        phase_result = self._execute_plan_and_execute(
                            phase_query, intent, accumulated_context,
                            long_term_context, verbose, "medium"
                        )
                        all_phase_results.append({
                            "phase_id": phase["phase_id"],
                            "description": phase_desc,
                            "result": phase_result,
                        })
                        accumulated_context += (
                            f"\n[阶段{phase['phase_id']}结果] {phase_desc}: {phase_result[:500]}"
                        )
                        tracer.set_span_attributes({
                            "task.phase_result_preview": phase_result[:300],
                        })

        # Step 4: 汇总所有阶段结果
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 汇总所有阶段结果")
            print(f"{'='*60}\n")

        with tracer.start_span("task.hierarchical.summarize"):
            final_answer = self._summarize_phases(user_query, all_phase_results, context, long_term_context)

        return final_answer

    def _generate_high_level_phases(self, user_query, intent, context, long_term_context):
        """生成高层阶段计划（含依赖关系，用于 DAG 并行执行）"""
        prompt_parts = []
        if long_term_context:
            prompt_parts.append(long_term_context)
        prompt_parts.append(f"""你是一个任务规划专家。用户的请求比较复杂，请将其拆分为2-4个高层阶段。

用户请求: {user_query}

对话上下文:
{context[:500]}

要求：
1. 每个阶段是一个独立的子目标，应该是一个可以用4-7步完成的中等任务
2. 用 depends_on 字段标注依赖关系（填写前置阶段的 phase_id 列表）
3. 没有依赖的阶段可以并行执行，请尽量识别可以并行的阶段

并行判断原则：
- 若阶段A的输入不依赖阶段B的输出 → depends_on 中不填B，可并行
- 若阶段A必须用到阶段B的结果 → depends_on: [B.phase_id]，必须串行

示例（3个阶段，前两个可并行，第三个依赖前两个）:
[
  {{"phase_id": 1, "description": "搜索手机类商品", "depends_on": []}},
  {{"phase_id": 2, "description": "搜索耳机类商品", "depends_on": []}},
  {{"phase_id": 3, "description": "综合前两步结果，生成推荐方案", "depends_on": [1, 2]}}
]

返回JSON数组格式:
[{{"phase_id": 1, "description": "阶段描述", "depends_on": []}}]

只返回JSON数组。""")

        try:
            response = self.llm_client.generate(
                prompt="\n".join(prompt_parts), temperature=self.react_temperature
            )
            phases = self._parse_json_safe(response, context_hint="high_level_phases")
            if not phases:
                return [{"phase_id": 1, "description": user_query, "depends_on": []}]
            # 补全缺失的 depends_on 字段
            for p in phases:
                p.setdefault("depends_on", [])
            return phases
        except Exception as e:
            self.logger.error(f"生成高层阶段失败: {e}")
            return [{"phase_id": 1, "description": user_query, "depends_on": []}]

    def _compute_dag_layers(self, phases: List[Dict]) -> List[List[Dict]]:
        """对阶段 DAG 做拓扑排序，返回可并行执行的分层列表。

        Returns:
            List[List[phase_dict]] — 每个内层 list 中的阶段可并行，层间严格串行。
        """
        id_map = {p["phase_id"]: p for p in phases}
        in_degree: Dict[int, int] = {p["phase_id"]: 0 for p in phases}
        children: Dict[int, List[int]] = {p["phase_id"]: [] for p in phases}

        for p in phases:
            for dep in p.get("depends_on", []):
                if dep in in_degree:
                    in_degree[p["phase_id"]] += 1
                    children[dep].append(p["phase_id"])

        layers: List[List[Dict]] = []
        ready = sorted([pid for pid, deg in in_degree.items() if deg == 0])

        while ready:
            layers.append([id_map[pid] for pid in ready])
            next_ready: List[int] = []
            for pid in ready:
                for child in children[pid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            ready = sorted(next_ready)

        # 有环保护：剩余未排入的阶段直接追加（降级串行）
        processed = {p["phase_id"] for layer in layers for p in layer}
        remaining = [p for p in phases if p["phase_id"] not in processed]
        if remaining:
            self.logger.warning(f"[TaskPlanner] DAG 检测到环，降级串行: {[p['phase_id'] for p in remaining]}")
            for p in remaining:
                layers.append([p])

        return layers

    def _compute_step_dag_layers(self, plan: List['PlanStep']) -> List[List['PlanStep']]:
        """对计划步骤做拓扑排序，返回可并行执行的分层列表。

        与 _compute_dag_layers 逻辑相同，但操作对象是 PlanStep（依赖 step_id: int）。

        Returns:
            List[List[PlanStep]] — 同层内步骤可并行，层间严格串行。
        """
        id_map = {s.step_id: s for s in plan}
        in_degree: Dict[int, int] = {s.step_id: 0 for s in plan}
        children: Dict[int, List[int]] = {s.step_id: [] for s in plan}

        for s in plan:
            for dep in s.depends_on:
                if dep in in_degree:
                    in_degree[s.step_id] += 1
                    children[dep].append(s.step_id)

        layers: List[List['PlanStep']] = []
        ready = sorted([sid for sid, deg in in_degree.items() if deg == 0])

        while ready:
            layers.append([id_map[sid] for sid in ready])
            next_ready: List[int] = []
            for sid in ready:
                for child in children[sid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            ready = sorted(next_ready)

        # 有环保护：剩余步骤直接追加（降级串行）
        processed = {s.step_id for layer in layers for s in layer}
        remaining = [s for s in plan if s.step_id not in processed]
        if remaining:
            self.logger.warning(f"[TaskPlanner] 步骤 DAG 检测到环，降级串行: {[s.step_id for s in remaining]}")
            for s in remaining:
                layers.append([s])

        return layers

    def _get_all_tool_names(self) -> List[str]:
        """返回 mcp_manager 中所有可用工具的名称列表（供 SubTask 白名单使用）"""
        if not self.mcp_manager:
            return []
        try:
            tools = self.mcp_manager.get_available_tools(use_cache=True)
            return [t.get("name", "") for t in tools if t.get("name")]
        except Exception:
            return []

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

请基于以上所有阶段的执行结果，为用户提供完整、准确、有条理的最终回答。
【重要】推荐商品时必须在商品名前标注商品ID，格式：[P001] 商品名 ¥价格。商品ID、订单号（ORD-XXX）、地址ID（ADDR-XXX）、银行卡ID（CARD-XXX）等关键标识符必须原样保留，不可省略。""")

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
        # 上下文工程：懒加载工具描述（仅名称+描述，不含schema；按意图类型过滤）
        tools_desc = self._get_tools_desc(
            brief=True,
            intent_type=intent.intent_type.value,
            query=user_query,
        )
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
- search_knowledge: 仅用于检索公司商品/内容知识库（产品介绍、FAQ等通用知识）
  ⚠️ 严禁用于查询用户私有数据（用户账户、收货地址、银行卡、订单记录等），这类数据只能用 call_mcp_tool
  action_input: {{"query": "搜索关键词"}}
- call_mcp_tool: 调用MCP工具（必须使用上方"可用工具"列表中的工具名，不能用 search_knowledge 作为工具名）
  action_input: {{"tool_name": "工具名", "parameters": {{...}}}}
- generate_answer: 基于已有信息生成回答
  action_input: {{"prompt_hint": "回答要点"}}

规则:
1. 最后一步应该是generate_answer，用于整合结果
2. 每步的action_input必须包含足够参数
3. 依赖前置步骤结果的在depends_on中标注
4. 前置步骤返回的ID、数据等会自动传递给后续步骤，不需要单独规划"获取ID"的步骤
5. 只使用可用工具列表中存在的工具
6. 若可用工具中没有满足需求的工具，直接用 generate_answer 向用户说明情况，不要规划无法执行的步骤

只返回JSON数组，不要有任何解释性文字。""")

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

            # 重新编号步骤ID（过滤后可能不连续），同步更新 depends_on 引用
            old_to_new = {s.step_id: i for i, s in enumerate(steps, 1)}
            for i, step in enumerate(steps, 1):
                step.step_id = i
                step.depends_on = [old_to_new[dep] for dep in step.depends_on if dep in old_to_new]

            return steps
        except Exception as e:
            self.logger.error(f"生成计划失败: {e}")
            return [PlanStep(step_id=1, description="直接回答用户问题", action_type="generate_answer",
                             action_input={"prompt_hint": "直接回答"})]

    # ================================================================
    # ReAct循环（简单任务使用）
    # ================================================================
    def _react_loop(self, user_query, intent, context, long_term_context, verbose):
        """简单任务的ReAct循环执行（委托给 UnifiedReActExecutor）"""
        from .unified_react_executor import UnifiedReActExecutor, ReActConfig
        executor = UnifiedReActExecutor(
            llm_client=self.llm_client,
            mcp_manager=self.mcp_manager,
            rag_engine=self.rag_engine,
            context_manager=self.context_manager,
            logger=self.logger,
        )
        cfg = ReActConfig(
            max_iterations=self.react_max_iterations,
            temperature=self.react_temperature,
            allowed_tools=None,   # TaskPlanner 无工具白名单限制
            enable_rag=True,
        )
        return executor.execute(
            task_desc=user_query,
            context=context,
            long_term_context=long_term_context,
            verbose=verbose,
            config=cfg,
        )

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

            parameters = action_input.get('parameters', {})
            # 必须动态提取参数的情况：
            # 1. 参数为空或含占位符
            # 2. 有前置步骤结果（参数可能依赖动态值如user_id）
            # 3. 懒加载模式：计划生成时LLM未见schema，参数不可靠
            has_prior_results = bool(execution_history)
            lazy_loading = bool(self.context_manager)
            if not parameters or self._has_placeholder(parameters) or has_prior_results or lazy_loading:
                return self._tool_caller.extract_params_and_call(tool_name, step.description, user_query, context, long_term_context, execution_history, verbose)
            return self._tool_caller.call(tool_name, parameters)
        elif action_type == "generate_answer":
            return self._gen_step_answer(user_query, step.description, execution_history, context, long_term_context)
        else:
            return f"未知的action_type: {action_type}"

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
        if not self.rag_engine:
            self.logger.warning("RAG引擎不可用，降级为直接对话")
            return self._direct_chat(user_query, context, long_term_context)

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
            if advanced and self.config.get('rag', {}).get('self_fix', {}).get('enabled', True):
                if verbose:
                    print("🔍 正在验证答案质量...\n")
                contexts = [r['document'] for r in results]
                fix_result = self.rag_engine.self_fix.verify_and_fix(
                    query=user_query, answer=answer, context=contexts,
                    conversation_context=context,
                )
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
        if not self.rag_engine:
            return "知识库不可用"
        try:
            results = self.rag_engine.retrieve(query=query, context=context, top_k=3, use_advanced=False)
            if not results:
                return "知识库中未找到相关信息"
            formatted = "\n\n".join([f"相关文档 {i+1}: {r['document'][:200]}..." for i, r in enumerate(results)])
            return f"检索到 {len(results)} 个相关文档:\n{formatted}"
        except Exception as e:
            return f"RAG检索失败: {str(e)}"

    def _infer_tool_name(self, user_query: str, context: str) -> Optional[str]:
        """从用户查询中推断目标工具名（意图识别不再提供tool_name时使用）"""
        if not self.llm_client or not self.mcp_manager:
            return None

        # 工具名推断需要见到所有工具，不按意图过滤，但传入 query 供 tool_manager 做关键词过滤
        tools_desc = self._get_tools_desc(brief=True, query=user_query)
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
如果某些操作失败了，请说明情况并给出建议。
【重要】推荐商品时必须在商品名前标注商品ID，格式：[P001] 商品名 ¥价格。商品ID、订单号（ORD-XXX）、地址ID（ADDR-XXX）、银行卡ID（CARD-XXX）等关键标识符必须原样保留在回答中，不可省略。""")

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
        parts.append(f"用户请求: {user_query}\n当前步骤: {step_desc}\n\n前置步骤结果:\n{hist if hist else '（无）'}\n\n对话上下文: {context}\n\n请基于前置步骤的结果，完成当前步骤的任务。\n【重要】推荐商品时必须在商品名前标注商品ID，格式：[P001] 商品名 ¥价格。商品ID、订单号（ORD-XXX）、地址ID（ADDR-XXX）、银行卡ID（CARD-XXX）等关键标识符必须原样保留，不可省略。")
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

    # ================================================================
    # 辅助方法
    # ================================================================
    def _get_tools_desc(self, brief=True, target_tool: str = None, query: str = "",
                        intent_type: str = None):
        """获取可用工具列表描述

        懒加载策略：
        - brief=True（默认）：仅返回名称+描述，用于工具选择阶段，节省上下文token
        - brief=False：包含完整Input Schema，仅在需要填充参数时使用
        - target_tool：指定需要加载完整 schema 的工具名
        - query：用户查询，用于关键词/分类过滤
        - intent_type：意图类型，用于按阶段过滤（避免导购阶段出现下单工具）

        当 tool_manager 可用时，通过多层过滤（关键词→频率→top_k）缩小候选集；
        否则回退到直接从 mcp_manager 获取，并按意图类型过滤。
        """
        # 优先使用 tool_manager（多层过滤 + 本地工具支持）
        if self.tool_manager:
            return self.tool_manager.get_tools_desc(query=query, target_tool=target_tool)

        # 回退：直接从 mcp_manager 获取，按意图类型过滤
        desc = ["1. search_knowledge - 从知识库中搜索相关信息"]
        if not brief:
            desc.append('   参数: {"query": "搜索关键词"}')
        idx = 2
        if self.mcp_manager:
            if intent_type:
                mcp_tools = self.mcp_manager.get_tools_for_context(
                    intent_type=intent_type, query=query
                )
            else:
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

        # cleaned 为空说明 LLM 返回了纯文本而非 JSON（如解释性回复），直接降级，不再修复
        if not cleaned:
            self.logger.warning(f"JSON 清理后为空({context_hint})，LLM 可能返回了非 JSON 文本，跳过修复")
            raise json.JSONDecodeError("cleaned response is empty", "", 0)

        # 用 LLM 修复损坏的 JSON（传入原始响应而非清理后的内容，避免信息丢失）
        try:
            repair_prompt = f"""以下内容应为JSON数组格式的执行计划，但格式有误。请修复并只返回修正后的JSON数组，不要有任何其他文字：

{response[:1500]}

只返回JSON数组："""
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
