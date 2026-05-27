"""
任务评估模块 (Task Evaluator)

在任务执行前、中、后进行评估，评估结果按三个级别反馈给规划器：
- MUST_FIX (必须解决): 阻塞性问题，必须修复后才能继续
- ACCEPTABLE (本次可接受): 存在瑕疵但不影响主流程，可继续执行
- REMINDER (提醒): 轻微建议，仅作为优化参考
"""
import json
import re
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from observability import get_tracer


class EvalSeverity(str, Enum):
    """评估严重级别"""
    MUST_FIX = "must_fix"         # 必须解决 - 阻塞性问题
    ACCEPTABLE = "acceptable"     # 本次可接受 - 有瑕疵但可继续
    REMINDER = "reminder"         # 提醒 - 轻微优化建议


class EvalPhase(str, Enum):
    """评估阶段"""
    PRE_EXECUTION = "pre_execution"      # 执行前评估
    MID_EXECUTION = "mid_execution"      # 执行中评估（每个步骤后）
    POST_EXECUTION = "post_execution"    # 执行后评估


class EvalResult:
    """评估结果"""

    def __init__(
        self,
        phase: EvalPhase,
        severity: EvalSeverity,
        passed: bool,
        message: str,
        suggestions: List[str] = None,
        details: Dict[str, Any] = None,
    ):
        self.phase = phase
        self.severity = severity
        self.passed = passed
        self.message = message
        self.suggestions = suggestions or []
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "suggestions": self.suggestions,
            "details": self.details,
        }

    @property
    def should_replan(self) -> bool:
        """是否需要重新规划"""
        return self.severity == EvalSeverity.MUST_FIX and not self.passed

    @property
    def should_continue(self) -> bool:
        """是否可以继续执行"""
        return self.severity != EvalSeverity.MUST_FIX or self.passed

    def __repr__(self):
        return (
            f"EvalResult(phase={self.phase.value}, severity={self.severity.value}, "
            f"passed={self.passed}, msg={self.message[:50]})"
        )


class TaskEvaluator:
    """
    任务评估器

    职责：
    1. 执行前评估 (pre_evaluate): 检查计划合理性、参数完整性、可行性
    2. 执行中评估 (mid_evaluate): 检查每个步骤的执行结果是否符合预期
    3. 执行后评估 (post_evaluate): 检查最终结果是否满足用户原始需求
    """

    def __init__(self, config: Dict[str, Any], llm_client=None, logger: logging.Logger = None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

        eval_config = config.get('evaluator', {})
        self.enabled = eval_config.get('enabled', True)
        self.pre_eval_enabled = eval_config.get('pre_evaluation', True)
        self.mid_eval_enabled = eval_config.get('mid_evaluation', True)
        self.post_eval_enabled = eval_config.get('post_evaluation', True)

        self.logger.info("TaskEvaluator 初始化完成")

    # ================================================================
    # 执行前评估
    # ================================================================
    def pre_evaluate(
        self,
        user_query: str,
        plan: List[Dict[str, Any]],
        context: str = "",
        completed_history: List[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        执行前评估：检查计划的合理性和完整性

        Args:
            user_query: 用户原始请求
            plan: 规划好的任务步骤列表
            context: 对话上下文
            completed_history: 已完成的步骤历史（重规划时传入）

        Returns:
            EvalResult 评估结果
        """
        tracer = get_tracer()

        with tracer.start_span("task.pre_evaluate", {"eval.phase": "pre_execution"}):
            if not self.enabled or not self.pre_eval_enabled:
                return EvalResult(
                    phase=EvalPhase.PRE_EXECUTION,
                    severity=EvalSeverity.ACCEPTABLE,
                    passed=True,
                    message="评估已跳过（功能未启用）"
                )

            self.logger.info("执行前评估...")

            # 基础检查：计划不能为空
            if not plan:
                result = EvalResult(
                    phase=EvalPhase.PRE_EXECUTION,
                    severity=EvalSeverity.MUST_FIX,
                    passed=False,
                    message="任务计划为空，无法执行",
                    suggestions=["请重新分析用户需求并生成执行计划"]
                )
                tracer.set_span_attributes({"eval.severity": result.severity.value, "eval.passed": False})
                return result

            # 使用LLM评估计划合理性
            if self.llm_client:
                result = self._llm_pre_evaluate(user_query, plan, context, completed_history)
            else:
                result = EvalResult(
                    phase=EvalPhase.PRE_EXECUTION,
                    severity=EvalSeverity.ACCEPTABLE,
                    passed=True,
                    message="基础检查通过（无LLM深度评估）"
                )

            tracer.set_span_attributes({
                "eval.severity": result.severity.value,
                "eval.passed": result.passed,
                "eval.message": result.message,
            })
            return result

    def _llm_pre_evaluate(
        self, user_query: str, plan: List[Dict[str, Any]], context: str,
        completed_history: List[Dict[str, Any]] = None,
    ) -> EvalResult:
        """使用LLM进行执行前评估"""
        plan_text = self._format_plan_for_eval(plan)

        # 构建已完成步骤上下文
        completed_text = ""
        if completed_history:
            parts = []
            for h in completed_history:
                parts.append(
                    f"- 步骤{h['step_id']} [{h['action_type']}] {h['description']} => 结果: {h['result'][:200]}"
                )
            completed_text = f"""\n【已完成的步骤】以下步骤已经成功执行，不需要在新计划中重复：
{chr(10).join(parts)}
评估时请将这些已完成步骤的结果视为已知信息，不要认为计划缺少这些步骤。\n"""

        prompt = f"""你是一个任务评估专家。在任务执行前，请评估以下计划是否合理。

用户请求: {user_query}

对话上下文:
{context}
{completed_text}
执行计划:
{plan_text}

**重要 — 本系统的执行机制：**
- 标注了"依赖步骤 X"的步骤，在执行时系统会自动从步骤 X 的返回结果中提取所需参数（如用户ID、订单号等），无需在计划中显式说明传递方式。
- 因此，"依赖步骤: 1"已经充分表达了参数传递关系，不属于缺陷。

请仅从以下维度评估：
1. **完整性**: 计划是否覆盖了用户的所有需求？是否缺少关键步骤？
2. **可行性**: 工具选择是否正确？
3. **顺序性**: 步骤的执行顺序是否合理？

返回JSON格式:
{{"severity": "must_fix/acceptable/reminder", "passed": true/false, "message": "评估总结", "suggestions": ["建议1", "建议2"]}}

严重级别说明：
- must_fix: 计划缺少关键步骤、工具选择明显错误、或执行顺序导致逻辑不通
- acceptable: 计划可行，有小瑕疵但不影响执行
- reminder: 计划良好，仅有轻微优化建议

只返回JSON。"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.2, max_tokens=300)
            result = self._parse_eval_response(response)

            severity = self._map_severity(result.get('severity', 'acceptable'))
            return EvalResult(
                phase=EvalPhase.PRE_EXECUTION,
                severity=severity,
                passed=result.get('passed', True),
                message=result.get('message', ''),
                suggestions=result.get('suggestions', []),
            )
        except Exception as e:
            self.logger.warning(f"LLM执行前评估失败: {e}")
            return EvalResult(
                phase=EvalPhase.PRE_EXECUTION,
                severity=EvalSeverity.ACCEPTABLE,
                passed=True,
                message=f"LLM评估异常，默认通过: {e}"
            )

    # ================================================================
    # 执行中评估
    # ================================================================
    def mid_evaluate(
        self,
        user_query: str,
        current_step: Dict[str, Any],
        step_result: str,
        completed_steps: List[Dict[str, Any]],
        remaining_steps: List[Dict[str, Any]],
    ) -> EvalResult:
        """
        执行中评估：检查当前步骤执行结果

        Args:
            user_query: 用户原始请求
            current_step: 当前执行的步骤
            step_result: 当前步骤的执行结果
            completed_steps: 已完成的步骤列表
            remaining_steps: 剩余待执行的步骤列表

        Returns:
            EvalResult 评估结果
        """
        tracer = get_tracer()

        with tracer.start_span("task.mid_evaluate", {
            "eval.phase": "mid_execution",
            "eval.step_description": current_step.get('description', ''),
        }):
            if not self.enabled or not self.mid_eval_enabled:
                return EvalResult(
                    phase=EvalPhase.MID_EXECUTION,
                    severity=EvalSeverity.ACCEPTABLE,
                    passed=True,
                    message="评估已跳过"
                )

            self.logger.info(f"执行中评估 - 步骤: {current_step.get('description', '未知')}")

            # 基础检查：执行结果是否包含错误
            if self._contains_error(step_result):
                result = EvalResult(
                    phase=EvalPhase.MID_EXECUTION,
                    severity=EvalSeverity.MUST_FIX,
                    passed=False,
                    message=f"步骤执行出错: {step_result[:200]}",
                    suggestions=["检查参数是否正确", "确认工具服务是否可用", "尝试修改参数后重试"]
                )
                tracer.set_span_attributes({"eval.severity": result.severity.value, "eval.passed": False})
                return result

            # 使用LLM评估步骤结果
            if self.llm_client:
                result = self._llm_mid_evaluate(user_query, current_step, step_result, completed_steps, remaining_steps)
            else:
                result = EvalResult(
                    phase=EvalPhase.MID_EXECUTION,
                    severity=EvalSeverity.ACCEPTABLE,
                    passed=True,
                    message="步骤执行完成，基础检查通过"
                )

            tracer.set_span_attributes({
                "eval.severity": result.severity.value,
                "eval.passed": result.passed,
                "eval.message": result.message,
            })
            return result

    def _llm_mid_evaluate(
        self,
        user_query: str,
        current_step: Dict[str, Any],
        step_result: str,
        completed_steps: List[Dict[str, Any]],
        remaining_steps: List[Dict[str, Any]],
    ) -> EvalResult:
        """使用LLM进行执行中评估"""
        current_text = self._format_step_for_eval(current_step)
        completed_text = "\n".join(self._format_step_for_eval(s) for s in completed_steps) if completed_steps else "（无）"
        remaining_text = "\n".join(self._format_step_for_eval(s) for s in remaining_steps) if remaining_steps else "（无）"

        prompt = f"""你是一个任务执行监控专家。请评估当前步骤的执行结果。

用户请求: {user_query}

当前步骤: {current_text}
执行结果: {step_result[:500]}

已完成步骤:
{completed_text}

剩余步骤:
{remaining_text}

请评估：
1. 当前步骤是否成功执行？
2. 执行结果是否符合预期？
3. 后续步骤是否需要调整？

返回JSON格式:
{{"severity": "must_fix/acceptable/reminder", "passed": true/false, "message": "评估总结", "suggestions": ["建议1"]}}

严重级别：
- must_fix: 执行失败或结果完全不符预期，需要重新规划此步骤或调整后续计划
- acceptable: 执行成功但结果有偏差，可以继续但后续需注意
- reminder: 执行成功且符合预期，可能有小优化点

只返回JSON。"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.2, max_tokens=300)
            result = self._parse_eval_response(response)

            severity = self._map_severity(result.get('severity', 'acceptable'))
            return EvalResult(
                phase=EvalPhase.MID_EXECUTION,
                severity=severity,
                passed=result.get('passed', True),
                message=result.get('message', ''),
                suggestions=result.get('suggestions', []),
            )
        except Exception as e:
            self.logger.warning(f"LLM执行中评估失败: {e}")
            return EvalResult(
                phase=EvalPhase.MID_EXECUTION,
                severity=EvalSeverity.ACCEPTABLE,
                passed=True,
                message=f"LLM评估异常，默认通过: {e}"
            )

    # ================================================================
    # 执行后评估
    # ================================================================
    def post_evaluate(
        self,
        user_query: str,
        final_answer: str,
        execution_history: List[Dict[str, Any]],
        context: str = "",
    ) -> EvalResult:
        """
        执行后评估：检查最终结果是否满足用户需求

        Args:
            user_query: 用户原始请求
            final_answer: 最终给用户的回答
            execution_history: 完整的执行历史
            context: 对话上下文

        Returns:
            EvalResult 评估结果
        """
        tracer = get_tracer()

        with tracer.start_span("task.post_evaluate", {"eval.phase": "post_execution"}):
            if not self.enabled or not self.post_eval_enabled:
                return EvalResult(
                    phase=EvalPhase.POST_EXECUTION,
                    severity=EvalSeverity.ACCEPTABLE,
                    passed=True,
                    message="评估已跳过"
                )

            self.logger.info("执行后评估...")

            # 基础检查：回答不能为空
            if not final_answer or not final_answer.strip():
                result = EvalResult(
                    phase=EvalPhase.POST_EXECUTION,
                    severity=EvalSeverity.MUST_FIX,
                    passed=False,
                    message="最终回答为空",
                    suggestions=["需要基于执行结果生成用户回答"]
                )
                tracer.set_span_attributes({"eval.severity": result.severity.value, "eval.passed": False})
                return result

            # 使用LLM评估最终结果
            if self.llm_client:
                result = self._llm_post_evaluate(user_query, final_answer, execution_history, context)
            else:
                result = EvalResult(
                    phase=EvalPhase.POST_EXECUTION,
                    severity=EvalSeverity.ACCEPTABLE,
                    passed=True,
                    message="基础检查通过"
                )

            tracer.set_span_attributes({
                "eval.severity": result.severity.value,
                "eval.passed": result.passed,
                "eval.message": result.message,
            })
            return result

    def _llm_post_evaluate(
        self,
        user_query: str,
        final_answer: str,
        execution_history: List[Dict[str, Any]],
        context: str,
    ) -> EvalResult:
        """使用LLM进行执行后评估"""
        history_text = ""
        for i, step in enumerate(execution_history, 1):
            history_text += f"步骤{i}: {step.get('description', '')} → {step.get('result', '')[:200]}\n"

        prompt = f"""你是一个任务结果评审专家。请评估最终结果是否满足用户的原始需求。

用户请求: {user_query}

对话上下文: {context[:500]}

执行过程:
{history_text if history_text else "（直接回答，无多步执行）"}

最终回答:
{final_answer[:800]}

请评估：
1. **需求覆盖**: 是否回答了用户的所有问题/完成了所有请求？
2. **准确性**: 回答内容是否准确，工具操作是否成功？
3. **完整性**: 是否有遗漏的关键信息？
4. **可理解性**: 回答是否清晰、易懂？

返回JSON格式:
{{"severity": "must_fix/acceptable/reminder", "passed": true/false, "message": "评估总结", "suggestions": ["建议1"]}}

严重级别：
- must_fix: 未能满足核心需求，需要重新执行或补充
- acceptable: 基本满足需求，有小缺陷但可接受
- reminder: 完全满足需求，仅有表达或格式上的微小建议

只返回JSON。"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.2, max_tokens=300)
            result = self._parse_eval_response(response)

            severity = self._map_severity(result.get('severity', 'acceptable'))
            return EvalResult(
                phase=EvalPhase.POST_EXECUTION,
                severity=severity,
                passed=result.get('passed', True),
                message=result.get('message', ''),
                suggestions=result.get('suggestions', []),
            )
        except Exception as e:
            self.logger.warning(f"LLM执行后评估失败: {e}")
            return EvalResult(
                phase=EvalPhase.POST_EXECUTION,
                severity=EvalSeverity.ACCEPTABLE,
                passed=True,
                message=f"LLM评估异常，默认通过: {e}"
            )

    # ---- 辅助方法 ----

    @staticmethod
    def _format_plan_for_eval(plan: List[Dict[str, Any]]) -> str:
        """将JSON计划转为自然语言摘要，供评估LLM阅读"""
        lines = []
        action_type_labels = {
            "search_knowledge": "知识检索",
            "call_mcp_tool": "调用工具",
            "generate_answer": "生成回答",
        }
        for step in plan:
            sid = step.get("step_id", "?")
            desc = step.get("description", "未知")
            atype = step.get("action_type", "")
            label = action_type_labels.get(atype, atype)
            deps = step.get("depends_on", [])
            tool_name = step.get("action_input", {}).get("tool_name", "")

            line = f"步骤{sid}: [{label}] {desc}"
            if tool_name:
                line += f"（工具: {tool_name}）"
            if deps:
                line += f"（依赖步骤: {', '.join(map(str, deps))}）"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_step_for_eval(step: Dict[str, Any]) -> str:
        """将单个步骤转为自然语言描述"""
        sid = step.get("step_id", "?")
        desc = step.get("description", "未知")
        atype = step.get("action_type", "")
        result = step.get("result", "")
        line = f"步骤{sid}: [{atype}] {desc}"
        if result:
            line += f" → {result[:200]}"
        return line

    @staticmethod
    def _contains_error(result_text: str) -> bool:
        """检查结果文本是否包含错误标志"""
        error_indicators = ['错误', '失败', 'error', 'Error', 'failed', 'exception', '异常', '未找到工具']
        return any(indicator in result_text for indicator in error_indicators)

    @staticmethod
    def _clean_json(response: str) -> str:
        """清理LLM返回的JSON，处理常见格式问题"""
        response = response.strip()

        # 1. 移除 markdown 代码块
        if '```' in response:
            parts = response.split('```')
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
                if cleaned.startswith('{') or cleaned.startswith('['):
                    response = cleaned
                    break

        # 2. 用平衡括号提取第一个完整 JSON 对象（避免贪婪正则被嵌套大括号干扰）
        start = response.find('{')
        if start >= 0:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(response)):
                ch = response[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    if in_str:
                        escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        response = response[start:i + 1]
                        break

        # 3. 移除尾部逗号（JSON 不允许 trailing comma）
        response = re.sub(r',\s*([}\]])', r'\1', response)

        # 4. 移除行注释
        response = re.sub(r'//[^\n]*', '', response)

        # 5. 移除控制字符（保留换行和tab）
        response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response)

        return response.strip()

    def _parse_eval_response(self, response: str) -> dict:
        """安全解析LLM评估响应，多策略容错

        策略链：
        1. _clean_json + json.loads（标准解析）
        2. json.JSONDecoder.raw_decode（处理JSON后有尾部文本的情况）
        3. 正则提取关键字段（兜底：JSON严重损坏但关键信息仍可提取）
        """
        cleaned = self._clean_json(response)

        # 策略1: 标准解析
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 策略2: raw_decode 容忍尾部多余文本
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(cleaned)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # 策略3: 正则提取关键字段（兜底）
        self.logger.info("[TaskEvaluator] JSON解析失败，使用正则提取关键字段")
        return self._extract_eval_fields(response)

    @staticmethod
    def _extract_eval_fields(text: str) -> dict:
        """从损坏的JSON文本中用正则提取评估关键字段"""
        # severity
        severity = 'acceptable'
        for s in ['must_fix', 'acceptable', 'reminder']:
            if s in text:
                severity = s
                break

        # passed
        passed = True
        passed_match = re.search(r'"passed"\s*:\s*(true|false)', text, re.IGNORECASE)
        if passed_match:
            passed = passed_match.group(1).lower() == 'true'

        # message（取第一个 "message": "..." 中的内容）
        message = '评估完成'
        msg_match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if msg_match:
            message = msg_match.group(1)

        # suggestions
        suggestions = []
        sugg_match = re.search(r'"suggestions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if sugg_match:
            suggestions = re.findall(r'"((?:[^"\\]|\\.)*)"', sugg_match.group(1))

        return {
            'severity': severity,
            'passed': passed,
            'message': message,
            'suggestions': suggestions,
        }

    @staticmethod
    def _map_severity(severity_str: str) -> EvalSeverity:
        """映射严重级别"""
        mapping = {
            'must_fix': EvalSeverity.MUST_FIX,
            'acceptable': EvalSeverity.ACCEPTABLE,
            'reminder': EvalSeverity.REMINDER,
        }
        return mapping.get(severity_str, EvalSeverity.ACCEPTABLE)
