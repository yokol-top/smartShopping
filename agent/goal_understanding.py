"""
目标理解层 (Goal Understanding Layer)

职责：
1. 理解用户意图、目标、限制条件、成功标准
2. 结合历史对话 + 长期记忆进行推理
3. 当置信度低时，要求用户澄清
4. 提取任务成功标准与约束条件

基于原有的 IntentRecognizer 扩展，保持向后兼容。
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from observability import get_tracer
from .intent_recognizer import IntentRecognizer, IntentType, TaskComplexity, IntentResult


@dataclass
class GoalUnderstandingResult:
    """目标理解结果"""
    # 基础意图（来自 IntentRecognizer）
    intent_result: IntentResult

    # 目标理解增强字段
    user_goal: str = ""                              # 用户目标描述
    constraints: List[str] = field(default_factory=list)  # 约束条件
    success_criteria: List[str] = field(default_factory=list)  # 成功标准
    needs_clarification: bool = False                # 是否需要用户澄清
    clarification_question: str = ""                 # 澄清问题
    confidence: float = 0.0                          # 综合置信度 [0, 1]
    reasoning: str = ""                              # 推理过程
    memory_context_used: bool = False                # 是否使用了长期记忆

    @property
    def intent_type(self) -> IntentType:
        return self.intent_result.intent_type

    @property
    def complexity(self) -> TaskComplexity:
        return self.intent_result.complexity

    @property
    def is_clear(self) -> bool:
        """意图是否足够清晰可以直接执行"""
        return not self.needs_clarification and self.confidence >= 0.6


class GoalUnderstanding:
    """
    目标理解引擎

    对原有 IntentRecognizer 的增强封装：
    - 低置信度时触发澄清机制
    - 利用长期记忆辅助推理
    - 提取约束条件和成功标准
    """

    # 触发澄清的置信度阈值
    CLARIFICATION_THRESHOLD = 0.6

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client,
        intent_recognizer: IntentRecognizer,
        long_term_memory=None,
        logger: logging.Logger = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.intent_recognizer = intent_recognizer
        self.long_term_memory = long_term_memory
        self.logger = logger or logging.getLogger(__name__)

        goal_config = config.get('goal_understanding', {})
        self.clarification_threshold = goal_config.get(
            'clarification_threshold', self.CLARIFICATION_THRESHOLD
        )
        self.use_memory_reasoning = goal_config.get('use_memory_reasoning', True)

        self.logger.info(
            f"GoalUnderstanding 初始化完成 | 澄清阈值: {self.clarification_threshold}"
        )

    def understand(
        self,
        user_input: str,
        conversation_context: str = "",
        orchestrator_context: str = "",
    ) -> GoalUnderstandingResult:
        """
        理解用户目标（主入口）

        流程:
        1. 调用 IntentRecognizer 识别基础意图
        2. 检查置信度 → 低置信度时尝试用长期记忆辅助
        3. 仍然不清晰 → 生成澄清问题
        4. 提取约束条件和成功标准

        Args:
            user_input: 用户输入
            conversation_context: 对话上下文
            orchestrator_context: Orchestrator结构化记忆（已知实体、方案、决策等）

        Returns:
            GoalUnderstandingResult
        """
        tracer = get_tracer()
        with tracer.start_span("goal.understand", {
            "input.length": len(user_input),
        }) if tracer else _noop():

            # 记录输入上下文状态（便于排查上下文传递问题）
            self.logger.info(
                f"[目标理解] === 开始 === | user_input: {user_input[:80]} | "
                f"conv_ctx_len: {len(conversation_context)} | "
                f"orch_ctx_len: {len(orchestrator_context) if orchestrator_context else 0}"
            )
            self.logger.debug(
                f"[目标理解] 对话上下文(last500): "
                f"{conversation_context[-500:] if conversation_context else '(空)'}"
            )
            if orchestrator_context:
                self.logger.debug(
                    f"[目标理解] orchestrator上下文(last300): "
                    f"{orchestrator_context[-300:]}"
                )

            # Step 2: 预注入长期记忆（避免低置信度时的二次LLM调用）
            # 策略：在第一次 recognize 之前就查询记忆并注入上下文，
            # 这样无论置信度高低都只需要一次 LLM 调用。
            memory_context = ""
            memory_used = False
            if self.use_memory_reasoning:
                memory_context = self._retrieve_memory_context(user_input)
                if memory_context:
                    memory_used = True
                    self.logger.info("[目标理解] 预注入长期记忆辅助推理（单次LLM调用）")

            # Step 1: 基础意图识别（预注入记忆上下文，单次LLM调用完成）
            enriched_context = conversation_context
            if memory_context:
                enriched_context = f"{conversation_context}\n\n[长期记忆参考]\n{memory_context}"
            intent_result = self.intent_recognizer.recognize(user_input, enriched_context)
            confidence = intent_result.confidence
            self.logger.info(
                f"[目标理解] 基础意图: {intent_result.intent_type.value} | "
                f"复杂度: {intent_result.complexity.value} | 置信度: {confidence:.2f} | "
                f"tool: {intent_result.tool_name or '(无)'}"
            )

            # 简单问候不需要深度理解
            if intent_result.intent_type == IntentType.GREETING:
                return GoalUnderstandingResult(
                    intent_result=intent_result,
                    user_goal="问候",
                    confidence=1.0,
                )

            # Step 3: 判断是否需要澄清
            needs_clarification = False
            clarification_question = ""

            # 3a: MCP执行类意图 → 如果已知具体工具，检查参数完整性
            #     tool_name 为 None 时跳过（工具选择由 TaskPlanner 负责）
            param_complete = True
            if (intent_result.intent_type in (IntentType.MCP_EXECUTE, IntentType.MCP_ASK_INFO)
                    and intent_result.tool_name):
                self.logger.info(
                    f"[目标理解] 检查参数完整性 | tool: {intent_result.tool_name} | "
                    f"conv_ctx_len: {len(conversation_context)} | "
                    f"orch_ctx_len: {len(orchestrator_context) if orchestrator_context else 0}"
                )
                param_check = self._check_parameter_completeness(
                    user_input, conversation_context, intent_result,
                    orchestrator_context=orchestrator_context,
                )
                param_complete = param_check.get('is_complete', True)
                self.logger.info(
                    f"[目标理解] 参数检查结果: complete={param_complete} | "
                    f"missing={param_check.get('missing', [])} | "
                    f"reason={param_check.get('reason', '')[:120]}"
                )
                if not param_complete:
                    needs_clarification = True
                    confidence = min(confidence, 0.5)
                    self.logger.info(
                        f"[目标理解] 参数不完整，触发澄清 | 缺失: {param_check.get('missing', [])}"
                    )
                    self.logger.debug(
                        f"[目标理解] 参数检查时的上下文 | "
                        f"conv(last300): {conversation_context[-300:] if conversation_context else '(空)'} | "
                        f"orch(last200): {orchestrator_context[-200:] if orchestrator_context else '(空)'}"
                    )

            # 3b: 置信度低于阈值 → 但如果参数完整且有 orchestrator 记忆，
            #     说明信息足够执行，不需要澄清（覆盖低置信度判断）
            if not needs_clarification and confidence < self.clarification_threshold:
                if param_complete and orchestrator_context:
                    # Orchestrator 记忆提供了足够的上下文信息，提升置信度
                    confidence = max(confidence, self.clarification_threshold)
                    self.logger.info(
                        f"[目标理解] 置信度偏低但orchestrator记忆充分，"
                        f"提升至 {confidence:.2f}，跳过澄清"
                    )
                else:
                    needs_clarification = True
                    self.logger.info(
                        f"[目标理解] 置信度低且无orchestrator记忆支撑 | "
                        f"confidence={confidence:.2f} | threshold={self.clarification_threshold} | "
                        f"param_complete={param_complete} | has_orch_ctx={bool(orchestrator_context)}"
                    )

            if needs_clarification:
                clarification_question = self._generate_clarification(
                    user_input, conversation_context, intent_result
                )
                self.logger.info(
                    f"[目标理解] 生成澄清问题: {clarification_question[:100]} | "
                    f"confidence={confidence:.2f} | intent={intent_result.intent_type.value}"
                )

            # Step 4: 从意图结果中派生目标信息（不再额外调用LLM）
            goal_info = self._derive_goal_info(user_input, intent_result)

            result = GoalUnderstandingResult(
                intent_result=intent_result,
                user_goal=goal_info.get('goal', user_input),
                constraints=goal_info.get('constraints', []),
                success_criteria=goal_info.get('success_criteria', []),
                needs_clarification=needs_clarification,
                clarification_question=clarification_question,
                confidence=confidence,
                reasoning=goal_info.get('reasoning', ''),
                memory_context_used=memory_used,
            )

            if tracer:
                tracer.set_span_attributes({
                    "goal.intent_type": intent_result.intent_type.value,
                    "goal.complexity": intent_result.complexity.value,
                    "goal.confidence": confidence,
                    "goal.needs_clarification": needs_clarification,
                    "goal.memory_used": memory_used,
                })

            return result

    def _retrieve_memory_context(self, query: str) -> str:
        """从长期记忆中检索相关上下文"""
        if not self.long_term_memory:
            return ""
        try:
            results = self.long_term_memory.search_similar_conversations(query, n_results=3)
            if not results:
                return ""
            parts = []
            for r in results:
                parts.append(f"- {r.get('summary', '')}")
            return "\n".join(parts)
        except Exception as e:
            self.logger.warning(f"[目标理解] 长期记忆检索失败: {e}")
            return ""

    def _check_parameter_completeness(
        self,
        user_input: str,
        context: str,
        intent_result: IntentResult,
        orchestrator_context: str = "",
    ) -> Dict[str, Any]:
        """
        检查用户输入是否包含了执行目标工具所需的关键参数

        信息来源（均视为"已提供"）：
        1. 用户当前输入
        2. 对话上下文（短期记忆）
        3. Orchestrator结构化记忆（子Agent产出的实体、方案等）

        Returns:
            {'is_complete': bool, 'missing': [str], 'reason': str}
        """
        # 获取工具的 inputSchema，递归展开嵌套对象
        tool_schema = self._get_tool_schema(intent_result.tool_name)
        schema_text = ""
        if tool_schema:
            params_desc = self._flatten_schema(tool_schema)
            schema_text = "\n".join(params_desc)

        # 构建已知信息段：先提取关键 ID，再附上上下文原文（防止截断丢失关键实体）
        known_info_section = ""
        if orchestrator_context:
            import re as _re
            # 优先提取最后出现的 CARD-xxx / ADDR-xxx，显式列出避免被截断遮蔽
            card_ids = _re.findall(r'CARD-\d+', orchestrator_context)
            addr_ids = _re.findall(r'ADDR-\d+', orchestrator_context)
            key_entity_lines = []
            if card_ids:
                key_entity_lines.append(f"  - 银行卡ID(card_id): {card_ids[-1]}")
            if addr_ids:
                key_entity_lines.append(f"  - 收货地址ID(address_id): {addr_ids[-1]}")
            key_entity_block = (
                "\n已知关键实体ID（上下文已提供，视为参数已知）:\n"
                + "\n".join(key_entity_lines)
                if key_entity_lines else ""
            )
            known_info_section = f"""
Orchestrator已知信息（包含之前子Agent的结果、用户已提供的实体信息）:
{orchestrator_context[:800]}{key_entity_block}
"""

        prompt = f"""你是一个参数完整性检查助手。用户想执行工具 "{intent_result.tool_name}"。

工具参数说明:
{schema_text if schema_text else "（无参数定义，请根据常识判断）"}

对话上下文:
{context[:400] if context else '(无)'}
{known_info_section}
用户输入: "{user_input}"

请严格按照上述"工具参数说明"中列出的参数来判断，用户是否提供了足够的信息。
判断规则：
1. 只检查工具参数说明中列出的参数，不要凭常识添加额外要求
2. 对话上下文和Orchestrator中已有的信息（如用户ID、之前提到的数据）和实体（商品、地址、ID、银行卡、方案等）也算作已提供
3. 用户说"用当前信息"、"用我的信息"、"系统中有我的信息"等表示使用已有数据，不需要再次提供
4. 可选参数（非必需）即使没提供也算完整
5. 如果工具参数是嵌套对象，只要用户提供了对象中的关键字段即可
6. quantity（数量）若未明确说明，默认为1，不算缺失

返回JSON格式:
{{"is_complete": true/false, "missing": ["缺少的工具参数名1", "缺少的工具参数名2"], "reason": "判断原因"}}

只返回JSON。"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.1)
            response = self._clean_json(response)
            result = json.loads(response)
            return {
                'is_complete': result.get('is_complete', True),
                'missing': result.get('missing', []),
                'reason': result.get('reason', ''),
            }
        except Exception as e:
            self.logger.warning(f"[目标理解] 参数完整性检查失败: {e}")
            # 失败时不阻塞，默认通过
            return {'is_complete': True, 'missing': [], 'reason': '检查失败，默认通过'}

    @staticmethod
    def _clean_json(resp: str) -> str:
        """从LLM响应中提取第一个完整的JSON对象"""
        import re
        resp = resp.strip()
        # 提取 ```json ... ``` 中的内容
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', resp, re.DOTALL)
        if code_block:
            resp = code_block.group(1).strip()
        # 用栈匹配花括号，找到第一个完整JSON对象
        brace_start = resp.find('{')
        if brace_start >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(brace_start, len(resp)):
                c = resp[i]
                if esc:
                    esc = False
                    continue
                if c == '\\':
                    esc = True
                    continue
                if c == '"' and not esc:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return resp[brace_start:i + 1]
        return resp

    def _get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具的 inputSchema"""
        if not self.intent_recognizer.mcp_manager:
            return None
        try:
            tools = self.intent_recognizer.mcp_manager.get_available_tools(use_cache=True)
            for tool in tools:
                if tool.get('name') == tool_name:
                    return tool.get('inputSchema', {})
        except Exception:
            pass
        return None

    def _flatten_schema(
        self,
        schema: Dict[str, Any],
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 3,
    ) -> list:
        """
        递归展开 JSON Schema，将嵌套对象的字段平铺为可读的参数列表。
        支持 $ref 引用解析、allOf/anyOf 展开。

        Returns:
            ["- operator_id（必需）: 操作者ID", "- card_data.card_num（可选）: 银行卡号", ...]
        """
        if depth > max_depth:
            return []

        defs = schema.get('$defs', {}) or schema.get('definitions', {})
        required = set(schema.get('required', []))
        properties = schema.get('properties', {})
        lines = []

        for name, info in properties.items():
            full_name = f"{prefix}{name}" if prefix else name
            req_mark = "（必需）" if name in required else "（可选）"
            desc = info.get('description', '')

            # 解析 $ref 引用
            resolved = self._resolve_ref(info, defs)

            # 如果是嵌套对象（有 properties），递归展开
            nested_props = resolved.get('properties')
            if nested_props:
                lines.append(f"- {full_name}{req_mark}: {desc}（复合对象，包含以下子字段）")
                sub_schema = {
                    'properties': nested_props,
                    'required': resolved.get('required', []),
                    '$defs': defs,
                }
                lines.extend(self._flatten_schema(sub_schema, prefix=f"{full_name}.", depth=depth + 1, max_depth=max_depth))
            else:
                # 叶子字段
                type_info = resolved.get('type', '')
                enum_vals = resolved.get('enum')
                extra = ""
                if enum_vals:
                    extra = f"，可选值: {enum_vals}"
                elif type_info:
                    extra = f"，类型: {type_info}"
                lines.append(f"- {full_name}{req_mark}: {desc}{extra}")

        return lines

    @staticmethod
    def _resolve_ref(info: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
        """解析 $ref 引用，支持 allOf/anyOf 中嵌套 $ref"""
        # 直接 $ref
        ref = info.get('$ref')
        if ref and ref.startswith('#/$defs/'):
            ref_name = ref.split('/')[-1]
            return defs.get(ref_name, info)
        if ref and ref.startswith('#/definitions/'):
            ref_name = ref.split('/')[-1]
            return defs.get(ref_name, info)

        # allOf / anyOf 中可能包含 $ref
        for key in ('allOf', 'anyOf'):
            items = info.get(key, [])
            for item in items:
                r = item.get('$ref')
                if r:
                    ref_name = r.split('/')[-1]
                    resolved = defs.get(ref_name)
                    if resolved:
                        return resolved

        return info

    def _generate_clarification(
        self,
        user_input: str,
        context: str,
        intent_result: IntentResult,
    ) -> str:
        """生成澄清问题"""
        # 如果有工具信息，生成更有针对性的澄清（递归展开嵌套参数）
        tool_hint = ""
        if intent_result.tool_name:
            tool_schema = self._get_tool_schema(intent_result.tool_name)
            if tool_schema:
                flat_params = self._flatten_schema(tool_schema)
                # 只取必需参数作为提示
                req_lines = [p for p in flat_params if "（必需）" in p]
                if req_lines:
                    tool_hint = "\n该操作需要的关键信息:\n" + "\n".join(req_lines)

        prompt = f"""用户说了: "{user_input}"

对话上下文:
{context[:500] if context else '(无)'}

当前识别结果:
- 意图类型: {intent_result.intent_type.value}
- 匹配工具: {intent_result.tool_name or '无'}
- 置信度: {intent_result.confidence:.2f}
{tool_hint}

用户的输入信息不够完整，无法直接执行操作。请生成一个友好、简洁的澄清问题，帮助了解用户的具体需求。
要求：
1. 只询问工具实际需要但用户未提供的参数，不要凭常识添加额外要求
2. 语气友好自然
3. 不要暴露内部工具名称

只返回澄清问题本身，不要加任何额外说明。"""

        try:
            return self.llm_client.generate(prompt=prompt, temperature=0.3).strip()
        except Exception as e:
            self.logger.error(f"[目标理解] 生成澄清问题失败: {e}")
            return "您能再详细描述一下您想要做什么吗？"

    @staticmethod
    def _derive_goal_info(user_input: str, intent_result: IntentResult) -> Dict[str, Any]:
        """从意图识别结果中派生目标信息（规则推导，不调用LLM）

        意图识别已返回 reason 字段，包含了 LLM 对用户目标的分析。
        无需再用一次独立的 LLM 调用提取相同信息。
        """
        intent_type = intent_result.intent_type
        reason = intent_result.reason or ""

        # 目标：优先用 _llm_recognize 扩展字段中的 goal，其次用 reason，最后退化用原始输入
        llm_goal = getattr(intent_result, '_goal', '')
        goal = llm_goal if llm_goal else (reason if reason else user_input)

        # 约束：优先用 _llm_recognize 扩展字段中的 constraints
        llm_constraints = getattr(intent_result, '_constraints', [])

        # 约束条件：合并 LLM 提取的约束 + 规则派生的约束
        constraints = list(llm_constraints)  # 先复制 LLM 提取的约束
        success_criteria = []

        if intent_type == IntentType.MCP_EXECUTE:
            constraints.append("操作须基于用户已提供的信息")
            success_criteria.append("工具操作成功执行")
            success_criteria.append("返回结果包含关键ID和状态")
        elif intent_type == IntentType.MCP_ASK_INFO:
            success_criteria.append("清晰列出所需参数及说明")
        elif intent_type in (IntentType.RAG_SIMPLE, IntentType.RAG_ADVANCED):
            constraints.append("回答须基于知识库检索结果")
            success_criteria.append("回答准确且有据可查")
        else:
            success_criteria.append("回复内容准确、友好")

        # 从用户输入中提取显式约束（预算、时间等）
        import re
        budget_match = re.search(r'预算[在是为]?(\d+[万w]?)', user_input)
        if budget_match:
            constraints.append(f"预算限制: {budget_match.group(1)}")

        return {
            "goal": goal,
            "constraints": constraints,
            "success_criteria": success_criteria,
            "reasoning": f"基于意图类型({intent_type.value})规则派生",
        }


class _noop:
    """空上下文管理器"""
    def __enter__(self): return self
    def __exit__(self, *a): pass
