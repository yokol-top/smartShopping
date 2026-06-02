"""
意图识别模块 (Intent Recognizer)

负责识别用户意图，判断应使用RAG、MCP还是普通对话处理。
同时评估任务复杂度（简单/中等/复杂），供任务规划器使用。
"""
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional

from observability import get_tracer


class IntentType(str, Enum):
    """意图类型枚举"""
    GREETING = "greeting"           # 问候
    SIMPLE_CHAT = "simple_chat"     # 简单对话
    RAG_SIMPLE = "rag_simple"       # 简单RAG检索
    RAG_ADVANCED = "rag_advanced"   # 高级RAG检索
    MCP_EXECUTE = "mcp_execute"     # MCP工具执行
    MCP_ASK_INFO = "mcp_ask_info"   # 询问MCP工具信息


class TaskComplexity(str, Enum):
    """任务复杂度枚举"""
    SIMPLE = "simple"       # 简单任务 → 直接ReAct
    MEDIUM = "medium"       # 中等任务 → Plan and Execute
    COMPLEX = "complex"     # 复杂任务 → 精细化拆分 + ReAct


class IntentResult:
    """意图识别结果"""

    def __init__(
        self,
        intent_type: IntentType,
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
        confidence: float = 0.0,
        tool_name: Optional[str] = None,
        reason: str = "",
        requires_rag: bool = False,
        requires_mcp: bool = False,
    ):
        self.intent_type = intent_type
        self.complexity = complexity
        self.confidence = confidence
        self.tool_name = tool_name
        self.reason = reason
        self.requires_rag = requires_rag
        self.requires_mcp = requires_mcp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type.value,
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "requires_rag": self.requires_rag,
            "requires_mcp": self.requires_mcp,
        }

    def __repr__(self):
        return (
            f"IntentResult(type={self.intent_type.value}, "
            f"complexity={self.complexity.value}, "
            f"confidence={self.confidence:.2f}, "
            f"tool={self.tool_name})"
        )


class IntentRecognizer:
    """
    意图识别器

    职责：
    1. 识别用户意图类型（问候 / 简单对话 / RAG / MCP）
    2. 评估任务复杂度（简单 / 中等 / 复杂）
    3. 判断是否需要RAG检索、MCP工具调用或两者结合
    """

    def __init__(self, config: Dict[str, Any], mcp_manager=None, llm_client=None, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.mcp_manager = mcp_manager
        self.llm_client = llm_client

        # 路由配置
        router_config = config.get('router', {})
        self.greeting_keywords = router_config.get('greeting_keywords', [])
        self.complexity_threshold = router_config.get('complexity_threshold', 10)

        # 复杂度判断配置
        complexity_config = config.get('intent', {}).get('complexity', {})
        self.medium_step_threshold = complexity_config.get('medium_step_threshold', 4)
        self.complex_step_threshold = complexity_config.get('complex_step_threshold', 7)

        self.logger.info("IntentRecognizer 初始化完成")

    def recognize(self, query: str, context: str = "") -> IntentResult:
        """
        识别用户意图（主入口）

        Args:
            query: 用户输入
            context: 对话上下文

        Returns:
            IntentResult 意图识别结果
        """
        tracer = get_tracer()

        with tracer.start_span("intent.recognize", {"intent.query": query}):
            self.logger.info(f"意图识别: {query}")
            query_lower = query.lower().strip()

            # 1. 检查是否是问候
            if self._is_greeting(query_lower):
                self.logger.info("意图: 问候")
                result = IntentResult(
                    intent_type=IntentType.GREETING,
                    complexity=TaskComplexity.SIMPLE,
                    confidence=1.0,
                    reason="匹配到问候关键词"
                )
                tracer.set_span_attributes({
                    "intent.type": result.intent_type.value,
                    "intent.complexity": result.complexity.value,
                    "intent.confidence": result.confidence,
                    "intent.method": "keyword",
                })
                return result

            # 2. 使用LLM进行综合意图识别（一次调用判断意图+复杂度）
            llm_result = self._llm_recognize(query, context)
            if llm_result:
                self.logger.info(f"意图识别结果: {llm_result}")
                tracer.set_span_attributes({
                    "intent.type": llm_result.intent_type.value,
                    "intent.complexity": llm_result.complexity.value,
                    "intent.confidence": llm_result.confidence,
                    "intent.tool_name": llm_result.tool_name or "",
                    "intent.method": "llm",
                })
                return llm_result

            # 3. 降级到启发式规则
            result = self._heuristic_recognize(query, query_lower, context)
            tracer.set_span_attributes({
                "intent.type": result.intent_type.value,
                "intent.complexity": result.complexity.value,
                "intent.confidence": result.confidence,
                "intent.method": "heuristic",
            })
            return result

    def _llm_recognize(self, query: str, context: str) -> Optional[IntentResult]:
        """使用LLM进行综合意图识别（一次调用完成意图类型+复杂度判断）

        职责边界：只判断"用户想干什么"（意图类型+复杂度），
        不负责选择具体工具（由TaskPlanner/子Agent负责）。
        """
        if not self.llm_client:
            return None

        tools_section = ""
        if self.mcp_manager:
            tools = self._get_tools_description()
            if tools:
                tools_section = f"\n**可用工具列表（仅 mcp_execute 时使用）**:\n{tools}\n"

        prompt = f"""你是智能购物平台的意图识别助手。分析用户输入，一次性判断：意图类型、任务复杂度。

注意：当 intent_type 为 "mcp_execute" 时，请从"可用工具列表"中选择最匹配的工具名。

{tools_section}对话历史：
{context}

用户输入: {query}

请从以下维度分析：

**一、意图类型** (intent_type):
- "simple_chat": 与购物/商品无关的纯闲聊、通用常识（如"今天天气怎么样"、"你好"）
- "rag_simple": 商品咨询——用户询问商品信息、参数、特点、价格、库存、使用场景等。知识库中已收录所有在售商品的详细描述、评测、FAQ和选购指南。
- "rag_advanced": 复杂商品分析——多商品对比、跨类目选购方案、预算组合推荐等
- "mcp_execute": 业务操作——下单/购买、查询订单、创建/修改用户信息、管理收货地址、管理银行卡、搜索商品等
- "mcp_ask_info": 询问操作所需信息（如"下单需要准备什么"、"怎么添加地址"）

**⚠️ simple_chat 与 rag_simple 的核心区别**:
- 凡是涉及商品（手机、电脑、耳机、平板、家电、运动鞋、手表等）→ 必须选 rag_simple 或 rag_advanced
- 只有完全与购物无关的通用问题 → 才选 simple_chat
- 拿不准时，优先选 rag_simple

**二、任务复杂度** (complexity):
- "simple": 单意图且步骤在4步以内（如：商品咨询、查询订单、查询用户信息）
- "medium": 单意图4-7步（如：推荐+下单、查商品→查用户→下单→汇总）
- "complex": 多阶段任务（如：跨类目组合推荐→用户选择→逐一下单→汇总）

**复杂度校准示例（严格参考）**:
- "iPhone 15 Pro Max怎么样" → simple（商品咨询，rag_simple）
- "推荐一款5000元左右的手机" → simple（rag_simple或rag_advanced）
- "手机和耳机哪个组合好" → simple（rag_advanced，多商品对比但无操作）
- "查询我的订单" → simple（单步查询，mcp_execute）
- "买两部小米14 Ultra" → medium（查商品→查用户信息→下单→汇总 = 4步）
- "帮我推荐一套数码装备然后下单" → medium（推荐→确认→下单）
- "帮我比较三款手机，选最好的那款下单，再配一副耳机" → complex（多阶段）

**三、关键判断规则**:
- 包含"然后"、"接着"、"先...再..."等多步骤连接词 → 复杂度至少medium
- 同时涉及推荐+购买操作 → 复杂度至少medium
- 纯商品咨询/推荐（无购买操作）→ 通常simple
- 涉及条件判断、多实体操作 → complex

**四、置信度校准规则**:
- 用户意图清晰明确 → confidence > 0.8
- 用户意图基本明确但细节不完整（如"推荐手机"未说预算）→ 0.5-0.8
- 用户意图模糊 → confidence < 0.5

返回JSON格式:
{{"intent_type": "...", "complexity": "simple/medium/complex", "confidence": 0.0-1.0, "requires_rag": true/false, "requires_mcp": true/false, "reason": "判断原因", "goal": "用户核心目标（一句话）", "constraints": ["明确约束1", "明确约束2"], "tool_name": "工具名或null（仅intent_type=mcp_execute时填写，必须与可用工具列表完全匹配）"}}

只返回JSON，不要其他说明。"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.1)
            response = self._clean_json_response(response)
            result = json.loads(response)

            intent_type_str = result.get('intent_type', 'simple_chat')
            complexity_str = result.get('complexity', 'simple')
            confidence = float(result.get('confidence', 0.5))
            requires_rag = result.get('requires_rag', False)
            requires_mcp = result.get('requires_mcp', False)
            reason = result.get('reason', '')
            # 新增：缓存 goal 和 constraints 到 intent_result（供 GoalUnderstanding 直接使用）
            goal = result.get('goal', '')
            constraints = result.get('constraints', [])

            # 映射意图类型和复杂度
            intent_type = self._map_intent_type(intent_type_str)
            complexity = self._map_complexity(complexity_str)

            # MCP类意图自动标记 requires_mcp
            if intent_type in (IntentType.MCP_EXECUTE, IntentType.MCP_ASK_INFO):
                requires_mcp = True

            # mcp_execute 时尝试从 LLM 返回中提取工具名
            tool_name_from_llm = result.get('tool_name') or None
            if tool_name_from_llm and not self._validate_tool(tool_name_from_llm):
                self.logger.debug(f"LLM返回的工具名 '{tool_name_from_llm}' 不在可用列表中，忽略")
                tool_name_from_llm = None

            result_obj = IntentResult(
                intent_type=intent_type,
                complexity=complexity,
                confidence=confidence,
                tool_name=tool_name_from_llm,
                reason=reason,
                requires_rag=requires_rag,
                requires_mcp=requires_mcp,
            )
            # 将 goal 和 constraints 附加到 result_obj，供 GoalUnderstanding 使用
            result_obj._goal = goal
            result_obj._constraints = constraints if isinstance(constraints, list) else []
            return result_obj

        except Exception as e:
            self.logger.warning(f"LLM意图识别失败: {e}")
            return None

    def _heuristic_recognize(self, query: str, query_lower: str, context: str) -> IntentResult:
        """基于启发式规则的降级识别"""
        # 短查询视为简单对话
        if len(query) < self.complexity_threshold:
            return IntentResult(
                intent_type=IntentType.SIMPLE_CHAT,
                complexity=TaskComplexity.SIMPLE,
                confidence=0.6,
                reason="查询过短，判定为简单对话"
            )

        # 检查多步骤特征
        complexity = self._heuristic_complexity(query_lower)

        # 检查是否包含RAG特征（公司/业务领域关键词优先级最高）
        shopping_keywords = [
            '商品', '产品', '价格', '多少钱', '售价', '库存', '有货',
            '推荐', '哪款', '哪个好', '对比', '区别', '值得买', '怎么选',
            '手机', '电脑', '笔记本', '耳机', '平板', '家电', '运动鞋', '手表',
            '下单', '购买', '买', '订单', '发货', '退换', '售后', '保修',
            '参数', '配置', '屏幕', '续航', '降噪', '芯片', '性价比',
            'iphone', 'macbook', 'airpods', 'ipad', '华为', '小米', '索尼', '戴森',
        ]
        rag_keywords = ['什么', '如何', '为什么', '怎么', '好不好', '怎么样', '适合']
        has_shopping_hint = any(kw in query_lower for kw in shopping_keywords)
        has_rag_hint = has_shopping_hint or any(kw in query_lower for kw in rag_keywords)

        if has_rag_hint:
            is_complex_query = len(query) > 30 or query.count('？') + query.count('?') > 1
            intent_type = IntentType.RAG_ADVANCED if is_complex_query else IntentType.RAG_SIMPLE
            return IntentResult(
                intent_type=intent_type,
                complexity=complexity,
                confidence=0.7,
                requires_rag=True,
                reason="启发式规则：包含知识检索类关键词"
            )

        return IntentResult(
            intent_type=IntentType.SIMPLE_CHAT,
            complexity=complexity,
            confidence=0.5,
            reason="启发式规则：默认为简单对话"
        )

    def _heuristic_complexity(self, query_lower: str) -> TaskComplexity:
        """基于启发式规则判断任务复杂度"""
        multi_step_patterns = [
            '然后', '接着', '之后', '再', '并且', '同时',
            '先', '最后', '首先', '其次'
        ]
        step_count = sum(1 for p in multi_step_patterns if p in query_lower)

        # 检查是否同时包含查询和操作动词
        query_verbs = ['查询', '查找', '搜索', '获取', '找', '检索']
        action_verbs = ['插入', '添加', '修改', '更新', '删除', '保存', '创建', '执行']
        has_query = any(v in query_lower for v in query_verbs)
        has_action = any(v in query_lower for v in action_verbs)
        if has_query and has_action:
            step_count += 2

        if step_count >= self.complex_step_threshold:
            return TaskComplexity.COMPLEX
        elif step_count >= self.medium_step_threshold:
            return TaskComplexity.MEDIUM
        return TaskComplexity.SIMPLE

    # ---- 辅助方法 ----

    def _is_greeting(self, query: str) -> bool:
        """检查是否是问候语"""
        for keyword in self.greeting_keywords:
            if keyword in query:
                return True
        return False

    def _get_tools_description(self) -> str:
        """获取MCP工具列表描述"""
        if not self.mcp_manager:
            return ""
        available_tools = self.mcp_manager.get_available_tools(use_cache=True)
        if not available_tools:
            return ""
        lines = []
        for i, tool in enumerate(available_tools, 1):
            lines.append(f"{i}. {tool.get('name', '')}: {tool.get('description', '')}")
        return "\n".join(lines)

    def _validate_tool(self, tool_name: str) -> bool:
        """验证工具是否存在"""
        if not self.mcp_manager:
            return False
        available_tools = self.mcp_manager.get_available_tools(use_cache=True)
        return any(t['name'] == tool_name for t in available_tools)

    @staticmethod
    def _clean_json_response(response: str) -> str:
        """清理LLM返回的JSON"""
        response = response.strip()
        if response.startswith('```'):
            response = response.split('```')[1]
            if response.startswith('json'):
                response = response[4:]
        return response.strip()

    @staticmethod
    def _map_intent_type(intent_str: str) -> IntentType:
        """映射意图类型字符串到枚举"""
        mapping = {
            'simple_chat': IntentType.SIMPLE_CHAT,
            'rag_simple': IntentType.RAG_SIMPLE,
            'rag_advanced': IntentType.RAG_ADVANCED,
            'mcp_execute': IntentType.MCP_EXECUTE,
            'mcp_ask_info': IntentType.MCP_ASK_INFO,
            'greeting': IntentType.GREETING,
        }
        return mapping.get(intent_str, IntentType.SIMPLE_CHAT)

    @staticmethod
    def _map_complexity(complexity_str: str) -> TaskComplexity:
        """映射复杂度字符串到枚举"""
        mapping = {
            'simple': TaskComplexity.SIMPLE,
            'medium': TaskComplexity.MEDIUM,
            'complex': TaskComplexity.COMPLEX,
        }
        return mapping.get(complexity_str, TaskComplexity.SIMPLE)

    def explain(self, result: IntentResult) -> str:
        """解释意图识别结果（用户友好）"""
        explanations = {
            IntentType.GREETING: "这是一个简单的问候，我会直接回复。",
            IntentType.SIMPLE_CHAT: "这是一个简单的对话请求，我会基于已有知识直接回答。",
            IntentType.RAG_SIMPLE: "这个问题需要查询知识库，我会使用基础检索来找到相关信息。",
            IntentType.RAG_ADVANCED: "这是一个复杂的问题，我会使用高级RAG技术来提供最准确的答案。",
            IntentType.MCP_EXECUTE: f"检测到工具调用意图，我会调用 {result.tool_name or '相关工具'} 来完成任务。",
            IntentType.MCP_ASK_INFO: f"检测到询问工具信息的意图，我会获取工具 {result.tool_name or '相关工具'} 的参数说明。",
        }
        complexity_desc = {
            TaskComplexity.SIMPLE: "单步执行",
            TaskComplexity.MEDIUM: "多步规划执行",
            TaskComplexity.COMPLEX: "精细化拆分执行",
        }
        base = explanations.get(result.intent_type, "未知意图类型")
        comp = complexity_desc.get(result.complexity, "")
        return f"{base} [复杂度: {comp}] (置信度: {result.confidence:.0%})"
