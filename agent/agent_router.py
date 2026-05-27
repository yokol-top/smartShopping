"""
Agent路由器

主Agent中的核心组件，负责：
1. 判断用户请求应该路由到哪个子Agent
2. 分派任务给子Agent
3. 处理子Agent的信息请求（从对话上下文/长期记忆中提取）
4. 接收子Agent的执行结果并返回给用户

路由策略：
- 售前咨询类（商品信息、比较、推荐）→ PreSaleAgent
- 功能操作类（创建/查询/修改用户、订单等）→ FunctionalAgent
- 其他类型 → 返回None，由主Agent原有逻辑处理
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Tuple
from .message_bus import (
    AsyncMessageBus, MessageHandler, Message, MessageType,
)
from .presale_agent import PreSaleAgent
from .functional_agent import FunctionalAgent
from .sub_agent_context import SubAgentContextBuilder
from .orchestrator_memory import OrchestratorMemory
from utils import LLMClient
from rag.rag_engine import RAGEngine


class AgentRouter(MessageHandler):
    """主Agent路由器

    作为主Agent的消息处理器，注册为 "main_agent"，
    负责子Agent的生命周期管理和消息路由。
    """

    # 路由目标
    ROUTE_PRESALE = "presale"
    ROUTE_FUNCTIONAL = "functional"
    ROUTE_MAIN = None  # 由主Agent自身处理

    def __init__(
        self,
        llm_client: LLMClient,
        rag_engine: RAGEngine,
        mcp_manager=None,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        # 创建消息总线
        self.bus = AsyncMessageBus(logger=self.logger)

        # 调用 MessageHandler.__init__ 注册主Agent
        super().__init__("main_agent", self.bus, self.logger)

        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.mcp_manager = mcp_manager
        self.config = config or {}

        # 创建子Agent
        self.presale_agent = PreSaleAgent(
            bus=self.bus,
            llm_client=llm_client,
            rag_engine=rag_engine,
            mcp_manager=mcp_manager,
            config=config,
            logger=logger,
        )

        self.functional_agent = FunctionalAgent(
            bus=self.bus,
            llm_client=llm_client,
            mcp_manager=mcp_manager,
            config=config,
            logger=logger,
        )

        # 子Agent上下文构建器（分别管理不同子Agent的上下文预算）
        self.presale_ctx_builder = SubAgentContextBuilder(
            agent_type="presale", config=config, logger=logger,
        )
        self.functional_ctx_builder = SubAgentContextBuilder(
            agent_type="functional", config=config, logger=logger,
        )

        # Orchestrator 结构化任务记忆
        # 从子Agent结果中提取结构化事实（方案、实体、决策），供：
        # - 目标理解层解析用户引用（"方案一"→具体商品）
        # - 子Agent请求信息时由orchestrator从记忆中应答
        # - 下一轮dispatch时enriche上下文
        self.memory = OrchestratorMemory(
            llm_client=llm_client, logger=logger,
        )

        # 事件循环（用于同步->异步桥接）
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.logger.info("[AgentRouter] 初始化完成，已注册子Agent: presale_agent, functional_agent")

    # ============================================================
    # 路由判断
    # ============================================================

    def route(self, user_query: str, intent_result=None, context: str = "") -> Optional[str]:
        """判断请求应路由到哪个子Agent

        Args:
            user_query: 用户输入
            intent_result: 意图识别结果（IntentResult对象）
            context: 对话上下文

        Returns:
            ROUTE_PRESALE / ROUTE_FUNCTIONAL / None(主Agent处理)
        """
        # 1. 基于意图识别结果的快速路由
        if intent_result:
            route = self._route_by_intent(intent_result)
            if route:
                self.logger.info(f"[AgentRouter] 意图路由 → {route}")
                return route

        # 2. 基于关键词的兜底路由
        route = self._route_by_keywords(user_query)
        if route:
            self.logger.info(f"[AgentRouter] 关键词路由 → {route}")
            return route

        self.logger.info("[AgentRouter] 路由 → 主Agent")
        return self.ROUTE_MAIN

    def _route_by_intent(self, intent_result) -> Optional[str]:
        """基于意图识别结果路由"""
        intent_type = getattr(intent_result, 'intent_type', None)
        if intent_type is None:
            return None

        intent_value = intent_type.value if hasattr(intent_type, 'value') else str(intent_type)
        tool_name = getattr(intent_result, 'tool_name', '') or ''

        # 功能操作类工具 → FunctionalAgent
        functional_tools = {
            "create_user", "get_user_detail", "update_user_profile_refined",
            "create_complex_order", "query_order_detail", "list_all_orders",
        }
        if tool_name in functional_tools:
            return self.ROUTE_FUNCTIONAL

        # 商品搜索可能是售前咨询
        if tool_name == "search_products":
            return self.ROUTE_PRESALE

        # 基于意图类型
        if intent_value in ("tool_call", "mcp_call"):
            return self.ROUTE_FUNCTIONAL

        if intent_value == "knowledge_query":
            # 如果查询包含商品关键词，路由到售前
            return None  # 交给关键词路由进一步判断

        return None

    def _route_by_keywords(self, query: str) -> Optional[str]:
        """基于关键词的兜底路由"""
        query_lower = query.lower()

        # 售前咨询关键词
        presale_keywords = [
            "推荐", "哪款", "哪个好", "对比", "区别", "值得买",
            "什么特点", "性价比", "选购", "建议买",
            "多少钱", "价格", "便宜", "贵", "划算",
            "有货", "库存", "补货",
            "参数", "配置", "屏幕", "续航", "降噪", "芯片",
            "适合", "买什么", "怎么选",
        ]
        if any(kw in query_lower for kw in presale_keywords):
            return self.ROUTE_PRESALE

        # 功能操作关键词
        functional_keywords = [
            "创建用户", "注册", "新建用户",
            "创建订单", "下单", "购买", "买一",
            "查询订单", "订单状态", "我的订单",
            "查询用户", "用户信息", "我的信息", "个人信息",
            "修改", "更新", "添加地址", "绑定银行卡",
            "删除", "移除",
        ]
        if any(kw in query_lower for kw in functional_keywords):
            return self.ROUTE_FUNCTIONAL

        return None

    def enrich_context(self, user_query: str, context: str) -> str:
        """供主Agent在目标理解前调用，用orchestrator记忆enriche上下文

        Orchestrator 职责：
        1. 从对话上下文中扫描补充实体（地址、银行卡等可能由主Agent直接创建）
        2. 如果用户引用了之前子Agent的结果（如"方案一"），解析并补充

        Args:
            user_query: 用户输入
            context: 主Agent当前上下文

        Returns:
            enriched后的上下文（可能追加了解析信息）
        """
        # 从对话上下文中扫描实体（补充非子Agent路由产生的实体）
        self.memory.extract_entities_from_context(context)

        # 解析用户引用（仅触发决策记录，不注入上下文；结构化记忆统一由 get_context_for_sub_agent 提供）
        resolved = self.memory.resolve_reference(user_query, context)
        if resolved:
            self.logger.info(f"[AgentRouter] 引用解析完成: {resolved[:100]}（已记录决策，不重复注入上下文）")
        return context

    # ============================================================
    # 任务分派与执行
    # ============================================================

    def dispatch(
        self,
        route: str,
        user_query: str,
        context: str = "",
        user_id: str = "",
        username: str = "",
        tool_name: str = "",
        long_term_context: str = "",
    ) -> str:
        """分派任务给子Agent并获取结果（同步接口）

        这是主Agent调用的入口。内部创建事件循环来运行异步任务。

        Args:
            route: 路由目标 (ROUTE_PRESALE / ROUTE_FUNCTIONAL)
            user_query: 用户请求
            context: 对话上下文
            user_id: 当前用户ID
            username: 当前用户名
            tool_name: 意图识别出的目标工具
            long_term_context: 长期记忆上下文

        Returns:
            子Agent的回复文本
        """
        # 选择对应的上下文构建器
        ctx_builder = (
            self.presale_ctx_builder if route == self.ROUTE_PRESALE
            else self.functional_ctx_builder
        )

        # 构建子Agent的精简上下文（从主Agent完整上下文裁剪/投影）
        ctx_sections = ctx_builder.build_task_context(
            user_query=user_query,
            context=context,
            user_id=user_id,
            username=username,
            long_term_context=long_term_context,
            tool_name=tool_name,
        )

        # Orchestrator 从结构化记忆中提供上下文（非原始文本注入）
        orchestrator_ctx = self.memory.get_context_for_sub_agent()
        if orchestrator_ctx:
            ctx_sections['main_replies'] = orchestrator_ctx
            self.logger.info(
                f"[AgentRouter] 注入orchestrator结构化记忆 ({len(orchestrator_ctx)}字符)"
            )

        # 解析用户引用（如"方案一"→具体商品），enriche 任务描述
        resolved = self.memory.resolve_reference(user_query, context)
        if resolved:
            # 将解析结果追加到task_brief，让子Agent明确知道用户要什么
            current_brief = ctx_sections.get('task_brief', '')
            ctx_sections['task_brief'] = current_brief + f"\n[引用解析] {resolved}"
            self.logger.info(f"[AgentRouter] 引用解析成功: {resolved[:100]}")

        # 组装为单一字符串（兼容子Agent现有接口）
        assembled_context = ctx_builder.assemble(ctx_sections)

        task = {
            "user_query": user_query,
            "context": assembled_context,
            "context_sections": ctx_sections,  # 分区结构（子Agent可按需使用）
            "user_id": user_id,
            "username": username,
            "tool_name": tool_name,
            "long_term_context": long_term_context,
            "_ctx_builder": ctx_builder,  # 传递构建器供子Agent动态更新上下文
        }

        try:
            # 使用事件循环运行异步任务
            result = self._run_async(self._dispatch_async(route, task))
            response_text = result.get("response", "子Agent未返回结果")

            # Orchestrator 从子Agent结果中提取结构化信息存入记忆
            self.memory.update_from_sub_agent_result(
                route=route, query=user_query, response=response_text,
            )

            return response_text
        except Exception as e:
            self.logger.error(f"[AgentRouter] 分派任务失败: {e}")
            return f"任务分派失败: {str(e)}"

    async def _dispatch_async(self, route: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """异步分派任务"""
        # 在当前事件循环中重建所有Queue（解决跨事件循环绑定问题）
        self.bus.prepare()

        # 构造任务消息
        target_agent = (
            self.presale_agent if route == self.ROUTE_PRESALE
            else self.functional_agent
        )

        task_msg = Message(
            msg_type=MessageType.TASK_DISPATCH,
            sender=self.agent_id,
            receiver=target_agent.agent_id,
            content=task,
        )

        # 启动信息请求处理器（异步监听子Agent的请求）
        handler_task = asyncio.create_task(
            self._handle_sub_agent_requests(task, timeout=60.0)
        )

        # 执行子Agent任务
        try:
            result = await asyncio.wait_for(
                target_agent.run_task_loop(task_msg),
                timeout=90.0,  # 总超时
            )
        except asyncio.TimeoutError:
            self.logger.error(f"[AgentRouter] 子Agent执行超时")
            result = {"success": False, "response": "操作超时，请稍后再试。"}
        finally:
            handler_task.cancel()
            try:
                await handler_task
            except asyncio.CancelledError:
                pass

        return result

    async def _handle_sub_agent_requests(self, original_task: Dict, timeout: float = 60.0):
        """持续监听并处理子Agent发来的信息请求

        这个协程在任务执行期间一直运行，处理子Agent的补充信息请求。
        当子Agent向主Agent的inbox发送请求消息时，这里负责响应。
        """
        import time
        start = time.time()

        while time.time() - start < timeout:
            msg = await self.bus.check_inbox(self.agent_id, timeout=0.5)
            if not msg:
                continue

            if msg.msg_type == MessageType.REQUEST:
                # 子Agent请求补充信息
                question = msg.content.get("question", "")
                self.logger.info(
                    f"[AgentRouter] 收到 {msg.sender} 的信息请求: {question[:60]}"
                )
                # 在线程池中执行LLM调用，避免阻塞事件循环导致死锁
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None, self._answer_sub_agent_question, question, original_task
                )
                await self.send_reply(msg, {"answer": answer})

            elif msg.msg_type == MessageType.TASK_RESULT:
                # 子Agent返回了任务结果（通常由run_task_loop直接返回）
                self.logger.info(f"[AgentRouter] 收到 {msg.sender} 的任务结果")
                break

    def _answer_sub_agent_question(self, question: str, task: Dict[str, Any]) -> str:
        """回答子Agent的信息请求

        Orchestrator 信息提供优先级：
        1. 从结构化记忆中直接命中 → 无需LLM，立即返回
        2. 从对话上下文+记忆中通过LLM提取 → 有答案则返回
        3. 都找不到 → 返回"无额外信息"（子Agent可选择向用户澄清）

        Args:
            question: 子Agent的问题
            task: 原始任务信息

        Returns:
            回答文本
        """
        # 优先级1：从 orchestrator 结构化记忆中直接查找
        memory_answer = self.memory.answer_sub_agent_question(question)
        if memory_answer:
            self.logger.info(
                f"[AgentRouter] 从orchestrator记忆中直接应答: {memory_answer[:80]}"
            )
            return memory_answer

        # 优先级2：结合记忆和上下文，用LLM提取
        context = task.get("context", "")
        user_id = task.get("user_id", "")
        username = task.get("username", "")
        long_term = task.get("long_term_context", "")

        # 将结构化记忆也作为上下文的一部分
        memory_ctx = self.memory.get_context_for_sub_agent()

        prompt = f"""你是主Agent的信息提取助手。子Agent在执行任务时缺少某些信息，请从已有上下文中提取。

子Agent的问题：{question}

已知信息：
- 当前用户：user_id={user_id}, username={username}
- 对话上下文：{context[:800]}
{f'- 历史记忆：{long_term[:300]}' if long_term else ''}
{f'- Orchestrator记忆：{memory_ctx[:500]}' if memory_ctx else ''}

请直接回答子Agent的问题。如果从上下文中找不到答案，回复"无额外信息"。"""

        try:
            answer = self.llm_client.generate(prompt=prompt, temperature=0.2)
            return answer.strip()
        except Exception as e:
            self.logger.error(f"[AgentRouter] 回答子Agent失败: {e}")
            return "无额外信息"

    # ============================================================
    # 任务节点执行（供TaskPlanner委托子Agent执行单步工具调用）
    # ============================================================

    def execute_tool_step(
        self,
        tool_name: str,
        step_description: str,
        user_query: str,
        context: str = "",
        user_id: str = "",
        username: str = "",
        long_term_context: str = "",
    ) -> Optional[str]:
        """作为任务节点执行单个工具调用（供主Agent的TaskPlanner在规划执行时委托）

        主Agent在Plan-and-Execute流程中，若某个步骤涉及子Agent域内的工具，
        可通过此方法将该步骤委托给对应子Agent执行，子Agent仅作为执行节点。

        Args:
            tool_name: 工具名称
            step_description: 当前步骤描述（作为子Agent的任务上下文）
            user_query: 原始用户请求
            context: 当前对话上下文（含前置步骤结果）
            user_id: 当前用户ID
            username: 当前用户名
            long_term_context: 长期记忆上下文

        Returns:
            工具执行结果文本，若工具不属于任何子Agent域则返回None
        """
        route = self._determine_tool_route(tool_name)
        if not route:
            return None

        self.logger.info(
            f"[AgentRouter] TaskPlanner委托子Agent执行工具步骤: "
            f"{tool_name} → {route}"
        )

        return self.dispatch(
            route=route,
            user_query=step_description,
            context=context,
            user_id=user_id,
            username=username,
            tool_name=tool_name,
            long_term_context=long_term_context,
        )

    def _determine_tool_route(self, tool_name: str) -> Optional[str]:
        """判断工具属于哪个子Agent的职责域

        Returns:
            路由目标，或None（不属于任何子Agent）
        """
        # 检查功能子Agent白名单
        if (FunctionalAgent.ALLOWED_TOOLS is not None
                and tool_name in FunctionalAgent.ALLOWED_TOOLS):
            return self.ROUTE_FUNCTIONAL

        # 检查售前子Agent白名单
        if (PreSaleAgent.ALLOWED_TOOLS is not None
                and tool_name in PreSaleAgent.ALLOWED_TOOLS):
            return self.ROUTE_PRESALE

        return None

    # ============================================================
    # 辅助方法
    # ============================================================

    def _run_async(self, coro):
        """在同步环境中运行异步协程"""
        try:
            loop = asyncio.get_running_loop()
            # 已在事件循环中（不应该发生在正常流程中）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=120)
        except RuntimeError:
            # 没有运行中的事件循环，创建新的
            return asyncio.run(coro)
