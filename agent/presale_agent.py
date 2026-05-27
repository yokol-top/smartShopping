"""
售前客服子Agent

职责：
- 回答商品基本信息咨询（功能、规格、价格、库存等）
- 商品对比和选购建议
- 基于RAG知识库提供专业的售前咨询

工作模式：
1. 优先从商品FAQ向量库（product_faq）中检索答案
2. 如果RAG无结果，调用 search_products MCP工具获取实时数据
3. 如果缺少用户偏好等信息，向主Agent请求补充
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from .sub_agent_base import SubAgentBase
from .message_bus import AsyncMessageBus
from utils import LLMClient
from rag.rag_engine import RAGEngine


class PreSaleAgent(SubAgentBase):
    """售前客服子Agent"""

    AGENT_TYPE = "presale"
    AGENT_NAME = "PreSaleAgent"

    # 店铺经营类目
    BUSINESS_CATEGORIES = [
        "手机", "笔记本电脑", "耳机", "平板电脑", "家电", "运动鞋", "智能手表",
    ]

    # 类目→关键词映射（用于查询改写和MCP搜索参数提取）
    CATEGORY_KEYWORDS = {
        "手机": ["手机", "phone", "iphone", "华为", "小米"],
        "笔记本电脑": ["笔记本", "电脑", "laptop", "macbook", "thinkpad"],
        "耳机": ["耳机", "airpods", "降噪", "头戴"],
        "平板电脑": ["平板", "ipad", "pad"],
        "家电": ["吸尘器", "冰箱", "家电", "戴森", "海尔"],
        "运动鞋": ["鞋", "跑步", "运动", "nike", "adidas"],
        "智能手表": ["手表", "watch"],
    }

    # 售前子Agent仅允许使用商品相关的MCP工具
    ALLOWED_TOOLS = [
        "search_products",
    ]

    # 需要查询知识库的关键词（用户在问商品具体信息）
    _KNOWLEDGE_KEYWORDS = [
        "怎么样", "好不好", "好用吗", "质量", "评价", "口碑", "缺点", "优点",
        "区别", "对比", "差别", "不同", "和.*比",
        "参数", "配置", "规格", "屏幕", "续航", "芯片", "电池", "内存",
        "保修", "售后", "退换", "发货", "快递",
        "功能", "特点", "支持", "兼容", "防水",
    ]
    # 规划/推荐类关键词（不需要知识库，需要MCP实时数据）
    _PLANNING_KEYWORDS = [
        "推荐", "方案", "预算", "搭配", "组合", "买什么", "怎么选", "选购",
        "帮我选", "帮我挑", "套装", "清单",
    ]

    def __init__(
            self,
            bus: AsyncMessageBus,
            llm_client: LLMClient,
            rag_engine: RAGEngine,
            mcp_manager=None,
            config: Dict[str, Any] = None,
            logger: logging.Logger = None,
    ):
        super().__init__(
            agent_id="presale_agent",
            bus=bus,
            llm_client=llm_client,
            config=config,
            logger=logger,
        )
        self.rag_engine = rag_engine
        self.mcp_manager = mcp_manager

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理售前咨询任务

        Args:
            task: {
                "user_query": 用户问题,
                "context": 已组装的上下文（由SubAgentContextBuilder构建）,
                "context_sections": 分区上下文dict（可按需使用）,
                "user_id": 当前用户ID,
                "_ctx_builder": 上下文构建器引用
            }

        Returns:
            {"success": bool, "response": str}
        """
        user_query = task.get("user_query", "")
        context = task.get("context", "")
        user_id = task.get("user_id", "")
        ctx_sections = task.get("context_sections", {})
        ctx_builder = task.get("_ctx_builder")

        self.logger.info(f"{self._log_tag} 处理咨询: {user_query[:60]}")

        # Step 0: 上下文感知查询改写（解决短查询/指代不明问题）
        effective_query = self._rewrite_query_with_context(user_query, context)
        if effective_query != user_query:
            self.logger.info(f"{self._log_tag} 查询改写: '{user_query}' → '{effective_query}'")

        # Step 1: 判断是否需要查询知识库
        # 只有用户在询问商品具体信息（参数、评价、对比等）时才检索RAG
        # 规划/推荐类查询只需MCP实时商品数据，无需知识库
        need_rag = self._is_knowledge_query(effective_query)
        rag_results = []
        if need_rag:
            rag_results = self._search_product_knowledge(effective_query, context)
        else:
            self.logger.info(f"{self._log_tag} 跳过RAG检索（规划/推荐类查询）")

        # Step 2: 调用MCP获取实时数据
        # 触发条件：RAG结果不足 或 查询涉及具体商品的价格/库存/参数
        mcp_data = ""
        if self._should_call_mcp(effective_query, rag_results):
            mcp_data = await self._search_products_via_mcp(effective_query)

        # Step 2.5: 动态更新领域上下文分区
        if ctx_builder and ctx_sections is not None:
            ctx_sections['domain_context'] = ctx_builder.build_domain_context(
                rag_results=rag_results,
                mcp_data=mcp_data,
            )

        # Step 3: 如果缺少关键信息（如用户预算、使用场景），向主Agent请求
        supplementary_info = ""
        if self._needs_more_info(user_query, rag_results, mcp_data):
            supplementary_info = await self._request_supplementary_info(user_query, context)
            # 动态追加主Agent回复到上下文
            if ctx_builder and ctx_sections is not None and supplementary_info:
                ctx_sections['main_replies'] = ctx_builder.append_main_reply(
                    ctx_sections.get('main_replies', ''),
                    "用户偏好信息",
                    supplementary_info,
                )

        # Step 4: 综合所有信息生成回答（使用结构化上下文）
        final_context = ctx_builder.assemble(ctx_sections) if ctx_builder and ctx_sections else context
        response = self._generate_response(
            user_query, rag_results, mcp_data, supplementary_info, final_context
        )

        return {"success": True, "response": response}

    def _rewrite_query_with_context(self, query: str, context: str) -> str:
        """上下文感知查询改写

        当用户查询较短或存在指代（如"它"、"这个"、省略主语）时，
        结合对话上下文补全查询，使RAG检索和MCP搜索更精确。
        """
        # 如果查询已经足够长且明确，不需要改写
        all_keywords = [kw for kws in self.CATEGORY_KEYWORDS.values() for kw in kws]
        all_keywords.extend(["MacBook", "iPhone", "iPad", "AirPods", "戴森", "海尔", "Nike", "索尼", "联想"])
        if len(query) > 15 and any(kw in query.lower() for kw in all_keywords):
            return query

        # 短查询或无明确主语时，用LLM结合上下文改写
        if not context:
            return query

        # 判断是否需要改写：查询很短 或 包含指代词/省略主语
        needs_rewrite = (
                len(query) <= 15
                or any(w in query for w in ["它", "这个", "那个", "这款", "那款"])
                or (not any(c.isascii() and c.isalpha() for c in query)
                    and not any(kw in query.lower() for kw in all_keywords))
        )

        if not needs_rewrite:
            return query

        prompt = f"""你是查询改写助手。用户在咨询商品时的新问题可能省略了主语或使用了指代词。
请结合对话上下文，将用户的新问题改写为一个完整、明确的商品咨询问题。

对话上下文（最近部分）：
{context[-500:]}

用户新问题：{query}

要求：
1. 只返回改写后的问题，不要其他内容
2. 补全省略的商品名称
3. 保持用户原始意图不变
4. 如果问题已经很明确，原样返回即可"""

        try:
            rewritten = self.llm_client.generate(prompt=prompt, temperature=0.1)
            rewritten = rewritten.strip().strip('"').strip("'")
            # 简单校验：改写结果不能太长也不能太短
            if 3 < len(rewritten) < 100:
                return rewritten
        except Exception as e:
            self.logger.warning(f"{self._log_tag} 查询改写失败: {e}")

        return query

    def _is_knowledge_query(self, query: str) -> bool:
        """判断是否为需要查询知识库的商品咨询

        Returns:
            True  → 用户在问商品具体信息（参数、评价、对比等），需要RAG
            False → 规划/推荐/选购方案类，只需MCP实时数据
        """
        import re
        query_lower = query.lower()

        # 明确的规划/推荐类 → 不需要知识库
        if any(kw in query_lower for kw in self._PLANNING_KEYWORDS):
            return False

        # 明确的知识咨询类 → 需要知识库
        for kw in self._KNOWLEDGE_KEYWORDS:
            if re.search(kw, query_lower):
                return True

        # 默认：查询知识库（保守策略）
        return True

    def _should_call_mcp(self, query: str, rag_results: list) -> bool:
        """判断是否应该调用MCP获取实时商品数据

        触发条件（满足任一即可）：
        1. RAG结果不足（< 2条）
        2. 查询涉及具体商品的价格、库存、实时参数
        3. 查询是推荐/预算方案类（需要全面的商品数据）
        """
        # 条件1：RAG结果不足
        if not rag_results or len(rag_results) < 2:
            return True

        # 条件2：查询涉及价格、库存等实时信息
        realtime_keywords = ["价格", "多少钱", "售价", "库存", "有货", "现货", "补货"]
        if any(kw in query for kw in realtime_keywords):
            return True

        # 条件3：推荐/预算方案类查询（需要完整商品列表和价格）
        recommend_keywords = ["推荐", "买什么", "预算", "方案", "搭配", "组合"]
        if any(kw in query for kw in recommend_keywords):
            return True

        return False

    def _search_product_knowledge(self, query: str, context: str) -> list:
        """从RAG知识库检索商品FAQ"""
        try:
            results = self.rag_engine.retrieve(
                query=query,
                context=context,
                top_k=3,
                use_advanced=False,  # 售前查询用简单检索即可
            )
            self.logger.info(f"{self._log_tag} RAG检索到 {len(results)} 条结果")
            return results
        except Exception as e:
            self.logger.error(f"{self._log_tag} RAG检索失败: {e}")
            return []

    async def _search_products_via_mcp(self, query: str) -> str:
        """通过MCP工具搜索商品实时信息"""
        if not self.mcp_manager:
            return ""

        try:
            # 权限检查
            if not self.is_tool_allowed("search_products"):
                self.logger.warning(f"{self._log_tag} search_products 不在权限白名单内")
                return ""

            # 从用户查询中提取搜索关键词
            keywords = self._extract_search_params(query)
            self.logger.info(f"{self._log_tag} MCP搜索: {keywords}")

            # 查找 search_products 所在的服务器（仅在白名单内查找）
            all_tools = self.filter_tools(self.mcp_manager.get_available_tools(use_cache=True))
            server_name = None
            for t in all_tools:
                if t.get("name") == "search_products":
                    server_name = t.get("server")
                    break

            if not server_name:
                self.logger.warning(f"{self._log_tag} search_products 工具不可用")
                return ""

            # 在线程池中调用，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.mcp_manager.call_tool(
                    server_name=server_name,
                    tool_name="search_products",
                    parameters=keywords,
                )
            )
            if result and not (isinstance(result, dict) and "error" in result):
                return str(result)
        except Exception as e:
            self.logger.error(f"{self._log_tag} MCP搜索失败: {e}")
        return ""

    def _extract_search_params(self, query: str) -> Dict[str, Any]:
        """从用户查询中提取商品搜索参数"""
        params = {}

        query_lower = query.lower()
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                params["category"] = cat
                break

        # 提取价格信息
        import re
        price_match = re.search(r'(\d+)\s*[元块万w]', query_lower)
        if price_match:
            price_val = float(price_match.group(1))
            if '万' in query_lower or 'w' in query_lower:
                price_val *= 10000
            params["max_price"] = price_val

        # 提取品牌关键词
        brand_keywords = ["苹果", "apple", "华为", "小米", "联想", "索尼", "戴森", "海尔", "nike", "adidas"]
        for brand in brand_keywords:
            if brand in query_lower:
                params["keyword"] = brand
                break

        # 如果没有提取到任何参数，用原始查询做关键词搜索
        if not params:
            # 取查询中的核心词
            params["keyword"] = query[:20]

        return params

    def _needs_more_info(self, query: str, rag_results: list, mcp_data: str) -> bool:
        """判断是否需要向主Agent请求更多信息"""
        # 如果已经有足够的检索结果，不需要补充
        if rag_results and len(rag_results) >= 2:
            return False
        if mcp_data and len(mcp_data) > 50:
            return False

        # 查询中包含需要用户偏好的关键词
        need_pref_keywords = ["推荐", "建议", "买什么", "哪个好", "怎么选", "对比"]
        return any(kw in query for kw in need_pref_keywords)

    async def _request_supplementary_info(self, query: str, context: str) -> str:
        """向主Agent请求补充信息"""
        question = (
            f"用户正在咨询商品: '{query}'。"
            f"请提供用户之前表达过的偏好信息（预算、品牌偏好、使用场景等），"
            f"如果没有，请返回'无额外偏好信息'。"
        )
        result = await self.request_info_from_main(question, timeout=15.0)
        return result or ""

    def _generate_response(
            self,
            query: str,
            rag_results: list,
            mcp_data: str,
            supplementary_info: str,
            context: str,
    ) -> str:
        """综合所有信息使用LLM生成回答"""
        # 构建RAG上下文
        rag_context = ""
        if rag_results:
            rag_parts = []
            for i, r in enumerate(rag_results, 1):
                doc = r.get("document", r.get("text", ""))
                meta = r.get("metadata", {})
                product_name = meta.get("product_name", "")
                product_id = meta.get("product_id", "")
                prefix = f"[{product_id}] {product_name}" if product_id else product_name
                rag_parts.append(f"{i}. {prefix}: {doc}")
            rag_context = "\n".join(rag_parts)

        # 构建提示词
        categories_str = "、".join(self.BUSINESS_CATEGORIES)
        prompt_parts = [
            f"你是一位专业的售前客服顾问，服务于一家经营「{categories_str}」的综合数码商城。",
            "请基于以下信息，为用户提供准确、简练的回答。",
            "重要：只推荐以上类目范围内的商品，不要推荐本店未经营的品类。",
            "风格要求：回答尽量精炼，突出核心信息，避免冗长铺垫和过度修饰。",
        ]

        # 注入对话上下文，帮助LLM理解指代关系和用户背景
        if context:
            prompt_parts.append(f"\n对话上下文（请注意用户之前讨论的商品和偏好）：\n{context[-600:]}")

        prompt_parts.append(f"\n用户当前问题：{query}")

        if rag_context:
            prompt_parts.append(f"\n商品知识库信息：\n{rag_context}")

        if mcp_data:
            prompt_parts.append(f"\n实时商品数据：\n{mcp_data}")

        if supplementary_info and supplementary_info != "无额外偏好信息":
            prompt_parts.append(f"\n用户偏好补充：\n{supplementary_info}")

        # 从主Agent获取的补充上下文
        extra_ctx = self.get_accumulated_context()
        if extra_ctx:
            prompt_parts.append(f"\n{extra_ctx}")

        prompt_parts.extend([
            "",
            "回答要求：",
            "1. 基于提供的信息回答，不要编造不存在的商品或参数",
            "2. 回答要简练，每个方案用1-2行概括核心组合和总价，不要长篇大论",
            "3. 推荐商品时必须标注商品ID，格式: [P001] 商品名 ¥价格",
            "4. 多方案时用简洁列表，每方案包含: 商品列表、合计价格、一句话亮点",
            f"5. 从本店类目（{categories_str}）中跨类目组合推荐",
            "6. 不要过度修饰和铺垫，直接给出推荐结果",
        ])

        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.5)
            return response.strip()
        except Exception as e:
            self.logger.error(f"{self._log_tag} LLM生成失败: {e}")
            # 降级：直接拼接RAG结果
            if rag_context:
                return f"根据我们的商品信息：\n{rag_context}"
            return "抱歉，暂时无法获取商品信息，请稍后再试。"
