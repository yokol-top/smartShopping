"""
上下文装配流水线 (Context Pipeline)

职责：装配对话上下文，一次性完成短期记忆、长期记忆、用户偏好的收集，
返回 ContextBundle，供 chat() 各阶段共享使用。
"""
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextBundle:
    """上下文装配结果，只读数据容器，在流水线各阶段传递"""
    context: str          # 对话上下文（短期记忆 + 用户信息）
    long_term: str        # 长期记忆（历史摘要 + 用户偏好，合并为一个字符串）
    session_id: str
    user_id: str
    username: str


class ContextPipeline:
    """
    上下文装配流水线。
    每次 chat() 调用只执行一次，统一管理所有上下文来源。
    """

    def __init__(
        self,
        short_term_memory,
        long_term_memory,        # 可为 None（降级模式）
        orchestrator,            # 可为 None
        config: dict,
        user_id: Optional[str],
        username: Optional[str],
        session_id: str,
        degraded_components: list,
        logger: logging.Logger = None,
    ):
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.orchestrator = orchestrator
        self.config = config
        self.user_id = user_id or ""
        self.username = username or ""
        self.session_id = session_id
        self.degraded_components = degraded_components
        self.logger = logger or logging.getLogger(__name__)

    def build(self, user_input: str) -> ContextBundle:
        """
        装配上下文，返回 ContextBundle。
        步骤：
        1. short_term_memory.get_context_string(last_n=8)
        2. 注入登录用户信息（[当前登录用户] user_id=... username=...）
        3. orchestrator.enrich_context()（sub_agents.enabled=True 时；
           解析"方案一"/"那款手机"等指代词，将解析结果注入 context）
        4. 一次性检索长期记忆 + 用户偏好，合并成 long_term 字符串
        """
        # Step 1: 短期记忆上下文
        context = self.short_term_memory.get_context_string(last_n=8)

        # Step 2: 注入当前登录用户信息
        if self.user_id:
            user_ctx = f"[当前登录用户] user_id={self.user_id}"
            if self.username:
                user_ctx += f", username={self.username}"
            context = user_ctx + "\n" + context

        # Step 3: Orchestrator enrich（解析用户对子Agent结果的引用，如"方案一"→具体商品）
        sub_agent_enabled = self.config.get('sub_agents', {}).get('enabled', True)
        if sub_agent_enabled and self.orchestrator is not None:
            context = self.orchestrator.enrich_context(user_input, context)

        # P1 fix: 降级为 DEBUG 并截断，避免完整对话上下文（含 PII）写入生产日志
        self.logger.debug(f"从短期记忆中获取到的上下文（前200字）：{context[:200]}")

        # Step 4: 一次性检索长期记忆 + 用户偏好
        long_term = self._retrieve_long_term(user_input)
        if long_term:
            self.logger.info("检索到相关历史对话")

        return ContextBundle(
            context=context,
            long_term=long_term,
            session_id=self.session_id,
            user_id=self.user_id,
            username=self.username,
        )

    def _retrieve_long_term(self, query: str) -> str:
        """
        一次性检索：
        1. long_term_memory.search_similar_conversations(query, user_id=self.user_id, n_results=3)
           格式化为 "[相关历史对话记忆]\n1. ...\n2. ..."
        2. long_term_memory.format_user_preferences(self.user_id)
           偏好字符串 prepend 到历史摘要前面
        失败时静默返回空字符串。
        跳过条件：long_term_memory 为 None 或 'long_term_memory' in degraded_components
        """
        if self.long_term_memory is None or 'long_term_memory' in self.degraded_components:
            return ""

        history_part = ""
        pref_part = ""

        # 检索相关历史对话
        try:
            similar_convs = self.long_term_memory.search_similar_conversations(
                query=query,
                user_id=self.user_id,
                n_results=3,
            )
            if similar_convs:
                context_parts = ["[相关历史对话记忆]"]
                for i, conv in enumerate(similar_convs, 1):
                    context_parts.append(f"{i}. {conv['summary']}")
                history_part = "\n".join(context_parts)
        except Exception as e:
            self.logger.error(f"检索长期记忆失败: {e}")

        # 检索用户偏好
        if self.user_id:
            try:
                pref_part = self.long_term_memory.format_user_preferences(self.user_id)
                if pref_part:
                    self.logger.info("已注入用户偏好上下文")
            except Exception as e:
                self.logger.debug(f"加载用户偏好失败（不影响主流程）: {e}")

        # 偏好在前，历史在后
        parts = [p for p in (pref_part, history_part) if p]
        return "\n\n".join(parts).strip()
