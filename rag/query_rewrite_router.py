"""
查询重写路由器 (Query Rewrite Router)

对每条 RAG 查询做分级判断，决定走哪个重写层级，避免对所有查询无差别执行
完整 LLM 重写。

四个层级（互斥，命中即退出）：
  Level 0  SKIP     — 不重写，直接检索
  Level 1  COREF    — 轻量指代消解（"它/这款/那个" → 具体实体）
  Level 2  SEMANTIC — 完整语义重写（语义模糊 / 口语化查询）
  Level 3  DEEP     — HyDE + 多查询（高召回率场景，由 intent 驱动）

决策顺序：
  1. 是否符合 SKIP 条件？→ Level 0（退出）
  2. 调用方是否要求 DEEP（rag_advanced intent）？→ Level 3（退出）
  3. 查询含指代词 + 有上下文？→ Level 1（退出）
  4. 查询语义模糊？→ Level 2（退出）
  5. 默认 → Level 0（退出）
"""

import logging
from enum import Enum
from typing import List, Optional


class RewriteLevel(str, Enum):
    SKIP = "skip"          # Level 0: 不重写
    COREF = "coref"        # Level 1: 轻量指代消解
    SEMANTIC = "semantic"  # Level 2: 完整语义重写
    DEEP = "deep"          # Level 3: HyDE / 多查询


class RewriteResult:
    """路由结果：包含选中的层级和最终用于检索的查询列表"""

    def __init__(self, level: RewriteLevel, queries: List[str], reason: str = ""):
        self.level = level
        # 去重后的查询列表（Level 3 可能有多条，其余通常 1 条）
        self.queries = list(dict.fromkeys(q for q in queries if q))
        self.reason = reason

    def __repr__(self):
        return (
            f"RewriteResult(level={self.level.value}, "
            f"queries={self.queries}, reason='{self.reason}')"
        )


class QueryRewriteRouter:
    """查询重写路由器

    职责：
    1. 判断当前查询属于哪个层级
    2. 调用对应的重写组件
    3. 返回最终用于检索的查询列表

    组件依赖（均可为 None，None 时对应层级降级到 SKIP）：
      coref_resolver   — Level 1  (CoRefResolver)
      semantic_rewriter — Level 2  (QueryRewriter)
      hyde             — Level 3  (HyDE，生成假设性文档)
      multi_query_gen  — Level 3  (MultiQueryGenerator，生成多角度查询)
    """

    # ----------------------------------------------------------------
    # 触发词表（Level 1：指代词 / 示指词）
    # ----------------------------------------------------------------
    _COREF_WORDS: List[str] = [
        # 近指
        '这个', '这款', '这台', '这种', '这些', '这里', '这边',
        # 远指
        '那个', '那款', '那台', '那种', '那些', '那里', '那边',
        # 人称代词
        '它', '它们', '他', '他们', '她', '她们',
        # 时间/位置指代
        '上面', '上述', '之前', '刚才', '前面', '前面说的',
        '刚刚说的', '刚说的', '提到的', '说的那个',
        # 疑问指代（在多轮中指代前文）
        '哪款', '哪个', '哪台', '哪种',
        # 价格/属性指代
        '什么价格', '多少钱', '多少价', '怎么买', '在哪买',
    ]

    # ----------------------------------------------------------------
    # 触发词表（Level 2：模糊评价词）
    # ----------------------------------------------------------------
    _AMBIGUITY_WORDS: List[str] = [
        '怎么样', '好不好', '好用吗', '值得买', '值不值',
        '怎么选', '哪个好', '哪款好', '哪个更好', '哪款更好',
        '推荐一下', '有没有推荐', '如何选', '选哪个', '选哪款',
        '靠谱吗', '质量怎么样', '口碑怎么样',
    ]

    # ----------------------------------------------------------------
    # Level 0 判断：疑问词列表（用于区分"纯关键词"和"真实疑问"）
    # ----------------------------------------------------------------
    _QUESTION_WORDS: List[str] = [
        '怎', '哪', '什么', '为什么', '如何', '是否', '能否',
        '多少', '几', '吗', '呢', '嘛',
    ]

    def __init__(
        self,
        coref_resolver=None,
        semantic_rewriter=None,
        hyde=None,
        multi_query_gen=None,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.coref_resolver = coref_resolver
        self.semantic_rewriter = semantic_rewriter
        self.hyde = hyde
        self.multi_query_gen = multi_query_gen
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Level 3 多查询数量
        self._multi_query_num: int = (
            self.config.get('multi_query', {}).get('num_queries', 3)
        )

        self.logger.info("[QueryRewriteRouter] 初始化完成")

    # ----------------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------------

    def route_and_rewrite(
        self,
        query: str,
        context: str = "",
        use_deep: bool = False,
    ) -> RewriteResult:
        """路由并执行重写，返回最终用于检索的查询列表。

        Args:
            query:    用户原始查询
            context:  对话历史摘要（短期记忆）
            use_deep: 是否允许 Level 3（由上层根据 rag_advanced intent 传入）

        Returns:
            RewriteResult(level, queries, reason)
        """
        # ── Level 3: DEEP（优先级最高，由调用方 intent 驱动，不靠规则触发）──
        # 放在 SKIP 之前：调用方显式要求高召回率时，不受 skip 规则拦截
        if use_deep:
            queries = self._build_deep_queries(query)
            self.logger.info(
                f"[Router] L3 DEEP | queries={len(queries)} | q='{query[:40]}'"
            )
            return RewriteResult(RewriteLevel.DEEP, queries, "高召回率检索")

        # ── Level 0: SKIP ──────────────────────────────────────────
        if self._should_skip(query, context):
            self.logger.debug(
                f"[Router] L0 SKIP | reason=no_rewrite_needed | q='{query[:40]}'"
            )
            return RewriteResult(RewriteLevel.SKIP, [query], "无需重写")

        # ── Level 1: COREF（含指代词 + 有上下文）──────────────────
        if context and self._has_coref(query):
            resolved = self._resolve_coref(query, context)
            self.logger.info(
                f"[Router] L1 COREF | '{query[:30]}' → '{resolved[:30]}'"
            )
            return RewriteResult(RewriteLevel.COREF, [resolved], "指代消解")

        # ── Level 2: SEMANTIC（语义模糊）──────────────────────────
        if self._is_ambiguous(query):
            rewritten = self._semantic_rewrite(query, context)
            self.logger.info(
                f"[Router] L2 SEMANTIC | '{query[:30]}' → '{rewritten[:30]}'"
            )
            return RewriteResult(RewriteLevel.SEMANTIC, [rewritten], "语义重写")

        # ── 默认: SKIP ─────────────────────────────────────────────
        self.logger.debug(
            f"[Router] L0 SKIP (default) | q='{query[:40]}'"
        )
        return RewriteResult(RewriteLevel.SKIP, [query], "规则未触发")

    # ----------------------------------------------------------------
    # 各层级处理
    # ----------------------------------------------------------------

    def _build_deep_queries(self, query: str) -> List[str]:
        """Level 3：构建 HyDE + 多查询列表"""
        queries = [query]

        # HyDE：生成假设性文档，用文档向量检索而非问题向量
        if self.hyde:
            try:
                hyp_doc = self.hyde.generate_hypothetical_document(query)
                if hyp_doc and hyp_doc != query:
                    queries.append(hyp_doc)
            except Exception as e:
                self.logger.warning(f"[Router] HyDE 失败，跳过: {e}")

        # 多查询：从不同角度扩展
        if self.multi_query_gen:
            try:
                extra = self.multi_query_gen.generate_queries(query, self._multi_query_num)
                queries.extend(extra)
            except Exception as e:
                self.logger.warning(f"[Router] 多查询生成失败，跳过: {e}")

        return queries

    def _resolve_coref(self, query: str, context: str) -> str:
        """Level 1：调用 CoRefResolver，失败时原样返回"""
        if self.coref_resolver:
            try:
                return self.coref_resolver.resolve(query, context)
            except Exception as e:
                self.logger.warning(f"[Router] CoRef 解析失败，使用原始查询: {e}")
        return query

    def _semantic_rewrite(self, query: str, context: str) -> str:
        """Level 2：调用 QueryRewriter，失败时原样返回"""
        if self.semantic_rewriter:
            try:
                return self.semantic_rewriter.rewrite_query(query, context)
            except Exception as e:
                self.logger.warning(f"[Router] 语义重写失败，使用原始查询: {e}")
        return query

    # ----------------------------------------------------------------
    # 判断方法（供外部测试）
    # ----------------------------------------------------------------

    def _should_skip(self, query: str, context: str) -> bool:
        """Level 0 条件（满足任一即跳过重写）"""
        # 无上下文 → 第一轮，无指代可消解，也不知道语义背景
        if not context or not context.strip():
            return True

        # 纯关键词短查询（≤10字，不含疑问词，不含模糊词）→ 本身就是好的检索词
        # 注：含模糊词（"好不好/怎么样"）的短查询需要走 Level 2，不能在这里跳过
        if (
            len(query) <= 10
            and not any(w in query for w in self._QUESTION_WORDS)
            and not self._is_ambiguous(query)
        ):
            return True

        # 长查询 + 无指代词 + 无模糊词 → 用户自己说得够清楚
        if (
            len(query) > 15
            and not self._has_coref(query)
            and not self._is_ambiguous(query)
        ):
            return True

        return False

    def _has_coref(self, query: str) -> bool:
        """是否含有指代词（Level 1 触发条件）"""
        return any(w in query for w in self._COREF_WORDS)

    def _is_ambiguous(self, query: str) -> bool:
        """是否语义模糊（Level 2 触发条件：含模糊词 + 查询较短）"""
        return any(w in query for w in self._AMBIGUITY_WORDS) and len(query) < 20
