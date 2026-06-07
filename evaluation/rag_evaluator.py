"""
RAG 质量评估器 (RAG Evaluator)

基于 RAGAS 理论实现的 RAG 链路质量评估，四个核心维度：
1. 检索相关性 (Retrieval Relevance): 检索到的文档是否与问题相关
2. 忠实性 (Faithfulness): 回答是否忠实于检索到的文档，不编造信息
3. 回答相关性 (Answer Relevance): 回答是否切题
4. 上下文利用率 (Context Utilization): 检索到的文档被利用了多少

每个维度使用 LLM-as-Judge 进行评估，返回 0-1 的分数。
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RAGEvalResult:
    """RAG 评估结果"""
    query: str                              # 用户查询
    retrieval_relevance: float = 0.0        # 检索相关性 (0-1)
    faithfulness: float = 0.0               # 忠实性 (0-1)
    answer_relevance: float = 0.0           # 回答相关性 (0-1)
    context_utilization: float = 0.0        # 上下文利用率 (0-1)
    overall_score: float = 0.0             # 加权总分 (0-1)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_relevance": round(self.retrieval_relevance, 3),
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_utilization": round(self.context_utilization, 3),
            "overall_score": round(self.overall_score, 3),
            "details": self.details,
        }

    def summary(self) -> str:
        return (
            f"📊 RAG评估 | 查询: {self.query[:30]}...\n"
            f"  检索相关性: {self.retrieval_relevance:.2f} | "
            f"忠实性: {self.faithfulness:.2f} | "
            f"回答相关性: {self.answer_relevance:.2f} | "
            f"上下文利用率: {self.context_utilization:.2f}\n"
            f"  综合得分: {self.overall_score:.2f}"
        )


class RAGEvaluator:
    """
    RAG 质量评估器

    使用 LLM-as-Judge 方法评估 RAG 链路的四个维度。
    可独立使用，也可集成到 AgentEvaluator 中。
    """

    def __init__(self, config: Dict[str, Any], llm_client=None, logger: logging.Logger = None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

        eval_config = config.get('evaluation', {}).get('rag', {})
        self.enabled = eval_config.get('enabled', True)

        # 各维度权重
        weights = eval_config.get('weights', {})
        self.weights = {
            'retrieval_relevance': weights.get('retrieval_relevance', 0.25),
            'faithfulness': weights.get('faithfulness', 0.30),
            'answer_relevance': weights.get('answer_relevance', 0.30),
            'context_utilization': weights.get('context_utilization', 0.15),
        }

        # 历史评估结果
        self._history: List[RAGEvalResult] = []

        self.logger.info("RAGEvaluator 初始化完成")

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[str],
        reference_answer: str = "",
    ) -> RAGEvalResult:
        """
        评估一次 RAG 调用的质量

        Args:
            query: 用户原始查询
            answer: Agent 生成的回答
            retrieved_docs: 检索到的文档列表
            reference_answer: 参考答案（可选，用于更精确的评估）

        Returns:
            RAGEvalResult
        """
        if not self.enabled:
            return RAGEvalResult(query=query, overall_score=-1.0)

        if not self.llm_client:
            self.logger.warning("LLM 客户端未配置，无法进行 RAG 评估")
            return RAGEvalResult(query=query, overall_score=-1.0)

        context_text = "\n---\n".join(retrieved_docs) if retrieved_docs else "(无检索结果)"

        # 并行评估四个维度
        retrieval_relevance = self._eval_retrieval_relevance(query, retrieved_docs)
        faithfulness = self._eval_faithfulness(query, answer, context_text)
        answer_relevance = self._eval_answer_relevance(query, answer)
        context_utilization = self._eval_context_utilization(answer, context_text)

        # 加权计算总分
        overall = (
            self.weights['retrieval_relevance'] * retrieval_relevance +
            self.weights['faithfulness'] * faithfulness +
            self.weights['answer_relevance'] * answer_relevance +
            self.weights['context_utilization'] * context_utilization
        )

        result = RAGEvalResult(
            query=query,
            retrieval_relevance=retrieval_relevance,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_utilization=context_utilization,
            overall_score=overall,
            details={
                "num_docs_retrieved": len(retrieved_docs),
                "answer_length": len(answer),
                "weights": self.weights,
            },
        )

        self._history.append(result)
        self.logger.info(f"[RAG评估] {result.summary()}")
        return result

    # ================================================================
    # 维度 1: 检索相关性
    # ================================================================
    def _eval_retrieval_relevance(self, query: str, docs: List[str]) -> float:
        """评估检索到的文档是否与查询相关"""
        if not docs:
            return 0.0

        docs_text = "\n---\n".join(doc[:300] for doc in docs[:5])  # 截取前5篇，每篇300字

        prompt = f"""你是一个信息检索质量评审员。请评估以下检索结果与用户查询的相关性。

用户查询: {query}

检索到的文档:
{docs_text}

请从0到10打分，10分表示检索结果与查询高度相关，0分表示完全不相关。

评分标准:
- 9-10: 检索文档直接回答了用户的问题
- 7-8: 检索文档包含与问题相关的重要信息
- 5-6: 检索文档部分相关，但缺少关键信息
- 3-4: 检索文档与问题关联度很低
- 0-2: 检索文档与问题完全无关

只返回一个数字(0-10)，不要其他说明。"""

        return self._get_score(prompt)

    # ================================================================
    # 维度 2: 忠实性
    # ================================================================
    def _eval_faithfulness(self, query: str, answer: str, context: str) -> float:
        """评估回答是否忠实于检索到的文档"""
        prompt = f"""你是一个事实核查专家。请评估以下回答是否忠实于提供的参考文档。

用户查询: {query}

参考文档:
{context[:1500]}

回答:
{answer[:800]}

请从0到10打分，评估回答的忠实性:
- 9-10: 回答完全基于参考文档，没有任何编造或臆测
- 7-8: 回答基本忠实，有少量合理推断但不影响准确性
- 5-6: 回答部分忠实，包含一些文档中未提及的信息
- 3-4: 回答包含较多编造内容
- 0-2: 回答严重失实，与文档内容矛盾或完全编造

只返回一个数字(0-10)，不要其他说明。"""

        return self._get_score(prompt)

    # ================================================================
    # 维度 3: 回答相关性
    # ================================================================
    def _eval_answer_relevance(self, query: str, answer: str) -> float:
        """评估回答是否与问题相关"""
        prompt = f"""你是一个回答质量评审员。请评估以下回答是否切题地回答了用户的问题。

用户查询: {query}

回答:
{answer[:800]}

请从0到10打分，评估回答的相关性:
- 9-10: 回答完整、直接地回答了用户的问题
- 7-8: 回答基本切题，覆盖了核心内容
- 5-6: 回答部分相关，但有重要遗漏
- 3-4: 回答偏题，只有少量内容与问题相关
- 0-2: 回答完全离题

只返回一个数字(0-10)，不要其他说明。"""

        return self._get_score(prompt)

    # ================================================================
    # 维度 4: 上下文利用率
    # ================================================================
    def _eval_context_utilization(self, answer: str, context: str) -> float:
        """评估回答对检索文档的利用程度"""
        if not context or context == "(无检索结果)":
            return 0.0

        prompt = f"""你是一个信息利用效率评审员。请评估回答对参考文档的利用程度。

参考文档:
{context[:1500]}

回答:
{answer[:800]}

请从0到10打分，评估上下文利用率:
- 9-10: 回答充分利用了文档中的关键信息
- 7-8: 回答利用了大部分相关信息
- 5-6: 回答只利用了部分信息，遗漏了一些有用内容
- 3-4: 回答很少利用文档内容
- 0-2: 回答几乎没有利用文档内容

只返回一个数字(0-10)，不要其他说明。"""

        return self._get_score(prompt)

    # ================================================================
    # 辅助方法
    # ================================================================
    def _get_score(self, prompt: str) -> float:
        """调用 LLM 获取评分并归一化到 0-1"""
        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.1, max_tokens=10)
            # 提取数字
            score_text = response.strip()
            # 尝试提取第一个数字
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', score_text)
            if match:
                score = float(match.group(1))
                return min(max(score / 10.0, 0.0), 1.0)  # 归一化到 0-1
            return 0.5  # 解析失败时返回中间值
        except Exception as e:
            self.logger.warning(f"RAG 评分获取失败: {e}")
            return 0.5

    def get_average_scores(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """获取历史平均分数"""
        records = self._history[-last_n:] if last_n else self._history
        if not records:
            return {
                "retrieval_relevance": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_utilization": 0.0,
                "overall_score": 0.0,
                "count": 0,
            }

        n = len(records)
        return {
            "retrieval_relevance": round(sum(r.retrieval_relevance for r in records) / n, 3),
            "faithfulness": round(sum(r.faithfulness for r in records) / n, 3),
            "answer_relevance": round(sum(r.answer_relevance for r in records) / n, 3),
            "context_utilization": round(sum(r.context_utilization for r in records) / n, 3),
            "overall_score": round(sum(r.overall_score for r in records) / n, 3),
            "count": n,
        }

    def print_summary(self, last_n: Optional[int] = None):
        """打印评估摘要"""
        avg = self.get_average_scores(last_n)
        print(
            f"\n📊 RAG 质量评估摘要 (最近 {avg['count']} 次)\n"
            f"  检索相关性: {avg['retrieval_relevance']:.2f}\n"
            f"  忠实性:     {avg['faithfulness']:.2f}\n"
            f"  回答相关性: {avg['answer_relevance']:.2f}\n"
            f"  上下文利用: {avg['context_utilization']:.2f}\n"
            f"  综合得分:   {avg['overall_score']:.2f}\n"
        )
