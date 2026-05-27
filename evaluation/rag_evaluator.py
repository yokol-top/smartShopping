"""
RAG效果评估模块 (RAG Evaluator)

多维度评估RAG检索与生成质量：
1. 检索相关性 (Retrieval Relevance): 检索到的文档与查询的相关程度
2. 答案忠实度 (Faithfulness): 答案是否基于检索文档生成，是否有幻觉
3. 答案相关性 (Answer Relevance): 答案与用户问题的匹配程度
4. 上下文利用率 (Context Utilization): 检索文档的信息被利用的程度
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RAGEvalResult:
    """RAG评估结果"""
    retrieval_relevance: float = 0.0    # 检索相关性 [0, 1]
    faithfulness: float = 0.0           # 答案忠实度 [0, 1]
    answer_relevance: float = 0.0       # 答案相关性 [0, 1]
    context_utilization: float = 0.0    # 上下文利用率 [0, 1]
    overall_score: float = 0.0          # 综合得分 [0, 1]
    details: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_relevance": round(self.retrieval_relevance, 3),
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_utilization": round(self.context_utilization, 3),
            "overall_score": round(self.overall_score, 3),
            "evaluation_time_ms": round(self.evaluation_time_ms, 1),
            "details": self.details,
        }


class RAGEvaluator:
    """
    RAG效果评估器

    使用LLM-as-Judge的方式对RAG质量进行多维度评估。
    """

    def __init__(self, config: Dict[str, Any], llm_client=None,
                 logger: logging.Logger = None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

        eval_config = config.get('evaluation', {}).get('rag', {})
        self.enabled = eval_config.get('enabled', True)

        # 各维度权重
        self.weights = eval_config.get('weights', {
            'retrieval_relevance': 0.25,
            'faithfulness': 0.30,
            'answer_relevance': 0.30,
            'context_utilization': 0.15,
        })

        self.logger.info("RAGEvaluator 初始化完成")

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[str],
        context: str = "",
    ) -> RAGEvalResult:
        """
        评估RAG效果

        Args:
            query: 用户查询
            answer: 生成的答案
            retrieved_docs: 检索到的文档列表
            context: 对话上下文

        Returns:
            RAGEvalResult
        """
        if not self.enabled or not self.llm_client:
            return RAGEvalResult(overall_score=-1.0, details={"reason": "评估未启用"})

        start_time = time.time()
        self.logger.info(f"[RAG评估] 开始评估 | 查询: {query[:50]}...")

        docs_text = "\n---\n".join(
            f"[文档{i+1}] {doc[:500]}" for i, doc in enumerate(retrieved_docs)
        ) if retrieved_docs else "（无检索结果）"

        prompt = f"""你是一个RAG系统质量评估专家。请对以下RAG问答结果进行多维度评估。

**用户查询**: {query}

**检索到的文档**:
{docs_text}

**生成的答案**: {answer[:1000]}

请从以下4个维度评估（每个维度0-1分）:

1. **retrieval_relevance** (检索相关性): 检索到的文档与用户查询的相关程度。
   - 1.0: 所有文档高度相关
   - 0.5: 部分相关
   - 0.0: 完全不相关或无检索结果

2. **faithfulness** (答案忠实度): 答案是否基于检索到的文档内容，是否存在幻觉（编造信息）。
   - 1.0: 答案完全基于文档，无幻觉
   - 0.5: 大部分基于文档，少量推断
   - 0.0: 大量编造信息

3. **answer_relevance** (答案相关性): 答案是否直接回答了用户的问题。
   - 1.0: 精准回答
   - 0.5: 部分回答
   - 0.0: 完全跑题

4. **context_utilization** (上下文利用率): 检索文档中的关键信息被利用的程度。
   - 1.0: 充分利用了所有相关信息
   - 0.5: 仅利用了部分信息
   - 0.0: 未利用检索信息

返回JSON格式:
{{"retrieval_relevance": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0, "context_utilization": 0.0, "brief_analysis": "简要分析"}}

只返回JSON。"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.1)
            response = self._clean_json(response)
            scores = json.loads(response)

            rr = float(scores.get('retrieval_relevance', 0.5))
            fa = float(scores.get('faithfulness', 0.5))
            ar = float(scores.get('answer_relevance', 0.5))
            cu = float(scores.get('context_utilization', 0.5))

            # 加权综合得分
            overall = (
                rr * self.weights.get('retrieval_relevance', 0.25) +
                fa * self.weights.get('faithfulness', 0.30) +
                ar * self.weights.get('answer_relevance', 0.30) +
                cu * self.weights.get('context_utilization', 0.15)
            )

            elapsed = (time.time() - start_time) * 1000
            result = RAGEvalResult(
                retrieval_relevance=rr,
                faithfulness=fa,
                answer_relevance=ar,
                context_utilization=cu,
                overall_score=overall,
                evaluation_time_ms=elapsed,
                details={"brief_analysis": scores.get('brief_analysis', '')},
            )

            self.logger.info(
                f"[RAG评估] 完成 | 综合: {overall:.2f} | 相关性: {rr:.2f} | "
                f"忠实度: {fa:.2f} | 答案: {ar:.2f} | 利用率: {cu:.2f}"
            )
            return result

        except Exception as e:
            self.logger.error(f"[RAG评估] 评估失败: {e}")
            return RAGEvalResult(
                overall_score=-1.0,
                evaluation_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)},
            )

    @staticmethod
    def _clean_json(response: str) -> str:
        response = response.strip()
        if response.startswith('```'):
            response = response.split('```')[1]
            if response.startswith('json'):
                response = response[4:]
        return response.strip()
