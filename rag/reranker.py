import logging
from typing import List, Dict, Any
from utils.llm_client import LLMClient


class Reranker:
    """重排序器：对检索结果进行重新排序"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化重排序器")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的文档列表
        """
        self.logger.info(f"重排序 {len(documents)} 个文档，返回top {top_k}")
        
        if not documents:
            return []
        
        if len(documents) <= top_k:
            self.logger.debug("文档数量少于top_k，无需重排序")
            return documents[:top_k]
        
        # 为每个文档计算相关性分数
        scored_documents = []
        
        for i, doc in enumerate(documents):
            try:
                doc_text = doc.get('document', '')
                score = self._compute_relevance_score(query, doc_text)
                
                doc_with_score = doc.copy()
                doc_with_score['rerank_score'] = score
                scored_documents.append(doc_with_score)
                
                self.logger.debug(f"文档 {i} 重排序分数: {score:.4f}")
            except Exception as e:
                self.logger.error(f"计算文档 {i} 相关性失败: {e}")
                doc_with_score = doc.copy()
                doc_with_score['rerank_score'] = 0.0
                scored_documents.append(doc_with_score)
        
        # 按重排序分数排序
        scored_documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        result = scored_documents[:top_k]
        self.logger.info(f"重排序完成，返回 {len(result)} 个文档")
        
        return result
    
    def _compute_relevance_score(self, query: str, document: str) -> float:
        """
        使用LLM计算相关性分数
        
        Args:
            query: 查询文本
            document: 文档文本
            
        Returns:
            相关性分数 (0-1)
        """
        # 简化prompt以提高速度
        prompt = f"""评估以下文档与问题的相关性。只输出一个0-10的分数，分数越高表示越相关。

问题: {query}

文档: {document[:500]}...

只输出数字分数（0-10）："""
        
        try:
            score_text = self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            ).strip()
            
            # 提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', score_text)
            if numbers:
                score = float(numbers[0])
                # 归一化到0-1
                score = min(max(score / 10.0, 0.0), 1.0)
                return score
            else:
                self.logger.warning(f"无法解析分数: {score_text}")
                return 0.5
        except Exception as e:
            self.logger.error(f"计算相关性分数失败: {e}")
            return 0.5
    
    def simple_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        简单重排序：基于关键词匹配
        
        这是一个更快的备选方案，不使用LLM
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的文档列表
        """
        self.logger.info(f"简单重排序 {len(documents)} 个文档")
        
        query_terms = set(query.lower().split())
        
        for doc in documents:
            doc_text = doc.get('document', '').lower()
            
            # 计算查询词在文档中的覆盖率
            matches = sum(1 for term in query_terms if term in doc_text)
            coverage = matches / len(query_terms) if query_terms else 0
            
            # 结合原始分数
            original_score = doc.get('score', 0.5)
            doc['rerank_score'] = 0.5 * original_score + 0.5 * coverage
        
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return documents[:top_k]
