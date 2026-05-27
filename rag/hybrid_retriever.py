import logging
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """混合检索器：结合向量检索和BM25"""
    
    def __init__(
        self,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        logger: logging.Logger = None
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.logger = logger or logging.getLogger(__name__)
        
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self.metadatas = []
        
        self.logger.info(
            f"初始化混合检索器 - 向量权重: {vector_weight}, BM25权重: {bm25_weight}"
        )
    
    def build_bm25_index(
        self,
        documents: List[str],
        document_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        构建BM25索引
        
        Args:
            documents: 文档列表
            document_ids: 文档ID列表
            metadatas: 元数据列表
        """
        self.logger.info(f"构建BM25索引，文档数量: {len(documents)}")
        
        self.documents = documents
        self.document_ids = document_ids
        self.metadatas = metadatas
        
        # 简单分词（对中文按字符分，对英文按空格分）
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        self.logger.info("BM25索引构建完成")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 对中文和英文混合文本进行分词
        tokens = []
        current_word = ""
        
        for char in text:
            if char.isspace():
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
            elif char.isalpha():
                current_word += char
            else:
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                if not char.isspace():
                    tokens.append(char)
        
        if current_word:
            tokens.append(current_word)
        
        return tokens
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (文档ID, 分数, 元数据) 的列表
        """
        if self.bm25_index is None:
            self.logger.warning("BM25索引未构建")
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self.document_ids[idx],
                    float(scores[idx]),
                    self.metadatas[idx]
                ))
        
        self.logger.debug(f"BM25检索返回 {len(results)} 个结果")
        return results
    
    def hybrid_search(
        self,
        vector_results: Dict[str, Any],
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        混合检索：结合向量检索和BM25结果
        
        Args:
            vector_results: 向量检索结果
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            混合检索结果
        """
        self.logger.info(f"执行混合检索，top_k: {top_k}")
        
        # 获取BM25结果
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        
        # 归一化向量检索分数
        vector_ids = vector_results['ids'][0] if vector_results['ids'] else []
        vector_distances = vector_results['distances'][0] if vector_results['distances'] else []
        vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
        vector_metas = vector_results['metadatas'][0] if vector_results['metadatas'] else []
        
        # 将距离转换为相似度分数（距离越小，相似度越高）
        if vector_distances:
            max_distance = max(vector_distances) if vector_distances else 1.0
            vector_scores = {
                vector_ids[i]: 1 - (vector_distances[i] / max_distance) if max_distance > 0 else 1.0
                for i in range(len(vector_ids))
            }
        else:
            vector_scores = {}
        
        # 归一化BM25分数
        if bm25_results:
            max_bm25_score = max(score for _, score, _ in bm25_results)
            bm25_scores = {
                doc_id: score / max_bm25_score if max_bm25_score > 0 else 0
                for doc_id, score, _ in bm25_results
            }
        else:
            bm25_scores = {}
        
        # 合并分数
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            vector_score = vector_scores.get(doc_id, 0)
            bm25_score = bm25_scores.get(doc_id, 0)
            
            combined_score = (
                self.vector_weight * vector_score +
                self.bm25_weight * bm25_score
            )
            combined_scores[doc_id] = combined_score
        
        # 排序并获取top_k
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )[:top_k]
        
        # 构建结果
        results = []
        for doc_id in sorted_ids:
            # 从向量检索结果或BM25结果中获取文档内容和元数据
            if doc_id in vector_ids:
                idx = vector_ids.index(doc_id)
                doc = vector_docs[idx]
                metadata = vector_metas[idx]
            else:
                # 从BM25结果中查找
                for bm25_id, _, bm25_meta in bm25_results:
                    if bm25_id == doc_id:
                        doc_idx = self.document_ids.index(doc_id)
                        doc = self.documents[doc_idx]
                        metadata = bm25_meta
                        break
            
            results.append({
                "id": doc_id,
                "document": doc,
                "metadata": metadata,
                "score": combined_scores[doc_id],
                "vector_score": vector_scores.get(doc_id, 0),
                "bm25_score": bm25_scores.get(doc_id, 0)
            })
        
        self.logger.info(f"混合检索完成，返回 {len(results)} 个结果")
        return results
