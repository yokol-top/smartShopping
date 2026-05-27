import logging
from typing import List
from utils.llm_client import LLMClient


class QueryRewriter:
    """查询重写：根据上下文改进查询"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化查询重写器")
    
    def rewrite_query(self, query: str, context: str = "") -> str:
        """
        重写查询以更好地表达用户意图
        
        Args:
            query: 原始查询
            context: 对话上下文
            
        Returns:
            重写后的查询
        """
        self.logger.info(f"重写查询: {query}")
        
        prompt = f"""你是一个查询优化专家。请根据用户的原始问题和对话上下文，重写问题使其更加清晰、具体和准确。

对话上下文:
{context if context else "无"}

原始问题: {query}

请直接输出重写后的问题，不要添加任何解释："""
        
        try:
            rewritten = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3
            ).strip()
            self.logger.info(f"重写结果: {rewritten}")
            return rewritten
        except Exception as e:
            self.logger.error(f"查询重写失败: {e}")
            return query


class MultiQueryGenerator:
    """多查询生成：从不同角度生成多个查询"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化多查询生成器")
    
    def generate_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        生成多个相关查询
        
        Args:
            query: 原始查询
            num_queries: 生成的查询数量
            
        Returns:
            查询列表（包含原始查询）
        """
        self.logger.info(f"生成 {num_queries} 个相关查询")
        
        prompt = f"""你是一个搜索专家。请根据用户的问题，生成{num_queries}个不同角度但意思相近的搜索查询。
这些查询应该从不同的视角来表达相同的信息需求。

原始问题: {query}

请以列表形式输出{num_queries}个查询，每行一个，不要编号："""
        
        try:
            queries_text = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7
            ).strip()
            queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
            
            # 去除可能的编号
            queries = [q.lstrip('0123456789.-) ') for q in queries]
            
            # 确保包含原始查询
            if query not in queries:
                queries.insert(0, query)
            
            # 限制数量
            queries = queries[:num_queries]
            
            self.logger.info(f"生成了 {len(queries)} 个查询: {queries}")
            return queries
        except Exception as e:
            self.logger.error(f"多查询生成失败: {e}")
            return [query]
