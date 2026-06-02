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


class CoRefResolver:
    """Level 1 轻量指代消解：只解决"这款/那个/它"等指代词，不做语义改写。

    设计原则：
    - Prompt 极短，只取最近 400 字上下文，减少 token 消耗
    - temperature=0.1，输出稳定
    - 失败时静默降级，返回原始查询
    """

    def __init__(self, llm_client: LLMClient, logger: logging.Logger = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("初始化 CoRefResolver (Level 1 指代消解)")

    def resolve(self, query: str, context: str) -> str:
        """将查询中的指代词替换为上下文中的具体实体。

        Args:
            query:   含指代词的用户查询（如"哪款价格最低？"）
            context: 近期对话历史（短期记忆摘要）

        Returns:
            消解后的独立完整问题；若无需修改则原样返回。
        """
        # 只取最近上下文，避免 prompt 过长
        ctx_snippet = context[-400:] if len(context) > 400 else context

        prompt = f"""你是一个指代消解工具。

对话历史（最近）：
{ctx_snippet}

用户提问：{query}

任务：如果提问中含有"这/那/它/上面提到的/哪款"等指代词，请将其替换为对话历史中的具体实体，\
输出一个完整独立的问题。如果没有指代词，直接原样输出问题。

只输出处理后的问题，不要任何解释。"""

        try:
            resolved = self.llm_client.generate(
                prompt=prompt, temperature=0.1
            ).strip()
            # 过滤掉明显的无效输出
            if not resolved or len(resolved) > len(query) * 5:
                return query
            return resolved
        except Exception as e:
            self.logger.warning(f"[CoRefResolver] 失败，使用原始查询: {e}")
            return query
