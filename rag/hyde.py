import logging
from utils.llm_client import LLMClient


class HyDE:
    """HyDE (Hypothetical Document Embeddings): 生成假设性文档"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化HyDE")
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        生成假设性文档
        
        HyDE的核心思想：生成一个假设性的答案文档，然后用这个文档去检索，
        而不是直接用问题检索。因为答案和知识库中的文档更相似。
        
        Args:
            query: 用户查询
            
        Returns:
            假设性文档
        """
        self.logger.info(f"生成假设性文档，查询: {query}")
        
        prompt = f"""请为以下问题生成一个详细的、专业的回答。即使你不确定答案，也请基于问题生成一个合理的、信息丰富的文档。

问题: {query}

请直接输出答案内容，不要添加"答案："等前缀："""
        
        try:
            hypothetical_doc = self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=300
            ).strip()
            self.logger.info(f"生成假设性文档，长度: {len(hypothetical_doc)}")
            self.logger.debug(f"假设性文档内容: {hypothetical_doc[:200]}...")
            
            return hypothetical_doc
        except Exception as e:
            self.logger.error(f"生成假设性文档失败: {e}")
            # 降级为返回原始查询
            return query
