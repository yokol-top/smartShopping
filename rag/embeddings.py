import logging
from typing import List, Optional
import numpy as np
from openai import OpenAI


class EmbeddingModel:
    """统一的嵌入模型接口，支持Ollama、OpenAI和本地模型"""
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = None,
        api_key: str = "EMPTY",
        base_url: str = None,
        logger: logging.Logger = None
    ):
        """
        初始化嵌入模型
        
        Args:
            provider: 提供商 ("ollama", "openai", 或 "local")
            model: 模型名称或本地路径
                - ollama: "nomic-embed-text" (默认)
                - openai: "text-embedding-3-small" (默认)
                - local: 本地模型路径 (如 "../rag_app/bge-small-zh-v1.5")
            api_key: API密钥（Ollama和local使用"EMPTY"）
            base_url: API基础URL
                - ollama: "http://localhost:11434/v1" (默认)
                - openai: "https://api.openai.com/v1" (默认)
                - local: 不使用
            logger: 日志记录器
        """
        self.provider = provider.lower()
        self.logger = logger or logging.getLogger(__name__)
        self.client = None
        self.local_model = None
        
        # 设置默认值
        if self.provider == "ollama":
            self.model = model or "nomic-embed-text"
            self.base_url = base_url or "http://localhost:11434/v1"
            # 初始化OpenAI客户端
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
        elif self.provider == "openai":
            self.model = model or "text-embedding-3-small"
            self.base_url = base_url or "https://api.openai.com/v1"
            # 初始化OpenAI客户端
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
        elif self.provider == "local":
            if not model:
                raise ValueError("本地提供商需要指定模型路径")
            self.model = model
            self.logger.info(f"从本地路径加载模型: {self.model}")
            # 加载本地 Sentence Transformers 模型
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(self.model)
            self.logger.info(f"本地模型加载成功，维度: {self.local_model.get_sentence_embedding_dimension()}")
        else:
            raise ValueError(f"不支持的提供商: {provider}。支持的提供商: ollama, openai, local")
        
        self.logger.info(f"初始化嵌入模型 - 提供商: {self.provider}, 模型: {self.model}")
        self.logger.info("嵌入模型初始化完成")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入向量
        
        Args:
            texts: 文档文本列表
            
        Returns:
            嵌入向量列表
        """
        self.logger.debug(f"生成 {len(texts)} 个文档的嵌入向量")
        
        embeddings = []
        
        if self.provider == "local":
            # 本地 Sentence Transformers 模型
            try:
                self.logger.info(f"使用本地模型生成 {len(texts)} 个嵌入向量")
                embeddings_array = self.local_model.encode(
                    texts,
                    show_progress_bar=len(texts) > 10,
                    convert_to_numpy=True
                )
                embeddings = embeddings_array.tolist()
                self.logger.info(f"完成 {len(texts)} 个文档的嵌入向量生成")
            except Exception as e:
                self.logger.error(f"本地模型生成嵌入向量失败: {e}")
                raise
        elif self.provider == "ollama":
            # Ollama使用单独的embeddings API
            import ollama as ollama_lib
            for i, text in enumerate(texts):
                try:
                    response = ollama_lib.embeddings(
                        model=self.model,
                        prompt=text
                    )
                    embeddings.append(response['embedding'])
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"已生成 {i + 1}/{len(texts)} 个嵌入向量")
                except Exception as e:
                    self.logger.error(f"生成嵌入向量失败 (文档 {i}): {e}")
                    embeddings.append([0.0] * 768)
        else:
            # OpenAI API批量处理
            try:
                # OpenAI支持批量嵌入
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    self.logger.debug(f"已生成 {min(i + batch_size, len(texts))}/{len(texts)} 个嵌入向量")
            except Exception as e:
                self.logger.error(f"批量生成嵌入向量失败: {e}")
                # 降级为逐个处理
                for i, text in enumerate(texts):
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=[text]
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as e2:
                        self.logger.error(f"生成嵌入向量失败 (文档 {i}): {e2}")
                        embeddings.append([0.0] * 768)
        
        self.logger.info(f"完成 {len(texts)} 个文档的嵌入向量生成")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入向量
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        try:
            self.logger.debug(f"生成查询的嵌入向量，文本长度: {len(text)}")
            
            if self.provider == "local":
                # 本地 Sentence Transformers 模型
                embedding = self.local_model.encode(
                    text,
                    convert_to_numpy=True
                )
                return embedding.tolist()
            elif self.provider == "ollama":
                import ollama as ollama_lib
                response = ollama_lib.embeddings(
                    model=self.model,
                    prompt=text
                )
                return response['embedding']
            else:
                # OpenAI API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"生成查询嵌入向量失败: {e}")
            raise
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# 保持向后兼容
OllamaEmbeddings = EmbeddingModel
