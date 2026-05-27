import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
import os


class VectorStore:
    """Chroma向量数据库封装"""
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "knowledge_base",
        logger: logging.Logger = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logger or logging.getLogger(__name__)
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.logger.info(f"初始化Chroma向量数据库: {persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger.info(f"加载/创建集合: {collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            ids: 文档ID列表
            embeddings: 嵌入向量列表（可选，如果不提供会自动生成）
        """
        try:
            if embeddings:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            self.logger.info(f"添加 {len(documents)} 个文档到向量数据库")
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise
    
    def query(
        self,
        query_texts: List[str] = None,
        query_embeddings: List[List[float]] = None,
        n_results: int = 5,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        查询向量数据库
        
        Args:
            query_texts: 查询文本列表
            query_embeddings: 查询嵌入向量列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            查询结果
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            self.logger.debug(f"查询向量数据库，返回 {n_results} 个结果")
            return results
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            raise
    
    def delete(self, ids: List[str] = None, where: Dict[str, Any] = None):
        """
        删除文档
        
        Args:
            ids: 要删除的文档ID列表
            where: 元数据过滤条件
        """
        try:
            self.collection.delete(ids=ids, where=where)
            self.logger.info(f"删除文档: {ids}")
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            raise
    
    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        try:
            count = self.collection.count()
            self.logger.debug(f"集合文档数量: {count}")
            return count
        except Exception as e:
            self.logger.error(f"获取文档数量失败: {e}")
            return 0
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """查看集合中的样本数据"""
        try:
            return self.collection.peek(limit=limit)
        except Exception as e:
            self.logger.error(f"查看样本数据失败: {e}")
            return {}
