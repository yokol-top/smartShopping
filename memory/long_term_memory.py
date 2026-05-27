import chromadb
from chromadb.config import Settings
# 包装 EmbeddingModel 为 ChromaDB 兼容的嵌入函数
from chromadb.api.types import EmbeddingFunction, Documents
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid

# 禁用ChromaDB遥测，避免网络连接超时
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_ANONYMOUS'] = 'False'


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input: Documents) -> list:
        return self.embedding_model.embed_documents(input)


class LongTermMemory:
    """长期记忆：使用Chroma向量库保存对话总结和重要信息"""

    def __init__(
            self,
            persist_directory: str,
            embedding_function=None,
            logger: logging.Logger = None
    ):
        """
        初始化长期记忆
        
        Args:
            persist_directory: Chroma持久化目录
            embedding_function: 嵌入函数（如果为None，使用Chroma默认）
            logger: 日志记录器
        """
        self.persist_directory = persist_directory
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_function = embedding_function

        os.makedirs(persist_directory, exist_ok=True)

        self._init_chroma()
        self.logger.info(f"初始化长期记忆向量库: {persist_directory}")

    def _get_or_recreate_collection(self, name: str, description: str, embedding_fn):
        """获取或重新创建集合，处理嵌入函数冲突"""
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(
                name=name,
                embedding_function=embedding_fn
            )
            self.logger.debug(f"集合 {name} 已存在，复用")
            return collection
        except ValueError as e:
            # 如果是嵌入函数冲突，删除旧集合并重新创建
            if "embedding function" in str(e).lower() or "conflict" in str(e).lower():
                self.logger.warning(f"检测到嵌入函数冲突，重新创建集合 {name}")
                try:
                    self.client.delete_collection(name=name)
                except:
                    pass
                return self.client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine", "description": description},
                    embedding_function=embedding_fn
                )
            raise
        except:
            # 集合不存在，创建新的
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "description": description},
                embedding_function=embedding_fn
            )

    def _init_chroma(self):
        """初始化Chroma集合"""
        # 初始化Chroma客户端（完全离线配置）
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # 准备嵌入函数包装器
        embedding_fn = None
        if self.embedding_function:
            # 包装 EmbeddingModel 为 ChromaDB 兼容的嵌入函数
            from chromadb.api.types import EmbeddingFunction, Documents
            
            class CustomEmbeddingFunction(EmbeddingFunction):
                def __init__(self, embedding_model):
                    self.embedding_model = embedding_model
                
                def __call__(self, input: Documents) -> list:
                    return self.embedding_model.embed_documents(input)
            
            embedding_fn = CustomEmbeddingFunction(self.embedding_function)

        # 创建或获取会话集合
        self.sessions_collection = self._get_or_recreate_collection("sessions", "对话会话记录", embedding_fn)

        # 创建或获取对话总结集合
        self.conversations_collection = self._get_or_recreate_collection("conversations", "对话片段总结", embedding_fn)

        # 创建或获取知识库集合
        self.knowledge_collection = self._get_or_recreate_collection("knowledge", "长期知识存储", embedding_fn)

        self.logger.debug("Chroma集合初始化完成")

    def create_session(self, session_id: str) -> bool:
        """
        创建新的会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功
        """
        try:
            now = datetime.now().isoformat()

            # 检查会话是否已存在
            existing = self.sessions_collection.get(ids=[session_id])
            if existing['ids']:
                self.logger.debug(f"会话已存在: {session_id}")
                return True

            # 添加新会话
            self.sessions_collection.add(
                documents=[f"会话 {session_id}"],
                metadatas=[{
                    "session_id": session_id,
                    "created_at": now,
                    "updated_at": now,
                    "summary": ""
                }],
                ids=[session_id]
            )

            self.logger.info(f"创建会话: {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"创建会话失败: {e}")
            return False

    def add_conversation_summary(
            self,
            session_id: str,
            summary: str,
            topics: List[str] = None,
            key_points: List[str] = None
    ):
        """
        添加对话总结（使用向量存储，支持语义检索）
        
        Args:
            session_id: 会话ID
            summary: 对话总结
            topics: 主题列表
            key_points: 关键点列表
        """
        try:
            now = datetime.now().isoformat()
            conv_id = f"{session_id}_{uuid.uuid4().hex[:8]}"

            # 添加对话总结到向量库
            self.conversations_collection.add(
                documents=[summary],  # 总结作为文档，用于向量搜索
                metadatas=[{
                    "session_id": session_id,
                    "timestamp": now,
                    "topics": json.dumps(topics or [], ensure_ascii=False),
                    "key_points": json.dumps(key_points or [], ensure_ascii=False)
                }],
                ids=[conv_id]
            )

            # 更新会话的最后更新时间
            try:
                session_data = self.sessions_collection.get(ids=[session_id])
                if session_data['ids']:
                    self.sessions_collection.update(
                        ids=[session_id],
                        metadatas=[{
                            **session_data['metadatas'][0],
                            "updated_at": now
                        }]
                    )
            except Exception as e:
                self.logger.warning(f"更新会话时间失败: {e}")

            self.logger.info(f"添加对话总结到会话 {session_id}, 主题: {topics}")
        except Exception as e:
            self.logger.error(f"添加对话总结失败: {e}")

    def get_session_summaries(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取会话的历史总结
        
        Args:
            session_id: 会话ID
            limit: 获取数量限制
            
        Returns:
            总结列表（按时间倒序）
        """
        try:
            # 查询该会话的所有对话总结
            results = self.conversations_collection.get(
                where={"session_id": session_id},
                limit=limit
            )

            summaries = []
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                summaries.append({
                    "id": doc_id,
                    "timestamp": metadata.get("timestamp", ""),
                    "summary": results['documents'][i],
                    "topics": json.loads(metadata.get("topics", "[]")),
                    "key_points": json.loads(metadata.get("key_points", "[]"))
                })

            # 按时间倒序排序
            summaries.sort(key=lambda x: x['timestamp'], reverse=True)

            self.logger.debug(f"获取会话 {session_id} 的总结，数量: {len(summaries)}")
            return summaries[:limit]
        except Exception as e:
            self.logger.error(f"获取会话总结失败: {e}")
            return []

    def search_similar_conversations(
            self,
            query: str,
            session_id: Optional[str] = None,
            n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        基于向量相似度搜索相关的历史对话
        
        Args:
            query: 查询文本
            session_id: 可选，限定会话ID
            n_results: 返回结果数量
            
        Returns:
            相关对话列表
        """
        try:
            where_filter = {"session_id": session_id} if session_id else None

            results = self.conversations_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )

            conversations = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    conversations.append({
                        "id": doc_id,
                        "summary": results['documents'][0][i],
                        "timestamp": metadata.get("timestamp", ""),
                        "session_id": metadata.get("session_id", ""),
                        "topics": json.loads(metadata.get("topics", "[]")),
                        "key_points": json.loads(metadata.get("key_points", "[]")),
                        "distance": results['distances'][0][i] if results.get('distances') else None
                    })

            self.logger.debug(f"搜索到 {len(conversations)} 条相关对话")
            return conversations
        except Exception as e:
            self.logger.error(f"搜索相关对话失败: {e}")
            return []

    def store_knowledge(self, key: str, value: str, category: str = "general"):
        """
        存储知识到长期记忆（使用向量存储）
        
        Args:
            key: 知识键
            value: 知识值
            category: 分类
        """
        try:
            now = datetime.now().isoformat()

            # 检查是否已存在
            existing = self.knowledge_collection.get(ids=[key])

            if existing['ids']:
                # 更新现有知识
                old_metadata = existing['metadatas'][0]
                self.knowledge_collection.update(
                    ids=[key],
                    documents=[value],
                    metadatas=[{
                        "key": key,
                        "category": category,
                        "created_at": old_metadata.get("created_at", now),
                        "updated_at": now
                    }]
                )
                self.logger.info(f"更新知识: {key} (分类: {category})")
            else:
                # 添加新知识
                self.knowledge_collection.add(
                    documents=[value],
                    metadatas=[{
                        "key": key,
                        "category": category,
                        "created_at": now,
                        "updated_at": now
                    }],
                    ids=[key]
                )
                self.logger.info(f"存储知识: {key} (分类: {category})")
        except Exception as e:
            self.logger.error(f"存储知识失败: {e}")

    def retrieve_knowledge(self, key: str) -> Optional[str]:
        """
        检索知识
        
        Args:
            key: 知识键
            
        Returns:
            知识值，如果不存在返回None
        """
        try:
            results = self.knowledge_collection.get(ids=[key])

            if results['ids']:
                self.logger.debug(f"检索到知识: {key}")
                return results['documents'][0]
            else:
                self.logger.debug(f"知识不存在: {key}")
                return None
        except Exception as e:
            self.logger.error(f"检索知识失败: {e}")
            return None

    def search_knowledge(
            self,
            query: Optional[str] = None,
            category: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索知识（支持向量相似度搜索）
        
        Args:
            query: 查询文本（用于向量搜索），如果为None则返回所有知识
            category: 分类过滤
            limit: 数量限制
            
        Returns:
            知识列表
        """
        try:
            where_filter = {"category": category} if category else None

            if query:
                # 向量相似度搜索
                results = self.knowledge_collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_filter
                )

                knowledge_list = []
                if results['ids'] and results['ids'][0]:
                    for i, doc_id in enumerate(results['ids'][0]):
                        metadata = results['metadatas'][0][i]
                        knowledge_list.append({
                            "key": metadata.get("key", doc_id),
                            "value": results['documents'][0][i],
                            "category": metadata.get("category", ""),
                            "created_at": metadata.get("created_at", ""),
                            "updated_at": metadata.get("updated_at", ""),
                            "distance": results['distances'][0][i] if results.get('distances') else None
                        })
            else:
                # 获取所有知识
                results = self.knowledge_collection.get(
                    where=where_filter,
                    limit=limit
                )

                knowledge_list = []
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    knowledge_list.append({
                        "key": metadata.get("key", doc_id),
                        "value": results['documents'][i],
                        "category": metadata.get("category", ""),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at", "")
                    })

                # 按更新时间倒序排序
                knowledge_list.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

            self.logger.debug(f"搜索知识，分类: {category}, 数量: {len(knowledge_list)}")
            return knowledge_list[:limit]
        except Exception as e:
            self.logger.error(f"搜索知识失败: {e}")
            return []
