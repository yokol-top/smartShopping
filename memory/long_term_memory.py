import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
# 包装 EmbeddingModel 为 ChromaDB 兼容的嵌入函数
from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.config import Settings

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

        # 创建或获取用户偏好集合（仅存储用户偏好/身份，增量更新）
        self.preferences_collection = self._get_or_recreate_collection(
            "user_preferences", "用户偏好和身份信息（增量更新）", embedding_fn
        )

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
            user_id: str = "",
            topics: List[str] = None,
            key_points: List[str] = None
    ):
        """添加对话总结（用户维度，支持跨会话语义检索）

        Args:
            session_id: 会话ID（仍保留用于溯源，但不作为检索维度）
            summary:    对话总结
            user_id:    用户ID（主检索维度）
            topics:     主题列表
            key_points: 关键点列表
        """
        try:
            now = datetime.now().isoformat()
            conv_id = f"{user_id or session_id}_{uuid.uuid4().hex[:8]}"

            self.conversations_collection.add(
                documents=[summary],
                metadatas=[{
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": now,
                    "topics": json.dumps(topics or [], ensure_ascii=False),
                    "key_points": json.dumps(key_points or [], ensure_ascii=False)
                }],
                ids=[conv_id]
            )

            self.logger.info(f"添加对话总结: user={user_id} session={session_id[:8]}...")
        except Exception as e:
            self.logger.error(f"添加对话总结失败: {e}")

    def get_user_summaries(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户的全部历史对话总结（跨会话，按时间倒序）

        Args:
            user_id: 用户ID
            limit:   最大返回条数

        Returns:
            总结列表
        """
        try:
            results = self.conversations_collection.get(
                where={"user_id": user_id},
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
            summaries.sort(key=lambda x: x['timestamp'], reverse=True)
            self.logger.debug(f"获取用户 {user_id} 的总结，数量: {len(summaries)}")
            return summaries[:limit]
        except Exception as e:
            self.logger.error(f"获取用户总结失败: {e}")
            return []

    def get_session_summaries(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """兼容旧接口：按会话ID查询总结（建议改用 get_user_summaries）"""
        try:
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
            summaries.sort(key=lambda x: x['timestamp'], reverse=True)
            return summaries[:limit]
        except Exception as e:
            self.logger.error(f"获取会话总结失败: {e}")
            return []

    def search_similar_conversations(
            self,
            query: str,
            user_id: Optional[str] = None,
            n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """基于向量相似度搜索用户的历史对话（跨会话）

        Args:
            query:    查询文本
            user_id:  用户ID（主过滤维度，跨会话检索）
            n_results: 返回结果数量

        Returns:
            相关对话列表
        """
        try:
            where_filter = {"user_id": user_id} if user_id else None

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

    def clear_conversation_summaries(self, user_id: str) -> int:
        """删除用户的全部对话摘要记录（跨会话）。

        Args:
            user_id: 用户 ID

        Returns:
            实际删除的条数
        """
        try:
            results = self.conversations_collection.get(
                where={"user_id": user_id}
            )
            count = len(results['ids'])
            if results['ids']:
                self.conversations_collection.delete(ids=results['ids'])
                self.logger.info(f"[LTM] 清除用户对话摘要: {user_id} ({count} 条)")
            return count
        except Exception as e:
            self.logger.error(f"[LTM] 清除对话摘要失败: {e}")
            return 0

    # ================================================================
    # 用户偏好（追加写入保留历史版本，查询时取最新值）
    # ================================================================
    #
    # 存储策略：append-only
    #   doc_id = "user:{user_id}:{pref_key}:{uuid4_hex8}"
    #   metadata.updated_at = ISO 时间戳
    #
    # 支持的操作：
    #   update_user_preferences   — 增量追加（字段级，只写本轮变化的字段）
    #   get_user_preferences      — 按 pref_key 分组，返回最新值
    #   get_preference_history    — 返回某个 key 的完整历史（时间倒序）
    #   clear_user_preference     — 硬删除某个 key 的所有历史记录
    #   clear_all_user_preferences — 硬删除用户的全部偏好（管理员操作）
    # ================================================================

    def update_user_preferences(self, user_id: str, prefs_dict: Dict[str, Any]) -> None:
        """增量追加用户偏好（字段级，只写本轮有变化的字段）。

        采用 append-only 策略保留历史版本，每次写入均生成唯一 doc_id。
        查询时自动取最新版本（见 get_user_preferences）。
        空字符串 / None 值自动跳过。

        Args:
            user_id:    用户 ID
            prefs_dict: 偏好字典，如 {'budget': '2万以内', 'interests': '电子产品'}
        """
        now = datetime.now().isoformat()
        written: List[str] = []
        for pref_key, pref_value in prefs_dict.items():
            if not pref_value:
                continue
            # 每条记录唯一 ID，保留历史
            doc_id = f"user:{user_id}:{pref_key}:{uuid.uuid4().hex[:8]}"
            value_str = str(pref_value)
            try:
                self.preferences_collection.add(
                    documents=[value_str],
                    metadatas=[{
                        "user_id": user_id,
                        "pref_key": pref_key,
                        "updated_at": now,
                    }],
                    ids=[doc_id],
                )
                written.append(f"{pref_key}={value_str[:40]}")
            except Exception as e:
                self.logger.error(f"[UserPref] 追加失败 {pref_key}: {e}")
        if written:
            self.logger.info(f"[UserPref] 追加偏好: {user_id} | {', '.join(written)}")

    def get_user_preferences(self, user_id: str) -> Dict[str, str]:
        """获取用户当前有效偏好（每个 key 取最新版本）。

        Returns:
            偏好键值对，如 {'budget': '2万以内', 'interests': '电子产品'}
        """
        try:
            results = self.preferences_collection.get(
                where={"user_id": user_id}
            )
            # 按 pref_key 分组，保留 updated_at 最大的那条
            latest: Dict[str, Dict[str, str]] = {}
            for i, _ in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                pref_key = metadata.get('pref_key', '')
                updated_at = metadata.get('updated_at', '')
                if not pref_key:
                    continue
                if pref_key not in latest or updated_at > latest[pref_key]['updated_at']:
                    latest[pref_key] = {
                        'value': results['documents'][i],
                        'updated_at': updated_at,
                    }
            return {k: v['value'] for k, v in latest.items()}
        except Exception as e:
            self.logger.error(f"[UserPref] 获取偏好失败: {e}")
            return {}

    def get_preference_history(
        self, user_id: str, pref_key: str
    ) -> List[Dict[str, str]]:
        """获取某个偏好 key 的完整历史版本（时间倒序，最新在前）。

        Args:
            user_id:  用户 ID
            pref_key: 偏好字段名，如 'budget'

        Returns:
            历史列表，每项 {'value': ..., 'updated_at': ...}
        """
        try:
            results = self.preferences_collection.get(
                where={"user_id": user_id, "pref_key": pref_key}
            )
            history = [
                {
                    'value': results['documents'][i],
                    'updated_at': results['metadatas'][i].get('updated_at', ''),
                }
                for i in range(len(results['ids']))
            ]
            history.sort(key=lambda x: x['updated_at'], reverse=True)
            return history
        except Exception as e:
            self.logger.error(f"[UserPref] 获取历史失败 {pref_key}: {e}")
            return []

    def clear_user_preference(self, user_id: str, pref_key: str) -> int:
        """删除某个偏好 key 的全部历史记录。

        Args:
            user_id:  用户 ID
            pref_key: 偏好字段名，如 'budget'

        Returns:
            实际删除的记录数
        """
        try:
            results = self.preferences_collection.get(
                where={"user_id": user_id, "pref_key": pref_key}
            )
            count = len(results['ids'])
            if results['ids']:
                self.preferences_collection.delete(ids=results['ids'])
                self.logger.info(f"[UserPref] 清除偏好: {user_id} | {pref_key} ({count} 条)")
            return count
        except Exception as e:
            self.logger.error(f"[UserPref] 清除偏好失败 {pref_key}: {e}")
            return 0

    def clear_all_user_preferences(self, user_id: str) -> int:
        """删除用户的全部偏好记录（管理员操作）。

        Args:
            user_id: 用户 ID

        Returns:
            实际删除的记录总数
        """
        try:
            results = self.preferences_collection.get(
                where={"user_id": user_id}
            )
            count = len(results['ids'])
            if results['ids']:
                self.preferences_collection.delete(ids=results['ids'])
                self.logger.info(f"[UserPref] 清空全部偏好: {user_id} ({count} 条)")
            return count
        except Exception as e:
            self.logger.error(f"[UserPref] 清空偏好失败: {e}")
            return 0

    def format_user_preferences(self, user_id: str) -> str:
        """将用户当前偏好格式化为可注入上下文的字符串。

        Returns:
            格式化字符串，如 "[用户偏好]\n- 预算: 2万以内\n- 兴趣: 电子产品"
            无偏好数据时返回空字符串
        """
        prefs = self.get_user_preferences(user_id)
        if not prefs:
            return ""
        _label_map = {
            'budget': '预算',
            'interests': '兴趣/偏好品类',
            'preferred_brands': '偏好品牌',
            'usage_scenario': '使用场景',
        }
        lines = ["[用户偏好]"]
        for key, value in prefs.items():
            label = _label_map.get(key, key)
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)
