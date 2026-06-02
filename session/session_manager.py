"""
会话管理模块

功能：
1. 为每个用户维护独立的会话列表
2. 每个会话保存完整对话历史（用户消息 + Agent 响应）
3. 支持创建、恢复、列表、删除会话
4. 基于 SQLite 持久化，轻量无外部依赖
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from utils.db_pool import ConnectionPool


@dataclass
class SessionInfo:
    """会话基本信息"""
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

    def to_display(self) -> str:
        """格式化为展示字符串"""
        ts = self.updated_at[:16] if len(self.updated_at) > 16 else self.updated_at
        return f"[{ts}] {self.title}  ({self.message_count} 条消息)"


@dataclass
class ConversationMessage:
    """会话中的单条消息"""
    message_id: str
    session_id: str
    role: str           # user / assistant / system
    content: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    会话管理器

    - 每个用户拥有独立的会话列表
    - 会话间上下文完全隔离
    - 支持恢复历史会话，加载完整对话记录
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        session_config = (config or {}).get('session', {})

        db_dir = session_config.get('data_dir', './data/sessions')
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, 'sessions.db')

        # 连接池（默认最大 5 个连接）
        pool_size = session_config.get('pool_size', 5)
        self._pool = ConnectionPool(self.db_path, max_size=pool_size)

        # 每用户最大会话数
        self.max_sessions_per_user = session_config.get('max_sessions_per_user', 50)

        self.logger.info(f"SessionManager 初始化 | DB: {self.db_path} | 连接池: {pool_size}")

    # ================================================================
    # 会话 CRUD
    # ================================================================

    def create_session(self, user_id: str, title: str = None) -> SessionInfo:
        """
        创建新会话

        Args:
            user_id: 用户ID
            title: 会话标题（默认"新会话"，首次用户发言后自动更新）

        Returns:
            SessionInfo
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        title = title or '新会话'

        with self._pool.get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, title, now, now)
            )
            conn.commit()

        self.logger.info(f"创建会话 {session_id[:8]}... | 用户: {user_id}")

        # 清理超限的旧会话
        self._cleanup_old_sessions(user_id)

        return SessionInfo(
            session_id=session_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_sessions(self, user_id: str, limit: int = 20) -> List[SessionInfo]:
        """
        列出用户的历史会话（按更新时间倒序）

        Args:
            user_id: 用户ID
            limit: 最大返回数

        Returns:
            SessionInfo 列表
        """
        with self._pool.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.user_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.message_id) as msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE s.user_id = ?
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (user_id, limit)
            ).fetchall()

        return [
            SessionInfo(
                session_id=r[0], user_id=r[1], title=r[2],
                created_at=r[3], updated_at=r[4], message_count=r[5],
            )
            for r in rows
        ]

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """获取单个会话信息"""
        with self._pool.get_conn() as conn:
            row = conn.execute(
                """
                SELECT s.session_id, s.user_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.message_id) as msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE s.session_id = ?
                GROUP BY s.session_id
                """,
                (session_id,)
            ).fetchone()

        if not row:
            return None

        return SessionInfo(
            session_id=row[0], user_id=row[1], title=row[2],
            created_at=row[3], updated_at=row[4], message_count=row[5],
        )

    def delete_session(self, session_id: str):
        """删除会话及其所有关联数据"""
        with self._pool.get_conn() as conn:
            # 先删除子表，再删主表
            conn.execute("DELETE FROM messages    WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM task_states WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions    WHERE session_id = ?", (session_id,))
            conn.commit()
        self.logger.info(f"删除会话 {session_id[:8]}...")

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        批量删除指定的多个会话及其所有消息

        Args:
            session_ids: 要删除的会话ID列表

        Returns:
            实际删除的会话数
        """
        if not session_ids:
            return 0

        placeholders = ','.join('?' * len(session_ids))
        with self._pool.get_conn() as conn:
            # 先删除所有子表记录，再删除主表，满足外键约束
            conn.execute(f"DELETE FROM messages    WHERE session_id IN ({placeholders})", session_ids)
            conn.execute(f"DELETE FROM task_states WHERE session_id IN ({placeholders})", session_ids)
            cursor = conn.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", session_ids)
            deleted = cursor.rowcount
            conn.commit()

        self.logger.info(f"批量删除 {deleted} 个会话")
        return deleted

    def delete_all_sessions(self, user_id: str) -> int:
        """
        删除用户的所有会话及其消息

        Args:
            user_id: 用户ID

        Returns:
            删除的会话数
        """
        with self._pool.get_conn() as conn:
            # 先获取该用户所有会话ID
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ?", (user_id,)
            ).fetchall()
            if not rows:
                return 0

            session_ids = [r[0] for r in rows]
            placeholders = ','.join('?' * len(session_ids))

            # 先删除所有子表记录，再删除主表，满足外键约束
            conn.execute(f"DELETE FROM messages    WHERE session_id IN ({placeholders})", session_ids)
            conn.execute(f"DELETE FROM task_states WHERE session_id IN ({placeholders})", session_ids)
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()

        self.logger.info(f"删除用户 {user_id} 的全部 {len(session_ids)} 个会话")
        return len(session_ids)

    def update_session_title(self, session_id: str, title: str):
        """更新会话标题"""
        with self._pool.get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), session_id)
            )
            conn.commit()

    # ================================================================
    # 消息 CRUD
    # ================================================================

    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Dict[str, Any] = None) -> str:
        """
        向会话添加一条消息

        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 附加元数据

        Returns:
            message_id
        """
        message_id = str(uuid.uuid4())
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._pool.get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (message_id, session_id, role, content, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, meta_json, now)
            )
            # 更新会话的 updated_at
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id)
            )
            conn.commit()

        return message_id

    def get_messages(self, session_id: str, last_n: int = None) -> List[Dict[str, Any]]:
        """
        获取会话的消息列表

        Args:
            session_id: 会话ID
            last_n: 仅返回最近 N 条（None 表示全部）

        Returns:
            消息字典列表 [{'role': ..., 'content': ..., 'created_at': ..., 'metadata': ...}]
        """
        with self._pool.get_conn() as conn:
            if last_n:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at, metadata FROM (
                        SELECT role, content, created_at, metadata
                        FROM messages WHERE session_id = ?
                        ORDER BY created_at DESC LIMIT ?
                    ) sub ORDER BY created_at ASC
                    """,
                    (session_id, last_n)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, created_at, metadata "
                    "FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,)
                ).fetchall()

        return [
            {
                'role': r[0],
                'content': r[1],
                'created_at': r[2],
                'metadata': json.loads(r[3]) if r[3] else {},
            }
            for r in rows
        ]

    def save_task_state(self, session_id: str, task_json: str) -> str:
        """将任务状态持久化到 task_states 表。

        同一 session 只保留最新一条（先删后插）。

        Args:
            session_id: 会话ID
            task_json: 任务状态JSON字符串

        Returns:
            state_id
        """
        state_id = str(uuid.uuid4())
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with self._pool.get_conn() as conn:
            # 删除旧的任务状态（保留最新）
            conn.execute(
                "DELETE FROM task_states WHERE session_id = ?",
                (session_id,)
            )
            conn.execute(
                "INSERT INTO task_states (state_id, session_id, task_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (state_id, session_id, task_json, now)
            )
            conn.commit()

        self.logger.debug(f"任务状态已保存: session={session_id[:8]}...")
        return state_id

    def get_last_task_state(self, session_id: str) -> Optional[str]:
        """获取会话的最新任务状态JSON，不存在返回 None。"""
        with self._pool.get_conn() as conn:
            row = conn.execute(
                "SELECT task_json FROM task_states "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,)
            ).fetchone()
        return row[0] if row else None

    def get_message_count(self, session_id: str) -> int:
        """获取会话消息数"""
        with self._pool.get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,)
            ).fetchone()
        return row[0] if row else 0

    # ================================================================
    # 会话标题自动生成
    # ================================================================

    def auto_generate_title(self, session_id: str, first_user_message: str) -> str:
        """
        根据用户首条消息自动生成会话标题

        Args:
            session_id: 会话ID
            first_user_message: 用户首条消息

        Returns:
            生成的标题
        """
        # 简单策略：截取前30个字符作为标题
        title = first_user_message.strip()
        if len(title) > 30:
            title = title[:30] + '...'
        title = title.replace('\n', ' ')

        self.update_session_title(session_id, title)
        return title

    # ================================================================
    # 内部方法
    # ================================================================

    def _cleanup_old_sessions(self, user_id: str):
        """清理超过上限的旧会话"""
        sessions = self.list_sessions(user_id, limit=self.max_sessions_per_user + 10)
        if len(sessions) > self.max_sessions_per_user:
            to_delete = sessions[self.max_sessions_per_user:]
            for s in to_delete:
                self.delete_session(s.session_id)
            self.logger.info(
                f"清理用户 {user_id} 的 {len(to_delete)} 个旧会话"
            )
