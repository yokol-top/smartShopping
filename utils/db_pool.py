"""
SQLite 线程安全连接池

功能：
1. 基于 Queue 的连接复用，避免每次操作重新建立连接
2. 上下文管理器自动借出 / 归还，异常时自动回滚
3. WAL 模式提升并发读写性能
4. 可多处共享同一个池实例，也可各自创建独立池

用法：
    from utils.db_pool import ConnectionPool

    pool = ConnectionPool("/path/to/db.sqlite", max_size=5)

    with pool.get_conn() as conn:
        conn.execute("SELECT ...")
        conn.commit()
"""

import sqlite3
import threading
from contextlib import contextmanager
from queue import Queue
from typing import Optional


class ConnectionPool:
    """SQLite 线程安全连接池"""

    def __init__(self, db_path: str, max_size: int = 5):
        """
        Args:
            db_path: SQLite 数据库文件路径
            max_size: 池中最大连接数
        """
        self._db_path = db_path
        self._max_size = max_size
        self._pool: Queue = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created = 0

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def size(self) -> int:
        """当前已创建的连接总数"""
        return self._created

    @property
    def idle(self) -> int:
        """当前池中空闲的连接数"""
        return self._pool.qsize()

    def _create_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def get_conn(self):
        """
        从池中获取连接，用完自动归还。

        用法::

            with pool.get_conn() as conn:
                conn.execute(...)
                conn.commit()
        """
        conn: Optional[sqlite3.Connection] = None

        # 尝试从池中取一个空闲连接
        if not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
            except Exception:
                conn = None

        # 池中没有空闲连接，按需创建
        if conn is None:
            with self._lock:
                if self._created < self._max_size:
                    conn = self._create_conn()
                    self._created += 1
            # 如果已达上限，阻塞等待归还
            if conn is None:
                conn = self._pool.get()

        try:
            yield conn
        except Exception:
            # 出异常时回滚，避免脏状态
            conn.rollback()
            raise
        finally:
            # 归还到池中
            try:
                self._pool.put_nowait(conn)
            except Exception:
                conn.close()

    def close_all(self):
        """关闭池中所有连接"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                break
        self._created = 0
