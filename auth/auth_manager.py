"""
认证管理模块

功能：
1. 用户名密码登录验证（SQLite 数据库存储）
2. 本地 Token 缓存（JSON 文件），用于免登录
3. Token 有效期校验（可配置，默认2天）
4. 自动复用未过期 Token 时，从数据库获取最新用户数据
"""

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional

from utils.db_pool import ConnectionPool


@dataclass
class TokenInfo:
    """本地 Token 信息"""
    user_id: str
    username: str
    token: str
    created_at: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenInfo':
        return cls(**data)


class AuthManager:
    """
    认证管理器

    - 用户数据存储在 SQLite 数据库中（通过 ConnectionPool）
    - 登录时从数据库校验用户名密码
    - 登录成功后在本地生成 Token 并缓存到 JSON 文件
    - 再次访问时检查本地 Token，未过期则跳过密码验证，
      但仍从数据库获取最新用户信息
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        auth_config = (config or {}).get('auth', {})

        # Token 有效期（秒），默认 2 天
        self.token_ttl = auth_config.get('token_ttl_hours', 48) * 3600

        # 存储路径
        self.data_dir = auth_config.get('data_dir', '/Users/weyang3/PycharmProjects/AILearn/data/db')
        os.makedirs(self.data_dir, exist_ok=True)

        # SQLite 数据库（用户数据）
        db_path = os.path.join(self.data_dir, 'app.db')
        pool_size = auth_config.get('pool_size', 3)
        self._pool = ConnectionPool(db_path, max_size=pool_size)

        # 本地 Token 缓存（JSON 文件，仅客户端用于免登录）
        self.tokens_file = os.path.join(self.data_dir, 'tokens.json')
        self._tokens: Dict[str, Dict[str, Any]] = self._load_json(self.tokens_file)

        self.logger.info(
            f"AuthManager 初始化 | DB: {db_path} | Token有效期: {self.token_ttl // 3600}h"
        )

    # ================================================================
    # 公开接口
    # ================================================================

    def login(self, username: str, password: str) -> Optional[TokenInfo]:
        """
        用户登录（从数据库校验用户名密码）

        Args:
            username: 用户名
            password: 密码

        Returns:
            登录成功返回 TokenInfo，失败返回 None
        """
        user = self._get_user_by_username(username)
        if not user:
            self.logger.warning(f"登录失败: 用户 '{username}' 不存在")
            return None

        password_hash = self._hash_password(password, user['salt'])
        if password_hash != user['password_hash']:
            self.logger.warning(f"登录失败: 用户 '{username}' 密码错误")
            return None

        # 登录成功，生成本地 Token
        token_info = self._generate_token(user['user_id'], username)
        self.logger.info(f"用户 '{username}' 登录成功，Token 有效至 "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(token_info.expires_at))}")
        return token_info

    def validate_token(self, token: str) -> Optional[TokenInfo]:
        """
        校验本地 Token 是否有效，有效则从数据库刷新用户数据

        Args:
            token: Token 字符串

        Returns:
            有效返回 TokenInfo（含最新用户信息），无效/过期返回 None
        """
        token_data = self._tokens.get(token)
        if not token_data:
            return None

        token_info = TokenInfo.from_dict(token_data)
        if token_info.is_expired:
            self.logger.info(f"Token 已过期: 用户 '{token_info.username}'")
            del self._tokens[token]
            self._save_json(self.tokens_file, self._tokens)
            return None

        # Token 有效，从数据库获取最新用户数据确认用户仍存在
        user = self._get_user_by_username(token_info.username)
        if not user:
            self.logger.warning(f"Token 对应用户 '{token_info.username}' 已不存在，清除 Token")
            del self._tokens[token]
            self._save_json(self.tokens_file, self._tokens)
            return None

        # 用最新 user_id 更新（防止不一致）
        token_info.user_id = user['user_id']
        return token_info

    def get_local_token(self, username: str) -> Optional[TokenInfo]:
        """
        获取用户本地未过期的 Token，并从数据库验证用户仍有效（免登录）

        Args:
            username: 用户名

        Returns:
            未过期且用户有效的 TokenInfo，否则 None
        """
        for token_str, token_data in list(self._tokens.items()):
            token_info = TokenInfo.from_dict(token_data)
            if token_info.username == username:
                if token_info.is_expired:
                    del self._tokens[token_str]
                    self._save_json(self.tokens_file, self._tokens)
                    continue

                # 从数据库确认用户仍然存在
                user = self._get_user_by_username(username)
                if not user:
                    self.logger.warning(f"用户 '{username}' 已不存在，清除本地 Token")
                    del self._tokens[token_str]
                    self._save_json(self.tokens_file, self._tokens)
                    return None

                # 用最新的 user_id 更新
                token_info.user_id = user['user_id']
                self.logger.info(f"用户 '{username}' 本地 Token 有效，免登录")
                return token_info
        return None

    def logout(self, token: str):
        """注销：删除本地 Token"""
        if token in self._tokens:
            username = self._tokens[token].get('username', '?')
            del self._tokens[token]
            self._save_json(self.tokens_file, self._tokens)
            self.logger.info(f"用户 '{username}' 已注销")

    def register(self, username: str, password: str) -> Optional[str]:
        """
        注册新用户（写入数据库）

        Returns:
            成功返回 user_id，用户已存在返回 None
        """
        if self._get_user_by_username(username):
            self.logger.warning(f"注册失败: 用户 '{username}' 已存在")
            return None

        salt = secrets.token_hex(16)
        user_id = f"U-{secrets.token_hex(4).upper()}"

        with self._pool.get_conn() as conn:
            conn.execute(
                "INSERT INTO users (user_id, username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, self._hash_password(password, salt), salt,
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()

        self.logger.info(f"用户 '{username}' 注册成功，ID: {user_id}")
        return user_id

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """
        从数据库获取用户信息（公开接口，不含密码哈希）

        Returns:
            {'user_id': ..., 'username': ..., 'created_at': ...} 或 None
        """
        user = self._get_user_by_username(username)
        if not user:
            return None
        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'created_at': user['created_at'],
        }

    # ================================================================
    # 内部方法
    # ================================================================

    def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """从数据库查询用户（内部方法，含密码哈希）"""
        with self._pool.get_conn() as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash, salt, created_at "
                "FROM users WHERE username = ?",
                (username,)
            ).fetchone()
        if not row:
            return None
        return {
            'user_id': row[0],
            'username': row[1],
            'password_hash': row[2],
            'salt': row[3],
            'created_at': row[4],
        }

    def _generate_token(self, user_id: str, username: str) -> TokenInfo:
        """生成 Token 并持久化到本地 JSON"""
        token = secrets.token_urlsafe(32)
        now = time.time()
        token_info = TokenInfo(
            user_id=user_id,
            username=username,
            token=token,
            created_at=now,
            expires_at=now + self.token_ttl,
        )

        # 清理该用户旧 Token
        self._tokens = {
            k: v for k, v in self._tokens.items()
            if v.get('username') != username
        }

        self._tokens[token] = token_info.to_dict()
        self._save_json(self.tokens_file, self._tokens)
        return token_info

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

    @staticmethod
    def _load_json(path: str) -> Dict:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    @staticmethod
    def _save_json(path: str, data: Dict):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
