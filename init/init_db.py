#!/usr/bin/env python3
"""
数据库初始化脚本

从SQL文件加载表结构和种子数据，初始化SQLite数据库。
用户密码的hash和salt在运行时动态生成（不存储明文密码到SQL文件中）。

用法:
    python -m init.init_db              # 使用默认路径
    python -m init.init_db --db ./data/db/app.db  # 指定数据库路径
    python -m init.init_db --reset      # 重建数据库（删除旧数据）
"""

import os
import sys
import sqlite3
import hashlib
import secrets
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 文件路径（相对于本脚本）
INIT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_SQL = os.path.join(INIT_DIR, 'seed_schema.sql')
SCHEMA_SESSIONS_SQL = os.path.join(INIT_DIR, 'seed_schema_sessions.sql')
DATA_SQL = os.path.join(INIT_DIR, 'seed_data.sql')

# 默认数据库路径
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(INIT_DIR), 'data', 'db', 'app.db'
)
DEFAULT_SESSIONS_DB_PATH = os.path.join(
    os.path.dirname(INIT_DIR), 'data', 'sessions', 'sessions.db'
)

# 种子用户（密码在运行时hash，不写入SQL文件）
SEED_USERS = [
    {
        'user_id': 'UID-0001',
        'username': 'admin',
        'password': 'admin123',
        'role': 'admin',
        'phone': '13800000000',
    },
    {
        'user_id': 'UID-8888',
        'username': 'test',
        'password': 'test123',
        'role': 'user',
        'phone': '18511112222',
    },
]


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


def init_database(db_path: str = None, reset: bool = False):
    """初始化数据库

    Args:
        db_path: 数据库文件路径（默认 ./data/db/app.db）
        reset: 是否重建（删除旧数据库重新创建）
    """
    db_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if reset and os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"已删除旧数据库: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. 执行表结构SQL
        logger.info(f"执行表结构: {SCHEMA_SQL}")
        with open(SCHEMA_SQL, 'r', encoding='utf-8') as f:
            cursor.executescript(f.read())

        # 2. 插入种子用户（需要运行时hash密码）
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            for user in SEED_USERS:
                salt = secrets.token_hex(16)
                password_hash = _hash_password(user['password'], salt)
                cursor.execute(
                    "INSERT INTO users (user_id, username, password_hash, salt, role, phone, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user['user_id'], user['username'], password_hash, salt,
                     user['role'], user['phone'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
            logger.info(f"已初始化 {len(SEED_USERS)} 个种子用户")
        else:
            logger.info("用户表已有数据，跳过种子用户")

        # 3. 执行商品种子数据SQL
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] == 0:
            logger.info(f"执行种子数据: {DATA_SQL}")
            with open(DATA_SQL, 'r', encoding='utf-8') as f:
                cursor.executescript(f.read())
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]
            logger.info(f"已初始化 {count} 条商品数据")
        else:
            logger.info("商品表已有数据，跳过种子数据")

        conn.commit()
        logger.info(f"✅ 数据库初始化完成: {db_path}")

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 数据库初始化失败: {e}")
        raise
    finally:
        conn.close()


def init_sessions_database(db_path: str = None, reset: bool = False):
    """初始化会话数据库（data/sessions/sessions.db）

    Args:
        db_path: 会话数据库路径（默认 ./data/sessions/sessions.db）
        reset: 是否重建
    """
    db_path = db_path or DEFAULT_SESSIONS_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if reset and os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"已删除旧会话数据库: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        logger.info(f"执行会话表结构: {SCHEMA_SESSIONS_SQL}")
        with open(SCHEMA_SESSIONS_SQL, 'r', encoding='utf-8') as f:
            conn.executescript(f.read())
        conn.commit()
        logger.info(f"✅ 会话数据库初始化完成: {db_path}")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 会话数据库初始化失败: {e}")
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='初始化智能购物平台数据库')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH,
                        help=f'业务数据库路径 (默认: {DEFAULT_DB_PATH})')
    parser.add_argument('--sessions-db', type=str, default=DEFAULT_SESSIONS_DB_PATH,
                        help=f'会话数据库路径 (默认: {DEFAULT_SESSIONS_DB_PATH})')
    parser.add_argument('--reset', action='store_true',
                        help='重建数据库（删除旧数据）')
    args = parser.parse_args()

    init_database(db_path=args.db, reset=args.reset)
    init_sessions_database(db_path=args.sessions_db, reset=args.reset)


if __name__ == '__main__':
    main()
