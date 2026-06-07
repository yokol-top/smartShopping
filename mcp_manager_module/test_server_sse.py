import os
import sys

# 将 test_project 加入搜索路径，以便复用共享的 ConnectionPool
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastmcp import FastMCP
import random
import sqlite3
import hashlib
import secrets
from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from utils.db_pool import ConnectionPool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

mcp = FastMCP("User_and_Order_Management_System")

# --- SQLite 数据库配置 ---
# 基于脚本位置定位：test_server_sse.py → mcp_manager_module/ → test_project/ → data/db/
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "db", "app.db")


os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
db_pool = ConnectionPool(DB_PATH, max_size=5)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


# ================================================================
# 数据库检查（建表和种子数据由 init/init_db.py 统一管理）
# ================================================================

def _check_db():
    """检查数据库是否已初始化，未初始化时给出提示"""
    try:
        with db_pool.get_conn() as conn:
            conn.execute("SELECT COUNT(*) FROM products").fetchone()
            conn.execute("SELECT COUNT(*) FROM users").fetchone()
        logging.info("数据库连接正常")
    except Exception as e:
        logging.error(
            f"数据库未初始化或表不存在: {e}\n"
            f"请先运行: python -m init.init_db"
        )
        raise SystemExit(1)


_check_db()


# ================================================================
# 权限检查工具函数
# ================================================================

def _get_user_role(identifier: str) -> Optional[str]:
    """查询用户角色，支持 user_id 或 username 查找，用户不存在返回 None"""
    with db_pool.get_conn() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE user_id = ? OR username = ?",
            (identifier, identifier)
        ).fetchone()
    return row[0] if row else None


def _is_admin(identifier: str) -> bool:
    return _get_user_role(identifier) == 'admin'


def _user_exists(identifier: str) -> bool:
    with db_pool.get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE user_id = ? OR username = ?",
            (identifier, identifier)
        ).fetchone()
    return row is not None


# ================================================================
# Pydantic 模型
# ================================================================

# --- 定义地址子模型 ---
class AddressModel(BaseModel):
    id: Optional[str] = Field(None, description="地址ID。新增时留空，修改或删除时必填")
    location: Optional[str] = Field(None, description="详细地址，如 '上海市黄浦区南京路1号'")
    phone_number: Optional[str] = Field(None, description="收获手机号，如 '1851111222'")
    tag: Optional[Literal["家", "公司", "学校"]] = Field(None, description="地址标签，仅限指定选项")


# --- 定义银行卡子模型 ---
class BankCardModel(BaseModel):
    id: Optional[str] = Field(None, description="卡片ID。新增时留空，修改或删除时必填")
    card_num: Optional[str] = Field(None, description="16位银行卡号")
    level: Optional[Literal["普通卡", "金卡", "白金卡"]] = Field("普通卡", description="卡片等级")


# --- 定义主更新请求模型 ---
class UpdateUserRequest(BaseModel):
    operator_id: str = Field(..., description="当前操作者的用户ID，用于权限校验。管理员可修改任何用户，普通用户只能修改自己")
    user_id: str = Field(..., description="必填的目标用户ID，格式如 UID-1234")
    new_username: Optional[str] = Field(None, description="新的用户名，如果不修改则不传")

    action: Optional[Literal["add", "update", "remove"]] = Field(
        None,
        description="操作类型：新增(add)、修改(update)或删除(remove)。与 address_data 或 card_data 配合使用；当添加一张银行卡是，请传入add；修改银行卡传入update；地址同理"
    )
    address_data: Optional[AddressModel] = Field(None, description="具体的地址数据内容。与 action 配合使用")
    card_data: Optional[BankCardModel] = Field(None, description="具体的银行卡数据内容。与 action 配合使用")


# ================================================================
# MCP 工具定义
# ================================================================

# --- 工具 1: 创建用户（仅管理员） ---
@mcp.tool()
def create_user(operator_id: str, username: str, password: str,
                role: str = "user", phone: str = "") -> str:
    """
    创建新用户。初始不含卡片和地址。
    权限：仅管理员(admin)可创建新用户。
    - operator_id: 当前操作者的用户ID，必须是管理员
    - role: 用户角色，默认 'user'，可选 'admin'
    """
    # 权限校验
    if not _is_admin(operator_id):
        return f"权限不足：只有管理员才能创建新用户。当前操作者 {operator_id} 不是管理员。"

    user_id = f"UID-{random.randint(1000, 9999)}"
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    with db_pool.get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users (user_id, username, password_hash, salt, role, phone, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, username, password_hash, salt, role, phone, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return f"创建失败：用户ID {user_id} 已存在，请重试。"

    return f"用户创建成功！用户ID: {user_id}，角色: {role}"


# --- 工具 2: 查询用户详情 ---
@mcp.tool()
def get_user_detail(user_id: str) -> str:
    """根据用户ID或用户名查询用户的所有详细信息，包括角色、银行卡详情和地址详情。"""
    with db_pool.get_conn() as conn:
        user_row = conn.execute(
            "SELECT user_id, username, role, phone, created_at FROM users WHERE user_id = ? OR username = ?",
            (user_id, user_id)
        ).fetchone()
        if not user_row:
            return "用户不存在。"

        actual_uid = user_row[0]
        addresses = conn.execute(
            "SELECT id, location, phone_number, tag FROM addresses WHERE user_id = ?",
            (actual_uid,)
        ).fetchall()
        cards = conn.execute(
            "SELECT id, card_num, level FROM bank_cards WHERE user_id = ?",
            (actual_uid,)
        ).fetchall()

    user_info = {
        "user_id": user_row[0], "username": user_row[1], "role": user_row[2],
        "phone": user_row[3], "created_at": user_row[4],
        "addresses": [{"id": a[0], "location": a[1], "phone_number": a[2], "tag": a[3]} for a in addresses],
        "cards": [{"id": c[0], "card_num": c[1], "level": c[2]} for c in cards],
    }
    logging.info(f"查询用户:{user_id} : {user_info}")
    return f"用户信息: {user_info}"


# --- 工具 3: 修改用户信息（含权限校验） ---
@mcp.tool()
def update_user_profile_refined(request: UpdateUserRequest) -> str:
    """
    修改用户核心画像。支持增量修改用户名、地址簿或银行卡列表。
    权限：管理员可修改任何用户，普通用户只能修改自己的信息。
    注意：
        1. 这是一个复合工具，模型应根据用户意图（如"搬家了"或"办了张金卡"）自动填充对应的 action 和 data。例如创建地址，action:add,address_data:{location:北京,phone_number:123}
        2. 如果修改地址或卡片，必须提供对应的 id（新增除外）。
    """
    logging.info(f"update_user_profile_refined 入参:{request.model_dump(exclude_none=True)}")

    # 权限校验：普通用户只能修改自己
    if not _is_admin(request.operator_id) and request.operator_id != request.user_id:
        return (f"权限不足：普通用户只能修改自己的信息。"
                f"操作者 {request.operator_id} 无权修改用户 {request.user_id}。")

    if not _user_exists(request.user_id):
        return f"失败：未找到用户 {request.user_id}，请先确认用户ID是否正确。"

    # 参数一致性校验（不需要连接，提前返回）
    if request.action and not (request.address_data or request.card_data):
        return "失败：action 已提供，但缺少 address_data 或 card_data。"
    if (request.address_data or request.card_data) and not request.action:
        return "失败：address_data 或 card_data 已提供，但缺少 action（add/update/remove）。"

    log_msg = []

    with db_pool.get_conn() as conn:
        # 1. 用户名修改
        if request.new_username:
            logging.info(f"用户:{request.user_id}，修改用户名为: {request.new_username}")
            conn.execute("UPDATE users SET username = ? WHERE user_id = ?",
                         (request.new_username, request.user_id))
            log_msg.append(f"用户名已变更为 {request.new_username}")

        # 2. 地址处理逻辑
        if request.address_data and request.action:
            data = request.address_data.model_dump(exclude_none=True)
            logging.info(f"用户:{request.user_id}，{request.action}地址: {data}")
            if data.get("tag") is None:
                return "失败：请设置地址标签"
            if request.action == "add":
                addr_id = f"ADDR-{random.randint(100, 999)}"
                conn.execute(
                    "INSERT INTO addresses (id, user_id, location, phone_number, tag) VALUES (?, ?, ?, ?, ?)",
                    (addr_id, request.user_id, data.get('location', ''),
                     data.get('phone_number', ''), data.get('tag', ''))
                )
                log_msg.append(f"新增地址 {addr_id}")
            elif request.action == "update":
                if not data.get("id"):
                    return "失败：更新地址需要提供 address_data.id。"
                sets, vals = [], []
                for col in ('location', 'phone_number', 'tag'):
                    if col in data:
                        sets.append(f"{col} = ?")
                        vals.append(data[col])
                if sets:
                    vals.append(data['id'])
                    conn.execute(f"UPDATE addresses SET {', '.join(sets)} WHERE id = ?", vals)
                log_msg.append(f"更新了地址 {data.get('id')}")
            elif request.action == "remove":
                if not data.get("id"):
                    return "失败：删除地址需要提供 address_data.id。"
                conn.execute("DELETE FROM addresses WHERE id = ? AND user_id = ?",
                             (data['id'], request.user_id))
                log_msg.append(f"删除了地址 {data.get('id')}")

        # 3. 银行卡处理逻辑
        if request.card_data and request.action:
            data = request.card_data.model_dump(exclude_none=True)
            logging.info(f"用户:{request.user_id}，{request.action}银行卡: {data}")
            if request.action == "add":
                card_id = f"CARD-{random.randint(100, 999)}"
                conn.execute(
                    "INSERT INTO bank_cards (id, user_id, card_num, level) VALUES (?, ?, ?, ?)",
                    (card_id, request.user_id, data.get('card_num', ''),
                     data.get('level', '普通卡'))
                )
                log_msg.append(f"绑定了新卡 {card_id}")
            elif request.action == "update":
                if not data.get("id"):
                    return "失败：更新银行卡需要提供 card_data.id。"
                sets, vals = [], []
                for col in ('card_num', 'level'):
                    if col in data:
                        sets.append(f"{col} = ?")
                        vals.append(data[col])
                if sets:
                    vals.append(data['id'])
                    conn.execute(f"UPDATE bank_cards SET {', '.join(sets)} WHERE id = ?", vals)
                log_msg.append(f"更新了银行卡 {data.get('id')}")
            elif request.action == "remove":
                if not data.get("id"):
                    return "失败：删除银行卡需要提供 card_data.id。"
                conn.execute("DELETE FROM bank_cards WHERE id = ? AND user_id = ?",
                             (data['id'], request.user_id))
                log_msg.append(f"删除了银行卡 {data.get('id')}")

        conn.commit()

    return "成功：" + "；".join(log_msg) if log_msg else "未执行任何修改，请检查输入参数。"


@mcp.tool()
def create_complex_order(
        product_ids: str,
        quantity: int,
        user_id: str,
        customer_name: str,
        address_id: str,
        card_id: str,
        idempotency_key: Optional[str] = None
) -> str:
    """
    批量创建订单。支持同时为多个商品下单，每个商品创建一笔独立订单。
    - product_ids: 商品ID，多个用英文逗号分隔，如 "P001" 或 "P001,P005,P008"
    - quantity: 每个商品的购买数量（所有商品统一数量）
    - user_id: 用户ID
    - customer_name: 收货人姓名
    - address_id: 收货地址ID
    - card_id: 支付银行卡ID
    - idempotency_key: 幂等键，相同 key 的重复请求直接返回首次创建的订单，避免重复下单
    """
    if not _user_exists(user_id):
        return f"失败：未找到用户 {user_id}，请先确认用户ID是否正确。"

    # 幂等检查：批量下单时每行存 "{key}_{pid}"，用前缀 LIKE 查询整批是否已创建
    if idempotency_key:
        with db_pool.get_conn() as conn:
            existing = conn.execute(
                "SELECT order_id, product, quantity, status "
                "FROM orders WHERE idempotency_key LIKE ?",
                (f"{idempotency_key}_%",)
            ).fetchall()
        if existing:
            summary = f"[幂等] 该请求已处理，返回原有订单（共 {len(existing)} 笔）：\n"
            for row in existing:
                summary += f"  ✓ {row[0]} - 商品:{row[1]} x{row[2]} 状态:{row[3]}\n"
            return summary

    # 解析商品ID列表
    pid_list = [pid.strip() for pid in product_ids.split(",") if pid.strip()]
    if not pid_list:
        return "失败：商品ID不能为空。"

    with db_pool.get_conn() as conn:
        # 校验地址ID（须属于该用户）
        addr = conn.execute(
            "SELECT id, location FROM addresses WHERE id = ? AND user_id = ?",
            (address_id, user_id)
        ).fetchone()
        if not addr:
            return f"失败：未找到地址 {address_id}，请确认地址ID是否正确且属于当前用户。"

        # 校验银行卡ID（须属于该用户）
        card = conn.execute(
            "SELECT id, level FROM bank_cards WHERE id = ? AND user_id = ?",
            (card_id, user_id)
        ).fetchone()
        if not card:
            return f"失败：未找到银行卡 {card_id}，请确认卡ID是否正确且属于当前用户。"

        # 先全部校验商品
        products = {}
        invalid_pids = []
        for pid in pid_list:
            product = conn.execute(
                "SELECT id, name, price FROM products WHERE id = ?", (pid,)
            ).fetchone()
            if not product:
                invalid_pids.append(pid)
            else:
                products[pid] = product

        if invalid_pids:
            return f"失败：以下商品ID不存在: {', '.join(invalid_pids)}，请确认后重新提交。"

        # 全部校验通过，批量创建订单
        results = []
        for pid in pid_list:
            product = products[pid]
            order_id = f"ORD-{random.randint(1000, 9999)}"
            # 每行用 "{key}_{pid}" 避免同一批次多商品共用同一 key 触发 UNIQUE 约束
            row_idem_key = f"{idempotency_key}_{pid}" if idempotency_key else None
            conn.execute(
                "INSERT INTO orders "
                "(order_id, user_id, product, quantity, customer, address, card_end, status, created_at, idempotency_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (order_id, user_id, pid, quantity, customer_name, address_id,
                 card_id, '已下单', datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 row_idem_key)
            )
            results.append(f"✓ {order_id} - {product[1]}（¥{product[2]}）x{quantity}")
        conn.commit()

    summary = f"批量下单完成：成功 {len(results)}/{len(pid_list)} 笔\n"
    summary += f"收货地址: {addr[1]}，支付卡: {card[1]}（{card_id}）\n"
    for r in results:
        summary += f"  {r}\n"
    return summary


@mcp.tool()
def query_order_detail(order_id: str) -> str:
    """根据订单号查询单个订单的详细信息，包含商品、地址、银行卡的ID和详情。"""
    with db_pool.get_conn() as conn:
        row = conn.execute(
            "SELECT o.order_id, o.user_id, o.product, o.quantity, o.customer, "
            "       o.address, o.card_end, o.status, o.created_at, "
            "       p.name AS product_name, p.price AS product_price, "
            "       a.location AS address_detail, a.phone_number AS address_phone, "
            "       c.level AS card_level "
            "FROM orders o "
            "LEFT JOIN products p ON o.product = p.id "
            "LEFT JOIN addresses a ON o.address = a.id "
            "LEFT JOIN bank_cards c ON o.card_end = c.id "
            "WHERE o.order_id = ?", (order_id,)
        ).fetchone()
    if not row:
        return f"未找到单号为 {order_id} 的订单。"
    order = {
        "order_id": row[0], "user_id": row[1],
        "product_id": row[2], "product_name": row[9] or row[2], "product_price": row[10],
        "quantity": row[3], "customer": row[4],
        "address_id": row[5], "address_detail": row[11] or row[5], "address_phone": row[12],
        "card_id": row[6], "card_level": row[13] or row[6],
        "status": row[7], "created_at": row[8],
    }
    return f"订单详情: {order}"


@mcp.tool(description="列出指定用户的所有订单列表。")
def list_all_orders(user_id: str) -> str:
    """列出指定用户的所有订单列表"""
    with db_pool.get_conn() as conn:
        rows = conn.execute(
            "SELECT o.order_id, o.product, o.quantity, o.status, o.address, o.card_end, "
            "       p.name AS product_name, p.price AS product_price, "
            "       a.location AS address_detail, "
            "       c.level AS card_level "
            "FROM orders o "
            "LEFT JOIN products p ON o.product = p.id "
            "LEFT JOIN addresses a ON o.address = a.id "
            "LEFT JOIN bank_cards c ON o.card_end = c.id "
            "WHERE o.user_id = ? ORDER BY o.created_at DESC",
            (user_id,)
        ).fetchall()

    if not rows:
        return f"用户 {user_id} 当前没有任何订单。"

    summary = f"用户 {user_id} 的订单列表：\n"
    for oid, product_id, qty, status, addr_id, card_id, product_name, price, addr_detail, card_level in rows:
        p_display = f"{product_name}({product_id})" if product_name else product_id
        a_display = f"{addr_detail}({addr_id})" if addr_detail else addr_id
        c_display = f"{card_level}({card_id})" if card_level else card_id
        summary += (f"- [{oid}] 商品: {p_display} x{qty} ¥{price or '?'}"
                    f" | 地址: {a_display} | 支付卡: {c_display} | 状态: {status}\n")
    return summary


# --- 工具 6: 搜索商品 ---
@mcp.tool()
def search_products(
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        product_ids: Optional[str] = None,
) -> str:
    """
    搜索商品列表。支持按关键词、类目、价格区间、商品ID列表进行筛选，所有条件可自由组合。
    - keyword: 商品名称关键词（模糊匹配），如 '华为'、'耳机'
    - category: 商品类目，如 '手机'、'笔记本电脑'、'耳机'、'平板电脑'、'家电'、'运动鞋'、'智能手表'
    - min_price: 最低价格
    - max_price: 最高价格
    - product_ids: 按商品ID批量查询，多个用英文逗号分隔，如 "P001,P005,P008"
    """
    logging.info(f"搜索商品 - keyword:{keyword}, category:{category}, min_price:{min_price}, max_price:{max_price}, product_ids:{product_ids}")

    conditions = []
    params = []

    if product_ids:
        pid_list = [pid.strip() for pid in product_ids.split(",") if pid.strip()]
        if pid_list:
            placeholders = ",".join("?" * len(pid_list))
            conditions.append(f"id IN ({placeholders})")
            params.extend(pid_list)

    if keyword:
        conditions.append("(name LIKE ? OR brand LIKE ? OR description LIKE ?)")
        like_val = f"%{keyword}%"
        params.extend([like_val, like_val, like_val])

    if category:
        conditions.append("category = ?")
        params.append(category)

    if min_price is not None:
        conditions.append("price >= ?")
        params.append(min_price)

    if max_price is not None:
        conditions.append("price <= ?")
        params.append(max_price)

    sql = "SELECT id, name, category, brand, price, stock, description FROM products"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY price ASC"

    with db_pool.get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "未找到符合条件的商品。请尝试调整搜索条件。"

    summary = f"找到 {len(rows)} 件商品：\n"
    for pid, name, cat, brand, price, stock, desc in rows:
        summary += (f"- 商品ID: {pid} | 商品名称: {name}  |  类目: {cat}  |  品牌: {brand}"
                    f"  |  价格: ¥{price}  |  库存: {stock}件\n"
                    f"  描述: {desc}\n")
    return summary


if __name__ == "__main__":
    mcp.run(transport="sse")
