-- ============================================================
-- 智能购物平台 - 数据库表结构定义
-- ============================================================

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    phone         TEXT DEFAULT '',
    created_at    TEXT NOT NULL
);

-- 收货地址表
CREATE TABLE IF NOT EXISTS addresses (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    location     TEXT NOT NULL DEFAULT '',
    phone_number TEXT DEFAULT '',
    tag          TEXT DEFAULT '',
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 银行卡表
CREATE TABLE IF NOT EXISTS bank_cards (
    id       TEXT PRIMARY KEY,
    user_id  TEXT NOT NULL,
    card_num TEXT NOT NULL,
    level    TEXT NOT NULL DEFAULT '普通卡',
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 订单表
CREATE TABLE IF NOT EXISTS orders (
    order_id   TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    product    TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    customer   TEXT NOT NULL,
    address    TEXT NOT NULL,
    card_end   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT '已下单',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 商品表
CREATE TABLE IF NOT EXISTS products (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    price       REAL NOT NULL,
    brand       TEXT NOT NULL,
    stock       INTEGER NOT NULL DEFAULT 0,
    description TEXT DEFAULT ''
);

