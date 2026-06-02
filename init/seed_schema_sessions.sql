-- ============================================================
-- 智能购物平台 - 会话管理表结构定义
-- 存储于 data/sessions/sessions.db
-- ============================================================

-- 会话表
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '新会话',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user
ON sessions(user_id, updated_at DESC);

-- 会话消息表
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    metadata   TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session
ON messages(session_id, created_at ASC);

-- 任务状态独立存储表
CREATE TABLE IF NOT EXISTS task_states (
    state_id   TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_json  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_task_states_session
ON task_states(session_id, created_at DESC);
