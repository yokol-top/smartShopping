"""
SmartAgent Web API

基于 FastAPI 的 Web 服务，为前端聊天界面提供 API 接口。

启动方式：
    cd test_project
    uvicorn web.app:app --reload --port 8080

API 概览：
    POST /api/login          登录
    POST /api/chat           发送消息
    GET  /api/sessions       会话列表
    POST /api/sessions/new   创建新会话
    POST /api/sessions/{id}/resume  切换会话
    DELETE /api/sessions/{id}       删除单个会话
    DELETE /api/sessions            删除所有会话
    GET  /api/sessions/{id}/history 对话历史
    GET  /api/tools                 工具列表
    GET  /api/rate-limit-stats      频率限制统计
"""

import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# 确保 test_project 的父目录在 sys.path 中，使 `from xxx` 可用
# web/app.py → test_project/web/app.py → 需要 AILearn/ 在 path 上
_this_dir = os.path.dirname(os.path.abspath(__file__))          # web/
_test_project_dir = os.path.dirname(_this_dir)                  # test_project/
_workspace_root = os.path.dirname(_test_project_dir)            # AILearn/
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

# 将 CWD 切换到 test_project/，确保所有相对路径（config、MCP、数据库等）一致
os.chdir(_test_project_dir)

from auth import AuthManager
from utils import ConfigLoader
from agent.agent import SmartAgent

# ================================================================
# 初始化
# ================================================================

# 使用基于脚本位置的绝对路径加载配置，不依赖 CWD
_config_path = os.path.join(_test_project_dir, "config", "settings.yaml")
config = ConfigLoader(_config_path).config
auth_mgr = AuthManager(config=config)

# 缓存已登录用户的 Agent 实例：{user_id: SmartAgent}
_agents: dict = {}

app = FastAPI(title="SmartAgent Web API", version="1.0.0")

# 静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ================================================================
# 请求/响应模型
# ================================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    user_id: str
    message: str


class SessionActionRequest(BaseModel):
    user_id: str


# ================================================================
# 辅助函数
# ================================================================

def _get_or_create_agent(user_id: str, username: str, session_id: str = None) -> SmartAgent:
    """获取或创建 Agent 实例（每个用户一个 Agent）"""
    if user_id in _agents:
        return _agents[user_id]

    agent = SmartAgent(
        config_path=_config_path,
        user_id=user_id,
        username=username,
        session_id=session_id,
    )
    _agents[user_id] = agent
    return agent


def _get_agent(user_id: str) -> Optional[SmartAgent]:
    """查找用户当前 Agent"""
    return _agents.get(user_id)


# ================================================================
# 页面路由
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    return FileResponse(os.path.join(static_dir, "index.html"))


# ================================================================
# API 路由
# ================================================================

@app.post("/api/login")
async def login(req: LoginRequest):
    """登录"""
    token_info = auth_mgr.login(req.username, req.password)
    if not token_info:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 创建默认 Agent
    agent = _get_or_create_agent(token_info.user_id, token_info.username)

    return {
        "success": True,
        "user_id": token_info.user_id,
        "username": token_info.username,
        "session_id": agent.session_id,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """发送消息"""
    agent = _get_agent(req.user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    try:
        response = agent.chat(req.message, verbose=True)
        return {
            "success": True,
            "response": response,
            "session_id": agent.session_id,
        }
    except Exception as e:
        return {
            "success": False,
            "response": f"处理出错: {str(e)}",
            "session_id": agent.session_id,
        }


@app.get("/api/sessions")
async def list_sessions(user_id: str):
    """获取会话列表"""
    agent = _get_agent(user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    sessions = agent.session_manager.list_sessions(user_id, limit=20)
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "title": s.title,
                "message_count": s.message_count,
                "updated_at": s.updated_at,
                "is_current": s.session_id == agent.session_id,
            }
            for s in sessions
        ],
        "current_session_id": agent.session_id,
    }


@app.post("/api/sessions/new")
async def new_session(req: SessionActionRequest):
    """创建新会话"""
    agent = _get_agent(req.user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    agent.reset_session()
    return {"success": True, "session_id": agent.session_id}


@app.post("/api/sessions/{session_id}/resume")
async def resume_session(session_id: str, req: SessionActionRequest):
    """切换到历史会话"""
    agent = _get_agent(req.user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    ok = agent.resume_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {"success": True, "session_id": session_id}


class UpdateTitleRequest(BaseModel):
    user_id: str
    title: str


@app.put("/api/sessions/{session_id}/title")
async def update_session_title(session_id: str, req: UpdateTitleRequest):
    """更新会话标题"""
    agent = _get_agent(req.user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    agent.session_manager.update_session_title(session_id, req.title)
    return {"success": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """删除单个会话"""
    agent = _get_agent(user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    agent.delete_sessions([session_id])
    return {"success": True}


@app.delete("/api/sessions")
async def delete_all_sessions(user_id: str):
    """删除所有会话"""
    agent = _get_agent(user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    deleted = agent.delete_all_sessions()
    return {"success": True, "deleted": deleted}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str, user_id: str, last_n: int = 50):
    """获取对话历史"""
    agent = _get_agent(user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    messages = agent.session_manager.get_messages(session_id, last_n=last_n)
    session_info = agent.session_manager.get_session(session_id)
    return {
        "title": session_info.title if session_info else "新对话",
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
    }


@app.get("/api/tools")
async def list_tools(user_id: str):
    """获取工具列表"""
    agent = _get_agent(user_id)
    if not agent:
        raise HTTPException(status_code=401, detail="请先登录")

    tools = agent.tool_manager.list_tools_brief()
    stats = agent.rate_limiter.get_stats()
    return {"tools": tools, "rate_limit_stats": stats}
