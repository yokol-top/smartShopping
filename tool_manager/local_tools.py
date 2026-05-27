"""
本地工具集合

存放注册到 ToolManager 的本地 Python 工具。
每个工具是一个函数，接受 parameters: dict 参数，返回结果。
"""


def delete_user_orders(parameters: dict) -> dict:
    """按用户ID删除该用户的所有订单（需人工确认）

    Args:
        parameters: {"user_id": "UID-8888", "confirmed": true}

    Returns:
        操作结果字典
    """
    user_id = parameters.get("user_id", "")
    confirmed = parameters.get("confirmed", False)

    if not user_id:
        return {"success": False, "error": "缺少必填参数 user_id"}

    if not confirmed:
        return {
            "success": False,
            "requires_confirmation": True,
            "message": f"⚠️ 即将删除用户 {user_id} 的所有订单，此操作不可恢复。请确认后将 confirmed 设为 true 再次调用。",
        }

    # 模拟删除逻辑
    deleted_count = 3  # 模拟删除了3条订单
    return {
        "success": True,
        "message": f"已成功删除用户 {user_id} 的 {deleted_count} 条订单",
        "deleted_count": deleted_count,
        "user_id": user_id,
    }


# ================================================================
# 工具注册信息（供 agent.py 统一注册使用）
# ================================================================

LOCAL_TOOLS = [
    {
        "name": "delete_user_orders",
        "description": "按用户ID删除该用户的所有订单（高危操作，必须人工确认）",
        "handler": delete_user_orders,
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "要删除订单的用户ID，如 UID-8888"
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "是否已经人工确认。首次调用必须设为false，收到确认提示后设为true再次调用",
                    "default": False
                }
            },
            "required": ["user_id"]
        },
    },
]
