"""
意图推断工具函数（公共模块）

从任务描述文字中轻量推断意图类型和复杂度，供无法调用 IntentRecognizer LLM 的场景使用：
- Orchestrator._infer_subtask_intent  （子任务意图推断）
- DynamicSubAgent._infer_intent       （子Agent自用意图推断）

规则完全基于关键词匹配，不调用 LLM。
"""

from typing import List, Optional, Tuple


def infer_intent_from_desc(
    desc: str,
    allowed_tools: Optional[List[str]] = None,
) -> Tuple[object, Optional[str], object]:
    """根据任务描述推断意图类型、工具名、复杂度（轻量版，无 LLM 调用）。

    Args:
        desc:          任务描述文字（调用方负责转 lower）
        allowed_tools: 已知可用工具列表（有值时直接推断为 MCP_EXECUTE）

    Returns:
        (IntentType, tool_name_or_None, TaskComplexity)
    """
    # 延迟导入，避免循环依赖
    from .intent_recognizer import IntentType, TaskComplexity

    # ── 意图类型推断 ──────────────────────────────────────────
    if allowed_tools:
        intent_type = IntentType.MCP_EXECUTE
        tool_name: Optional[str] = allowed_tools[0]
    else:
        tool_name = None
        rag_hints = [
            '推荐', '搜索商品', '查询商品', '了解', '对比', '哪款', '多少钱',
            '知识库', '商品信息', '参数', '评测', '选购', '怎么样', '好不好',
        ]
        if any(kw in desc for kw in rag_hints):
            intent_type = IntentType.RAG_SIMPLE
        else:
            mcp_hints = [
                '下单', '购买', '创建订单', '查询订单', '修改', '添加地址',
                '获取用户', '更新用户', '银行卡', '支付',
            ]
            if any(kw in desc for kw in mcp_hints):
                intent_type = IntentType.MCP_EXECUTE
            else:
                intent_type = IntentType.SIMPLE_CHAT

    # ── 复杂度推断 ────────────────────────────────────────────
    # 仅当出现 ≥2 个多步骤连接词，或连接词与操作动词同时出现时，才判定为 MEDIUM
    multi_step_hints = ['然后', '接着', '之后', '再', '并且', '同时', '最后', '先']
    multi_step_count = sum(1 for kw in multi_step_hints if kw in desc)
    has_action = any(kw in desc for kw in ['下单', '购买', '创建', '查询', '搜索', '推荐'])
    if multi_step_count >= 2 or (multi_step_count >= 1 and has_action and len(desc) > 20):
        complexity = TaskComplexity.MEDIUM
    else:
        complexity = TaskComplexity.SIMPLE

    return intent_type, tool_name, complexity
