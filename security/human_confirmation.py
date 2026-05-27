"""
人类确认模块 (Human Confirmation)

职责：
对于敏感操作（如删除、修改重要数据等），要求人类确认后才执行。
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ConfirmationLevel(str, Enum):
    """确认级别"""
    NONE = "none"               # 无需确认，直接执行
    INFORM = "inform"           # 告知用户即将执行，但不等待确认
    CONFIRM = "confirm"         # 需要用户明确确认
    DOUBLE_CONFIRM = "double"   # 双重确认（高危操作）


@dataclass
class ConfirmationResult:
    """确认结果"""
    confirmed: bool
    level: ConfirmationLevel
    operation: str
    user_response: str = ""
    reason: str = ""


class HumanConfirmation:
    """
    人类确认管理器

    根据操作类型和风险级别决定是否需要用户确认。
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        confirm_config = config.get('security', {}).get('human_confirmation', {})
        self.enabled = confirm_config.get('enabled', True)

        # 需要确认的操作关键词 → 确认级别
        self.sensitive_operations = confirm_config.get('sensitive_operations', {}) or {
            'delete': ConfirmationLevel.DOUBLE_CONFIRM,
            'remove': ConfirmationLevel.DOUBLE_CONFIRM,
            '删除': ConfirmationLevel.DOUBLE_CONFIRM,
            '移除': ConfirmationLevel.DOUBLE_CONFIRM,
            'drop': ConfirmationLevel.DOUBLE_CONFIRM,
            'update': ConfirmationLevel.CONFIRM,
            'modify': ConfirmationLevel.CONFIRM,
            '修改': ConfirmationLevel.CONFIRM,
            '更新': ConfirmationLevel.CONFIRM,
            'create': ConfirmationLevel.INFORM,
            '创建': ConfirmationLevel.INFORM,
            '添加': ConfirmationLevel.INFORM,
        }

        # 确认回调函数（由外部注入，例如交互式CLI或Web接口）
        self._confirm_callback: Optional[Callable] = None

        self.logger.info("HumanConfirmation 初始化完成")

    def set_confirm_callback(self, callback: Callable):
        """
        设置确认回调

        Args:
            callback: 回调函数，签名 (operation: str, details: str, level: str) -> bool
        """
        self._confirm_callback = callback

    def check_operation(
        self,
        tool_name: str,
        operation_desc: str,
        parameters: Dict[str, Any],
    ) -> ConfirmationResult:
        """
        检查操作是否需要人类确认

        Args:
            tool_name: 工具名称
            operation_desc: 操作描述
            parameters: 操作参数

        Returns:
            ConfirmationResult
        """
        if not self.enabled:
            return ConfirmationResult(
                confirmed=True,
                level=ConfirmationLevel.NONE,
                operation=operation_desc,
            )

        # 判断确认级别
        level = self._determine_level(tool_name, operation_desc)

        if level == ConfirmationLevel.NONE:
            return ConfirmationResult(
                confirmed=True,
                level=level,
                operation=operation_desc,
            )

        # 构建确认信息
        details = self._build_confirmation_message(tool_name, operation_desc, parameters, level)

        self.logger.info(f"[人类确认] 操作需要确认 | 级别: {level.value} | 操作: {operation_desc}")

        # 如果设置了回调函数，使用回调获取确认
        if self._confirm_callback:
            try:
                user_confirmed = self._confirm_callback(operation_desc, details, level.value)
                return ConfirmationResult(
                    confirmed=user_confirmed,
                    level=level,
                    operation=operation_desc,
                    user_response="用户已确认" if user_confirmed else "用户已拒绝",
                )
            except Exception as e:
                self.logger.error(f"[人类确认] 回调异常: {e}")
                return ConfirmationResult(
                    confirmed=False,
                    level=level,
                    operation=operation_desc,
                    reason=f"确认流程异常: {e}",
                )

        # 没有回调函数，INFORM级别自动通过，其他级别拒绝
        if level == ConfirmationLevel.INFORM:
            self.logger.info(f"[人类确认] INFORM级别，自动通过: {operation_desc}")
            return ConfirmationResult(
                confirmed=True,
                level=level,
                operation=operation_desc,
                reason="INFORM级别，已记录但无需等待确认",
            )

        self.logger.warning(f"[人类确认] 未设置确认回调，敏感操作被拒绝: {operation_desc}")
        return ConfirmationResult(
            confirmed=False,
            level=level,
            operation=operation_desc,
            reason="敏感操作需要人类确认，但未设置确认回调",
        )

    def _determine_level(self, tool_name: str, operation_desc: str) -> ConfirmationLevel:
        """判断操作的确认级别"""
        combined = f"{tool_name} {operation_desc}".lower()

        highest_level = ConfirmationLevel.NONE
        level_order = {
            ConfirmationLevel.NONE: 0,
            ConfirmationLevel.INFORM: 1,
            ConfirmationLevel.CONFIRM: 2,
            ConfirmationLevel.DOUBLE_CONFIRM: 3,
        }

        for keyword, level in self.sensitive_operations.items():
            if isinstance(level, str):
                level = ConfirmationLevel(level)
            if keyword.lower() in combined:
                if level_order.get(level, 0) > level_order.get(highest_level, 0):
                    highest_level = level

        return highest_level

    def _build_confirmation_message(
        self,
        tool_name: str,
        operation_desc: str,
        parameters: Dict[str, Any],
        level: ConfirmationLevel,
    ) -> str:
        """构建确认消息"""
        level_icons = {
            ConfirmationLevel.INFORM: "ℹ️",
            ConfirmationLevel.CONFIRM: "⚠️",
            ConfirmationLevel.DOUBLE_CONFIRM: "🚨",
        }
        icon = level_icons.get(level, "")
        params_str = ", ".join(f"{k}={v}" for k, v in parameters.items()) if parameters else "无"

        msg = f"""{icon} 即将执行敏感操作:
  工具: {tool_name}
  操作: {operation_desc}
  参数: {params_str}
  风险级别: {level.value}"""

        if level == ConfirmationLevel.DOUBLE_CONFIRM:
            msg += "\n  ⚠️ 此操作不可逆，请务必确认！"

        return msg
