"""
权限控制模块 (Permission Manager)

职责：
1. 用户角色与权限管理
2. 工具调用权限检查
3. 操作频率限制
"""
import time
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PermissionCheckResult:
    """权限检查结果"""
    allowed: bool
    reason: str = ""
    required_role: str = ""
    current_role: str = ""


class PermissionManager:
    """
    权限管理器

    基于角色的访问控制（RBAC）+ 频率限制
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        perm_config = config.get('security', {}).get('permissions', {})
        self.enabled = perm_config.get('enabled', True)

        # 角色定义及其允许的操作
        self.role_permissions: Dict[str, Set[str]] = {
            'admin': {'read', 'write', 'delete', 'execute', 'manage'},
            'user': {'read', 'write', 'execute'},
            'viewer': {'read'},
        }

        # 从配置加载自定义角色权限
        custom_roles = perm_config.get('roles', {})
        for role, perms in custom_roles.items():
            self.role_permissions[role] = set(perms)

        # 工具 → 所需权限映射
        self.tool_permissions: Dict[str, str] = perm_config.get('tool_permissions', {}) or {}

        # 频率限制配置
        rate_config = perm_config.get('rate_limit', {})
        self.rate_limit_enabled = rate_config.get('enabled', True)
        self.max_calls_per_minute = rate_config.get('max_calls_per_minute', 30)
        self._call_timestamps: Dict[str, List[float]] = defaultdict(list)

        # 当前用户角色（默认为user）
        self.current_role = perm_config.get('default_role', 'user')

        self.logger.info(f"PermissionManager 初始化完成 | 当前角色: {self.current_role}")

    def set_role(self, role: str):
        """设置当前用户角色"""
        if role not in self.role_permissions:
            self.logger.warning(f"未知角色: {role}，保持当前角色: {self.current_role}")
            return
        self.current_role = role
        self.logger.info(f"用户角色已切换为: {role}")

    def check_tool_permission(self, tool_name: str) -> PermissionCheckResult:
        """
        检查当前用户是否有权使用指定工具

        Args:
            tool_name: 工具名称

        Returns:
            PermissionCheckResult
        """
        if not self.enabled:
            return PermissionCheckResult(allowed=True)

        # 获取工具所需权限
        required_perm = self.tool_permissions.get(tool_name, 'execute')

        # 检查当前角色是否拥有该权限
        current_perms = self.role_permissions.get(self.current_role, set())
        if required_perm not in current_perms:
            self.logger.warning(
                f"[权限] 拒绝: 角色 {self.current_role} 无权执行 {tool_name} (需要: {required_perm})"
            )
            return PermissionCheckResult(
                allowed=False,
                reason=f"权限不足: 角色 '{self.current_role}' 没有 '{required_perm}' 权限",
                required_role=required_perm,
                current_role=self.current_role,
            )

        return PermissionCheckResult(allowed=True, current_role=self.current_role)

    def check_rate_limit(self, user_id: str = "default") -> PermissionCheckResult:
        """
        检查频率限制

        Args:
            user_id: 用户标识

        Returns:
            PermissionCheckResult
        """
        if not self.enabled or not self.rate_limit_enabled:
            return PermissionCheckResult(allowed=True)

        now = time.time()
        window_start = now - 60  # 1分钟窗口

        # 清理过期记录
        self._call_timestamps[user_id] = [
            ts for ts in self._call_timestamps[user_id] if ts > window_start
        ]

        if len(self._call_timestamps[user_id]) >= self.max_calls_per_minute:
            self.logger.warning(f"[频率限制] 用户 {user_id} 超过频率限制: {self.max_calls_per_minute}/min")
            return PermissionCheckResult(
                allowed=False,
                reason=f"请求频率超限（最大: {self.max_calls_per_minute}次/分钟）",
            )

        self._call_timestamps[user_id].append(now)
        return PermissionCheckResult(allowed=True)
