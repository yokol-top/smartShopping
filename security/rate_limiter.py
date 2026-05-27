"""
工具调用频率限制器（Rate Limiter）

防止 Agent 过度调用工具导致：
- 预算花超（LLM 驱动的循环调用）
- 外部服务（MCP）压力过大
- 单个会话垄断资源

实现：
- 滑动窗口算法（精确到秒级）
- 支持 per-tool 独立限额 + 全局限额
- 支持自定义高危工具的更严格限制
- 调用计数与拒绝记录（可观测性）
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class RateLimitResult:
    """频率检查结果"""
    allowed: bool
    reason: str = ""
    remaining: int = 0        # 剩余可用次数
    retry_after: float = 0.0  # 建议等待秒数


class RateLimiter:
    """
    工具调用频率限制器

    配置示例（settings.yaml）：
        security:
          rate_limit:
            enabled: true
            global_max_per_minute: 30     # 全局每分钟最大调用次数
            per_tool_max_per_minute: 10   # 单工具每分钟最大调用次数
            per_session_max_total: 200    # 单会话最大总调用次数
            tool_overrides:               # 特定工具的自定义限额
              delete_order: 3             # 删除订单每分钟最多3次
              create_order: 5
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        rate_config = (config or {}).get('security', {}).get('rate_limit', {})
        self.enabled = rate_config.get('enabled', True)

        # 限额配置
        self.global_max_per_minute = rate_config.get('global_max_per_minute', 30)
        self.per_tool_max_per_minute = rate_config.get('per_tool_max_per_minute', 10)
        self.per_session_max_total = rate_config.get('per_session_max_total', 200)

        # 特定工具覆盖限额 {tool_name: max_per_minute}
        self.tool_overrides: Dict[str, int] = rate_config.get('tool_overrides', {})

        # 滑动窗口记录：{tool_name: [timestamp, ...]}
        self._tool_calls: Dict[str, List[float]] = defaultdict(list)
        # 全局调用记录
        self._global_calls: List[float] = []
        # 会话总调用计数
        self._session_total: int = 0
        # 拒绝计数（可观测性）
        self._rejected_count: int = 0

        if self.enabled:
            self.logger.info(
                f"RateLimiter 初始化 | 全局: {self.global_max_per_minute}/min | "
                f"单工具: {self.per_tool_max_per_minute}/min | "
                f"会话总量: {self.per_session_max_total} | "
                f"特殊限额: {self.tool_overrides or '无'}"
            )
        else:
            self.logger.info("RateLimiter 已禁用")

    def check(self, tool_name: str) -> RateLimitResult:
        """
        检查工具调用是否被允许

        Args:
            tool_name: 工具名称

        Returns:
            RateLimitResult
        """
        if not self.enabled:
            return RateLimitResult(allowed=True, remaining=999)

        now = time.time()
        window_start = now - 60.0  # 1分钟滑动窗口

        # 1. 会话总量检查
        if self._session_total >= self.per_session_max_total:
            self._rejected_count += 1
            return RateLimitResult(
                allowed=False,
                reason=f"会话总调用次数已达上限({self.per_session_max_total}次)",
                remaining=0,
            )

        # 2. 全局频率检查
        self._cleanup_window(self._global_calls, window_start)
        if len(self._global_calls) >= self.global_max_per_minute:
            self._rejected_count += 1
            retry_after = self._global_calls[0] + 60.0 - now
            return RateLimitResult(
                allowed=False,
                reason=f"全局调用频率已达上限({self.global_max_per_minute}次/分钟)",
                remaining=0,
                retry_after=max(0, retry_after),
            )

        # 3. 单工具频率检查
        tool_calls = self._tool_calls[tool_name]
        self._cleanup_window(tool_calls, window_start)
        tool_limit = self.tool_overrides.get(tool_name, self.per_tool_max_per_minute)
        if len(tool_calls) >= tool_limit:
            self._rejected_count += 1
            retry_after = tool_calls[0] + 60.0 - now
            return RateLimitResult(
                allowed=False,
                reason=f"工具 {tool_name} 调用频率已达上限({tool_limit}次/分钟)",
                remaining=0,
                retry_after=max(0, retry_after),
            )

        remaining = min(
            self.global_max_per_minute - len(self._global_calls),
            tool_limit - len(tool_calls),
            self.per_session_max_total - self._session_total,
        )
        return RateLimitResult(allowed=True, remaining=remaining)

    def is_allowed(self, tool_name: str) -> bool:
        """快捷方法：是否允许调用"""
        return self.check(tool_name).allowed

    def record_call(self, tool_name: str):
        """记录一次工具调用"""
        now = time.time()
        self._tool_calls[tool_name].append(now)
        self._global_calls.append(now)
        self._session_total += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取频率限制统计信息"""
        now = time.time()
        window_start = now - 60.0

        # 清理过期记录
        self._cleanup_window(self._global_calls, window_start)

        per_tool = {}
        for name, calls in self._tool_calls.items():
            self._cleanup_window(calls, window_start)
            if calls:
                limit = self.tool_overrides.get(name, self.per_tool_max_per_minute)
                per_tool[name] = {
                    "calls_last_minute": len(calls),
                    "limit_per_minute": limit,
                    "remaining": max(0, limit - len(calls)),
                }

        return {
            "enabled": self.enabled,
            "global_calls_last_minute": len(self._global_calls),
            "global_limit_per_minute": self.global_max_per_minute,
            "session_total_calls": self._session_total,
            "session_limit": self.per_session_max_total,
            "rejected_count": self._rejected_count,
            "per_tool": per_tool,
        }

    def reset(self):
        """重置所有计数（新会话时调用）"""
        self._tool_calls.clear()
        self._global_calls.clear()
        self._session_total = 0
        self._rejected_count = 0

    @staticmethod
    def _cleanup_window(calls: List[float], window_start: float):
        """清理滑动窗口中过期的记录"""
        while calls and calls[0] < window_start:
            calls.pop(0)
