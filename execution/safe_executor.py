"""
执行层 (Safe Executor)

职责：
1. 在受控环境中安全地执行工具操作
2. 超时控制、异常捕获
3. 执行结果标准化
4. 与安全层（权限检查、人类确认）集成
5. 执行审计日志
"""
import json
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from observability import get_tracer


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    result: str
    tool_name: str
    elapsed_ms: float = 0.0
    error: str = ""
    blocked: bool = False
    block_reason: str = ""


class SafeExecutor:
    """
    安全执行器

    在受控环境中执行工具调用，提供：
    - 超时控制
    - 异常隔离
    - 权限检查集成
    - 人类确认集成
    - 执行审计
    """

    def __init__(self, config: Dict[str, Any], mcp_manager=None, tool_registry=None,
                 permission_manager=None, human_confirmation=None,
                 logger: logging.Logger = None):
        self.config = config
        self.mcp_manager = mcp_manager
        self.tool_registry = tool_registry
        self.permission_manager = permission_manager
        self.human_confirmation = human_confirmation
        self.logger = logger or logging.getLogger(__name__)

        exec_config = config.get('execution', {})
        self.default_timeout = exec_config.get('timeout', 30)
        self.max_retries = exec_config.get('max_retries', 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=exec_config.get('max_workers', 4))

        self.logger.info("SafeExecutor 初始化完成")

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        operation_desc: str = "",
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        安全执行工具调用

        流程: 权限检查 → 人类确认 → 超时执行 → 结果标准化

        Args:
            tool_name: 工具名称
            parameters: 调用参数
            operation_desc: 操作描述（用于人类确认提示）
            timeout: 超时秒数（None使用默认值）

        Returns:
            ExecutionResult
        """
        tracer = get_tracer()
        effective_timeout = timeout or self.default_timeout

        with tracer.start_span("execution.execute_tool", {
            "exec.tool_name": tool_name,
            "exec.timeout": effective_timeout,
        }) if tracer else _noop():
            start_time = time.time()

            # ======== 1. 权限检查 ========
            if self.permission_manager:
                perm_result = self.permission_manager.check_tool_permission(tool_name)
                if not perm_result.allowed:
                    self.logger.warning(f"[执行层] 权限拒绝: {tool_name} - {perm_result.reason}")
                    return ExecutionResult(
                        success=False,
                        result="",
                        tool_name=tool_name,
                        blocked=True,
                        block_reason=perm_result.reason,
                    )

                # 频率检查
                rate_result = self.permission_manager.check_rate_limit()
                if not rate_result.allowed:
                    return ExecutionResult(
                        success=False,
                        result="",
                        tool_name=tool_name,
                        blocked=True,
                        block_reason=rate_result.reason,
                    )

            # ======== 2. 人类确认 ========
            if self.human_confirmation:
                desc = operation_desc or f"调用工具 {tool_name}"
                confirm_result = self.human_confirmation.check_operation(
                    tool_name=tool_name,
                    operation_desc=desc,
                    parameters=parameters,
                )
                if not confirm_result.confirmed:
                    self.logger.info(f"[执行层] 操作被用户拒绝: {tool_name}")
                    return ExecutionResult(
                        success=False,
                        result="",
                        tool_name=tool_name,
                        blocked=True,
                        block_reason=f"操作未通过人类确认: {confirm_result.reason}",
                    )

            # ======== 3. 执行工具 ========
            self.logger.info(f"[执行层] 执行工具: {tool_name} | 超时: {effective_timeout}s")

            for attempt in range(1, self.max_retries + 1):
                try:
                    result = self._execute_with_timeout(
                        tool_name, parameters, effective_timeout
                    )
                    elapsed = (time.time() - start_time) * 1000

                    if tracer:
                        tracer.set_span_attributes({
                            "exec.success": True,
                            "exec.elapsed_ms": elapsed,
                            "exec.attempts": attempt,
                        })

                    self.logger.info(
                        f"[执行层] 执行成功: {tool_name} | 耗时: {elapsed:.1f}ms | 尝试: {attempt}"
                    )
                    return ExecutionResult(
                        success=True,
                        result=result,
                        tool_name=tool_name,
                        elapsed_ms=elapsed,
                    )

                except FutureTimeoutError:
                    elapsed = (time.time() - start_time) * 1000
                    error_msg = f"工具执行超时 ({effective_timeout}s)"
                    self.logger.error(f"[执行层] {error_msg}: {tool_name}")
                    if attempt == self.max_retries:
                        return ExecutionResult(
                            success=False,
                            result="",
                            tool_name=tool_name,
                            elapsed_ms=elapsed,
                            error=error_msg,
                        )

                except Exception as e:
                    elapsed = (time.time() - start_time) * 1000
                    error_msg = f"工具执行异常: {str(e)}"
                    self.logger.error(f"[执行层] {error_msg}: {tool_name}")
                    if tracer:
                        tracer.record_exception(e)
                    if attempt == self.max_retries:
                        return ExecutionResult(
                            success=False,
                            result="",
                            tool_name=tool_name,
                            elapsed_ms=elapsed,
                            error=error_msg,
                        )
                    self.logger.info(f"[执行层] 重试 {attempt}/{self.max_retries}")

            # 不应到达此处
            return ExecutionResult(
                success=False, result="", tool_name=tool_name,
                error="未知执行错误",
            )

    def _execute_with_timeout(self, tool_name: str, parameters: Dict[str, Any],
                              timeout: int) -> str:
        """带超时的工具执行"""
        future = self.thread_pool.submit(self._do_execute, tool_name, parameters)
        return future.result(timeout=timeout)

    def _do_execute(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """实际执行逻辑"""
        # 先检查是否为本地工具
        if self.tool_registry:
            tool_info = self.tool_registry.get_tool(tool_name)
            if tool_info and tool_info.handler:
                result = tool_info.handler(parameters)
                return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        # MCP工具调用
        if self.mcp_manager:
            server_name = None
            if self.tool_registry:
                server_name = self.tool_registry.find_tool_server(tool_name)
            if not server_name:
                server_name = self._find_server_fallback(tool_name)

            if not server_name:
                return f"错误：找不到工具 {tool_name} 所属的服务"

            result = self.mcp_manager.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                parameters=parameters,
            )
            if isinstance(result, dict) and 'error' in result:
                return f"工具调用失败: {result['error']}"
            return f"工具执行成功: {json.dumps(result, ensure_ascii=False)}"

        return f"错误：无法执行工具 {tool_name}（MCP管理器未初始化）"

    def _find_server_fallback(self, tool_name: str) -> Optional[str]:
        """回退方式查找工具所属服务"""
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_available_tools(use_cache=True):
            if tool['name'] == tool_name:
                return tool.get('server')
        return None

    def shutdown(self):
        """关闭线程池"""
        self.thread_pool.shutdown(wait=False)


class _noop:
    """空上下文管理器"""
    def __enter__(self): return self
    def __exit__(self, *a): pass
