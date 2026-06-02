"""
子Agent工厂

负责根据SubTask的配置动态创建子Agent实例。
子Agent创建后独立运行，完成后返回结果即销毁。

职责：
1. 根据SubTask的agent_role/agent_tools/agent_context创建DynamicSubAgent
2. 管理子Agent的生命周期（创建→运行→销毁）
3. 提供同步/异步执行接口
4. 实现熔断器模式（连续失败时停止委派）

企业级失败处理策略：
- 重试（Retry）：子Agent内部自动重试工具调用
- 降级（Fallback）：子Agent失败后由主Agent接管
- 熔断（Circuit Breaker）：连续失败N次后暂停委派
- 超时（Timeout）：严格的执行超时控制
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List

from utils import LLMClient
from .dynamic_sub_agent import DynamicSubAgent
from .task_state import SubTask, SubAgentResult


class CircuitBreaker:
    """熔断器

    连续失败达到阈值后进入OPEN状态，暂停委派。
    经过冷却期后进入HALF_OPEN状态，允许少量试探。
    试探成功则恢复CLOSED状态。

    参考企业级方案：Netflix Hystrix / resilience4j
    """

    CLOSED = "closed"         # 正常工作
    OPEN = "open"             # 熔断中（拒绝委派）
    HALF_OPEN = "half_open"   # 试探中

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        logger: logging.Logger = None,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger or logging.getLogger(__name__)

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count_in_half_open = 0

    @property
    def state(self) -> str:
        # 自动从OPEN转HALF_OPEN（冷却期过后）
        if self._state == self.OPEN:
            if time.time() - self._last_failure_time > self.cooldown_seconds:
                self._state = self.HALF_OPEN
                self._success_count_in_half_open = 0
                self.logger.info("[CircuitBreaker] OPEN → HALF_OPEN（冷却期结束）")
        return self._state

    def allow_request(self) -> bool:
        """是否允许委派"""
        s = self.state
        if s == self.CLOSED:
            return True
        if s == self.HALF_OPEN:
            return True  # 允许试探
        return False  # OPEN状态拒绝

    def record_success(self):
        """记录成功"""
        if self._state == self.HALF_OPEN:
            self._success_count_in_half_open += 1
            if self._success_count_in_half_open >= 2:
                self._state = self.CLOSED
                self._failure_count = 0
                self.logger.info("[CircuitBreaker] HALF_OPEN → CLOSED（试探成功）")
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """记录失败"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            self.logger.warning("[CircuitBreaker] HALF_OPEN → OPEN（试探失败）")
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            self.logger.warning(
                f"[CircuitBreaker] CLOSED → OPEN（连续失败{self._failure_count}次）"
            )


class SubAgentFactory:
    """子Agent工厂

    核心职责：
    1. 从SubTask配置动态创建DynamicSubAgent
    2. 运行子Agent并收集结果
    3. 通过熔断器控制委派风险
    4. 支持并行执行多个子Agent
    """

    def __init__(
        self,
        llm_client: LLMClient,
        mcp_manager=None,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None,
        context_manager=None,
    ):
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.context_manager = context_manager

        # 熔断器
        cb_config = self.config.get('orchestrator', {}).get('circuit_breaker', {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get('failure_threshold', 3),
            cooldown_seconds=cb_config.get('cooldown_seconds', 60.0),
            logger=self.logger,
        )

        self.logger.info("[SubAgentFactory] 初始化完成")

    # ================================================================
    # 熔断器管理
    # ================================================================

    def _get_circuit_breaker(self, tools: List[str]) -> CircuitBreaker:
        """根据工具列表获取对应的熔断器（按类别分组）"""
        if not tools:
            return self.circuit_breaker

        # 判断工具类别
        tools_lower = [t.lower() for t in tools]
        if any('order' in t for t in tools_lower):
            category = 'order_ops'
        elif any('product' in t or 'search' in t for t in tools_lower):
            category = 'product_search'
        elif any('user' in t or 'address' in t or 'card' in t for t in tools_lower):
            category = 'user_ops'
        else:
            category = 'general'

        if category not in self._typed_circuit_breakers:
            cb_config = self.config.get('orchestrator', {}).get('circuit_breaker', {})
            self._typed_circuit_breakers[category] = CircuitBreaker(
                failure_threshold=cb_config.get('failure_threshold', 3),
                cooldown_seconds=cb_config.get('cooldown_seconds', 60.0),
                logger=self.logger,
            )
            self.logger.info(f"[SubAgentFactory] 创建熔断器: category={category}")

        return self._typed_circuit_breakers[category]

    # ================================================================
    # 子Agent创建
    # ================================================================

    def create_agent(self, sub_task: SubTask) -> DynamicSubAgent:
        """根据SubTask配置创建动态子Agent

        Args:
            sub_task: 子任务定义（包含角色、工具、上下文等配置）

        Returns:
            配置好的DynamicSubAgent实例
        """
        agent_id = f"dyn_{sub_task.id}_{str(uuid.uuid4())[:4]}"

        agent = DynamicSubAgent(
            agent_id=agent_id,
            llm_client=self.llm_client,
            mcp_manager=self.mcp_manager,
            role=sub_task.agent_role,
            tools=sub_task.agent_tools,
            context=sub_task.agent_context,
            timeout=sub_task.timeout,
            context_manager=self.context_manager,
            config=self.config,
            logger=self.logger,
        )

        self.logger.info(
            f"[SubAgentFactory] 创建子Agent: {agent_id} | "
            f"角色={sub_task.agent_role[:30]} | "
            f"工具={sub_task.agent_tools}"
        )
        return agent

    # ================================================================
    # 子Agent执行
    # ================================================================

    def execute_subtask(self, sub_task: SubTask) -> SubAgentResult:
        """同步执行单个子任务（阻塞直到完成或超时）

        Args:
            sub_task: 子任务定义

        Returns:
            SubAgentResult（主Agent的"总结邮件"）
        """
        # 熔断检查（类别熔断器 + 全局熔断器双重检查）
        cb = self._get_circuit_breaker(sub_task.agent_tools)
        if not cb.allow_request():
            self.logger.warning(
                f"[SubAgentFactory] 类别熔断器OPEN，拒绝委派子任务: {sub_task.id}"
            )
            return SubAgentResult(
                task_id=sub_task.id,
                success=False,
                summary="",
                error="子Agent服务暂时不可用（熔断中），请稍后重试或由主Agent直接处理",
            )
        if not self.circuit_breaker.allow_request():
            self.logger.warning(
                f"[SubAgentFactory] 全局熔断器OPEN，拒绝委派子任务: {sub_task.id}"
            )
            return SubAgentResult(
                task_id=sub_task.id,
                success=False,
                summary="",
                error="子Agent服务暂时不可用（熔断中），请稍后重试或由主Agent直接处理",
            )

        try:
            result = self._run_async(self._execute_async(sub_task))
            if result.success:
                cb.record_success()
                self.circuit_breaker.record_success()   # 同时更新全局熔断器
            else:
                cb.record_failure()
                self.circuit_breaker.record_failure()
            return result
        except Exception as e:
            cb.record_failure()
            self.circuit_breaker.record_failure()
            self.logger.error(f"[SubAgentFactory] 子任务执行异常: {e}")
            return SubAgentResult(
                task_id=sub_task.id,
                success=False,
                summary="",
                error=str(e),
            )

    def execute_subtasks_parallel(
        self, sub_tasks: List[SubTask]
    ) -> Dict[str, SubAgentResult]:
        """并行执行多个子任务

        Args:
            sub_tasks: 子任务列表（这些任务之间无依赖关系）

        Returns:
            {sub_task_id: SubAgentResult} 映射
        """
        if not sub_tasks:
            return {}

        if len(sub_tasks) == 1:
            result = self.execute_subtask(sub_tasks[0])
            return {sub_tasks[0].id: result}

        self.logger.info(
            f"[SubAgentFactory] 并行执行 {len(sub_tasks)} 个子任务"
        )

        try:
            results = self._run_async(self._execute_parallel_async(sub_tasks))
            return results
        except Exception as e:
            self.logger.error(f"[SubAgentFactory] 并行执行异常: {e}")
            return {
                st.id: SubAgentResult(
                    task_id=st.id, success=False, summary="", error=str(e)
                )
                for st in sub_tasks
            }

    async def _execute_async(self, sub_task: SubTask) -> SubAgentResult:
        """异步执行单个子任务（直接调用 handle_task，不经过 MessageBus）"""
        agent = self.create_agent(sub_task)
        task_payload = {
            "task_id": sub_task.id,
            "user_query": sub_task.description,
            "context": sub_task.agent_context,
        }

        try:
            raw_result = await asyncio.wait_for(
                agent.handle_task(task_payload),
                timeout=sub_task.timeout,
            )

            agent_result_dict = raw_result.get("agent_result")
            if agent_result_dict:
                return SubAgentResult.from_dict(agent_result_dict)

            return SubAgentResult(
                task_id=sub_task.id,
                success=raw_result.get("success", False),
                summary=raw_result.get("response", ""),
            )

        except asyncio.TimeoutError:
            self.logger.error(
                f"[SubAgentFactory] 子任务 {sub_task.id} 执行超时 ({sub_task.timeout}s)"
            )
            return SubAgentResult(
                task_id=sub_task.id,
                success=False,
                summary="",
                error=f"执行超时（{sub_task.timeout}秒）",
            )

    async def _execute_parallel_async(
        self, sub_tasks: List[SubTask]
    ) -> Dict[str, SubAgentResult]:
        """异步并行执行多个子任务"""

        async def _run_one(st: SubTask) -> tuple:
            result = await self._execute_async(st)
            return st.id, result

        tasks = [_run_one(st) for st in sub_tasks]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for i, item in enumerate(completed):
            st = sub_tasks[i]
            if isinstance(item, Exception):
                self.circuit_breaker.record_failure()
                results[st.id] = SubAgentResult(
                    task_id=st.id,
                    success=False,
                    summary="",
                    error=str(item),
                )
            else:
                task_id, result = item
                if result.success:
                    self.circuit_breaker.record_success()
                else:
                    self.circuit_breaker.record_failure()
                results[task_id] = result

        return results

    # ================================================================
    # 辅助
    # ================================================================

    def _run_async(self, coro):
        """同步环境中运行异步协程"""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=120)
        except RuntimeError:
            return asyncio.run(coro)

    @property
    def circuit_breaker_state(self) -> str:
        """当前熔断器状态"""
        return self.circuit_breaker.state

    @property
    def typed_circuit_breaker_states(self) -> Dict[str, str]:
        """各类别熔断器的状态"""
        return {cat: cb.state for cat, cb in self._typed_circuit_breakers.items()}
