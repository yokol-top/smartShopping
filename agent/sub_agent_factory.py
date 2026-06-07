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
import os
import time
import uuid
from typing import Dict, Any, List, Optional

import yaml as _yaml

from utils import LLMClient
from utils.logger import get_trace_id, set_trace_id
from .circuit_breaker import CircuitBreaker
from .dynamic_sub_agent import DynamicSubAgent
from .exceptions import NeedUserInputException
from .task_state import SubTask, SubAgentResult


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
        task_planner=None,
    ):
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.context_manager = context_manager
        self.task_planner = task_planner

        # 熔断器
        cb_config = self.config.get('orchestrator', {}).get('circuit_breaker', {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get('failure_threshold', 3),
            cooldown_seconds=cb_config.get('cooldown_seconds', 60.0),
            logger=self.logger,
        )
        self._typed_circuit_breakers: Dict[str, CircuitBreaker] = {}

        # 加载 Agent 模板配置
        templates_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'agent_templates.yaml'
        )
        self._agent_templates: dict = {}
        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                data = _yaml.safe_load(f)
                self._agent_templates = data.get('templates', {})
            self.logger.info(f"[SubAgentFactory] 加载 Agent 模板: {list(self._agent_templates.keys())}")
        except Exception as e:
            self.logger.warning(f"[SubAgentFactory] 模板配置加载失败（降级动态生成）: {e}")

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

    def _match_template(self, tools: List[str]) -> Optional[dict]:
        """根据工具列表匹配最合适的 Agent 模板。

        匹配规则：sub_task 的工具列表是模板 allowed_tools 的子集时视为匹配，
        优先选择 allowed_tools 数量最少（最精确）的模板。
        """
        if not tools or not self._agent_templates:
            return None

        tool_set = set(tools)
        best = None
        best_size = float('inf')

        for name, tmpl in self._agent_templates.items():
            tmpl_tools = set(tmpl.get('allowed_tools', []))
            if tool_set and tool_set.issubset(tmpl_tools):
                if len(tmpl_tools) < best_size:
                    best = tmpl
                    best_size = len(tmpl_tools)

        return best

    def create_agent(self, sub_task: SubTask) -> DynamicSubAgent:
        """根据SubTask配置创建动态子Agent

        Args:
            sub_task: 子任务定义（包含角色、工具、上下文等配置）

        Returns:
            配置好的DynamicSubAgent实例
        """
        agent_id = f"dyn_{sub_task.id}_{str(uuid.uuid4())[:4]}"

        # 优先匹配固定模板（角色和工具更可预测）
        template = self._match_template(sub_task.agent_tools)
        if template:
            role = template.get('role', sub_task.agent_role)
            tools = template.get('allowed_tools', sub_task.agent_tools)
            max_iter = template.get('max_iterations', 5)
            self.logger.info(
                f"[SubAgentFactory] 使用模板 | agent={agent_id} | "
                f"tools={tools} | checkpoint={template.get('checkpoint_before_execute', False)}"
            )
        else:
            role = sub_task.agent_role
            tools = sub_task.agent_tools
            max_iter = 5   # 默认值

        agent = DynamicSubAgent(
            agent_id=agent_id,
            llm_client=self.llm_client,
            mcp_manager=self.mcp_manager,
            role=role,
            tools=tools,
            context=sub_task.agent_context,
            timeout=sub_task.timeout,
            context_manager=self.context_manager,
            config=self.config,
            logger=self.logger,
            task_planner=self.task_planner,
        )
        # 将 max_iterations 存到 agent 以便 _execute_with_tools 使用
        agent._max_iterations = max_iter

        self.logger.info(
            f"[SubAgentFactory] 创建子Agent: {agent_id} | "
            f"角色={role[:30]} | 工具={tools}"
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
            trace_id = get_trace_id()
            result = self._run_async(self._execute_async(sub_task, trace_id))
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
            trace_id = get_trace_id()
            results = self._run_async(self._execute_parallel_async(sub_tasks, trace_id))
            return results
        except Exception as e:
            self.logger.error(f"[SubAgentFactory] 并行执行异常: {e}")
            return {
                st.id: SubAgentResult(
                    task_id=st.id, success=False, summary="", error=str(e)
                )
                for st in sub_tasks
            }

    @staticmethod
    def _capture_otel_context():
        """捕获当前 OpenTelemetry context（用于线程传播）"""
        try:
            from opentelemetry import context as otel_ctx
            return otel_ctx.get_current()
        except ImportError:
            return None

    @staticmethod
    def _attach_otel_context(ctx):
        """在新线程/协程中附加 OTel context，返回 token（用于 detach）"""
        if ctx is None:
            return None
        try:
            from opentelemetry import context as otel_ctx
            return otel_ctx.attach(ctx)
        except Exception:
            return None

    @staticmethod
    def _detach_otel_context(token):
        """释放 OTel context token"""
        if token is None:
            return
        try:
            from opentelemetry import context as otel_ctx
            otel_ctx.detach(token)
        except Exception:
            pass

    async def _execute_async(self, sub_task: SubTask, trace_id: str = None) -> SubAgentResult:
        """异步执行单个子任务（直接调用 handle_task，不经过 MessageBus）"""
        # 显式恢复 trace_id：asyncio.run() 创建新事件循环时 ContextVar 不一定继承
        if trace_id:
            set_trace_id(trace_id)
        # 捕获当前线程的 OTel context（在 create_agent 之前）
        otel_ctx = self._capture_otel_context()
        agent = self.create_agent(sub_task)
        task_payload = {
            "task_id": sub_task.id,
            "user_query": sub_task.description,
            "context": sub_task.agent_context,
        }

        otel_token = self._attach_otel_context(otel_ctx)
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

        except NeedUserInputException as e:
            # 转换为特殊 SubAgentResult，Orchestrator._phase_execute() 会识别
            self.logger.info(
                f"[SubAgentFactory] 子任务 {sub_task.id} 需要用户输入: {e.question[:60]}"
            )
            return SubAgentResult(
                task_id=sub_task.id,
                success=False,
                summary="",
                error=f"__NEED_INPUT__:{e.question}",
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

        finally:
            self._detach_otel_context(otel_token)

    async def _execute_parallel_async(
        self, sub_tasks: List[SubTask], trace_id: str = None
    ) -> Dict[str, SubAgentResult]:
        """异步并行执行多个子任务"""

        async def _run_one(st: SubTask) -> tuple:
            result = await self._execute_async(st, trace_id)
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
