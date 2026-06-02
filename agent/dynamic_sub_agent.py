"""
动态子Agent

运行时根据任务需要动态创建，执行完毕后销毁。
角色、工具白名单、上下文均由主Agent按需配置，只返回结果摘要。

通讯模型：
    主Agent → [创建DynamicSubAgent，注入任务prompt] → 子Agent独立执行
    子Agent → [返回SubAgentResult(summary=...)] → 主Agent

执行引擎：工具调用委托给 UnifiedReActExecutor，无工具时走纯推理模式。
"""

import logging
import time
from typing import Dict, Any, List

from utils import LLMClient
from .sub_agent_base import SubAgentBase
from .task_state import SubAgentResult


class DynamicSubAgent(SubAgentBase):
    """动态子Agent

    在运行时根据SubTask配置创建，具有独立的：
    - 角色身份（system prompt）
    - 工具白名单
    - 上下文窗口
    - 执行超时和重试策略
    """

    AGENT_TYPE = "dynamic"
    AGENT_NAME = "DynamicSubAgent"

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        mcp_manager=None,
        role: str = "",
        tools: List[str] = None,
        context: str = "",
        timeout: float = 60.0,
        context_manager=None,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None,
    ):
        self.ALLOWED_TOOLS = tools or []
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            config=config,
            logger=logger,
        )
        self.mcp_manager = mcp_manager
        self.role = role
        self.injected_context = context
        self.timeout = timeout
        self.context_manager = context_manager
        self._log_tag = f"[DynAgent-{agent_id[:8]}]"

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理主Agent分派的子任务

        执行流程：
        1. 构造含角色+上下文的prompt
        2. 如果有工具白名单，尝试用LLM选择并调用工具
        3. 综合工具结果生成最终摘要
        4. 封装为SubAgentResult返回

        主Agent只能看到返回的summary字段。
        """
        start_time = time.time()
        sub_task_desc = task.get("user_query", "")
        extra_context = task.get("context", "")

        self.logger.info(
            f"{self._log_tag} 开始执行 | 角色={self.role[:40]} | "
            f"工具={self.ALLOWED_TOOLS} | 任务={sub_task_desc[:60]}"
        )

        try:
            # 合并上下文
            full_context = self._build_full_context(extra_context)

            # 判断是否需要工具调用
            if self.ALLOWED_TOOLS and self.mcp_manager:
                result_text = await self._execute_with_tools(
                    sub_task_desc, full_context
                )
            else:
                result_text = self._execute_reasoning_only(
                    sub_task_desc, full_context
                )

            # 将详细结果压缩为"总结邮件"
            summary = self._compress_to_summary(sub_task_desc, result_text)

            execution_time = time.time() - start_time
            self.logger.info(
                f"{self._log_tag} 执行完成 | 耗时={execution_time:.1f}s | "
                f"摘要长度={len(summary)}"
            )

            agent_result = SubAgentResult(
                task_id=task.get("task_id", self.agent_id),
                success=True,
                summary=summary,
                execution_time=execution_time,
            )
            return {"success": True, "response": summary, "agent_result": agent_result.to_dict()}

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"{self._log_tag} 执行失败: {e}")
            agent_result = SubAgentResult(
                task_id=task.get("task_id", self.agent_id),
                success=False,
                summary="",
                error=str(e),
                execution_time=execution_time,
            )
            return {"success": False, "response": f"子任务执行失败: {e}",
                    "agent_result": agent_result.to_dict()}

    # ================================================================
    # 内部执行逻辑
    # ================================================================

    def _build_full_context(self, extra_context: str) -> str:
        """构建完整上下文"""
        parts = []
        if self.injected_context:
            parts.append(self.injected_context)
        if extra_context:
            parts.append(extra_context)
        return "\n\n".join(parts)

    async def _execute_with_tools(self, task_desc: str, context: str) -> str:
        """使用工具完成任务（委托给 UnifiedReActExecutor）"""
        import asyncio
        from .unified_react_executor import UnifiedReActExecutor, ReActConfig

        executor = UnifiedReActExecutor(
            llm_client=self.llm_client,
            mcp_manager=self.mcp_manager,
            rag_engine=None,
            context_manager=self.context_manager,
            logger=self.logger,
        )
        cfg = ReActConfig(
            max_iterations=3,      # 子Agent 内部轮次保持较小
            temperature=0.3,
            allowed_tools=self.ALLOWED_TOOLS if self.ALLOWED_TOOLS else None,
            enable_rag=False,      # 子Agent 不开放 RAG
            system_role=self.role,
        )

        # UnifiedReActExecutor.execute 是同步的，在 async 上下文中用 run_in_executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: executor.execute(
                task_desc=task_desc,
                context=context,
                long_term_context="",
                verbose=False,
                config=cfg,
            )
        )
        return result

    def _execute_reasoning_only(self, task_desc: str, context: str) -> str:
        """纯推理模式（无工具调用）"""
        prompt = f"""{self._build_system_prompt()}

你需要完成以下任务：
{task_desc}

上下文信息：
{context[:2000]}

请直接给出分析和结论。"""

        return self.llm_client.generate(prompt=prompt, temperature=0.5).strip()

    # ================================================================
    # 辅助方法
    # ================================================================

    def _build_system_prompt(self) -> str:
        """构建子Agent的系统提示词"""
        if self.role:
            return f"你是{self.role}。请专注于你的职责范围，高效完成分配给你的任务。"
        return "你是一个专业的任务执行助手。请高效完成分配给你的任务。"

    def _compress_to_summary(self, task_desc: str, detailed_result: str) -> str:
        """将详细结果压缩为"总结邮件"

        主Agent只看到这个压缩后的摘要。
        子Agent内部的推理链、中间步骤、错误恢复细节全部被隐藏。
        """
        # 如果结果已经很短，直接返回
        if len(detailed_result) <= 500:
            return detailed_result

        prompt = f"""你是一个结果总结助手。请将以下任务执行的详细过程压缩为简洁的结论摘要。

任务描述：{task_desc}

详细执行结果：
{detailed_result[:2000]}

要求：
1. 只保留最终结论和关键数据（ID、价格、状态等）
2. 去除中间推理过程和工具调用细节
3. 控制在300字以内
4. 保持准确性

结论摘要："""

        try:
            summary = self.llm_client.generate(
                prompt=prompt, temperature=0.2, max_tokens=500
            ).strip()
            return summary
        except Exception:
            # 降级：截断
            return detailed_result[:500] + "..."
