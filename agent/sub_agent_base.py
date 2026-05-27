"""
子Agent基类

定义子Agent的通用接口和与主Agent的异步通讯协议。
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from .message_bus import (
    AsyncMessageBus, MessageHandler, Message, MessageType
)
from utils import LLMClient


class SubAgentBase(MessageHandler):
    """子Agent基类

    子Agent具有以下能力：
    1. 接收主Agent分派的任务
    2. 独立执行任务（使用自己的LLM/RAG/MCP能力）
    3. 缺少信息时向主Agent发送请求并等待回复
    4. 完成后将结果返回给主Agent

    异步通讯模式：
    - 子Agent发现缺少信息 → 向主Agent的inbox投递请求
    - 继续执行其他可做的事情（或等待回复）
    - 主Agent回复后 → 子Agent从inbox取回信息继续处理
    """

    AGENT_TYPE = "base"  # 子类需要覆盖
    AGENT_NAME = "SubAgent"  # 子类覆盖，用于日志前缀

    # 工具权限白名单：子类覆盖此列表以声明可使用的MCP工具
    # 为空列表表示不允许调用任何工具，为None表示不做限制（允许所有）
    ALLOWED_TOOLS: list = None

    def __init__(
        self,
        agent_id: str,
        bus: AsyncMessageBus,
        llm_client: LLMClient,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None,
    ):
        super().__init__(agent_id, bus, logger)
        self.llm_client = llm_client
        self.config = config or {}
        self.main_agent_id = "main_agent"  # 主Agent的固定标识
        self._log_tag = f"[{self.AGENT_NAME}]"  # 统一日志前缀

        # 子Agent自己的上下文（从主Agent获取的信息会追加到这里）
        self._context_from_main: list = []

    def filter_tools(self, all_tools: list) -> list:
        """根据权限白名单过滤工具列表

        Args:
            all_tools: MCPManager.get_available_tools() 返回的完整工具列表

        Returns:
            仅包含本子Agent有权使用的工具
        """
        if self.ALLOWED_TOOLS is None:
            return all_tools  # 不做限制
        allowed = set(self.ALLOWED_TOOLS)
        filtered = [t for t in all_tools if t.get("name") in allowed]
        self.logger.debug(
            f"{self._log_tag} 工具权限过滤: {len(all_tools)} → {len(filtered)} "
            f"(允许: {', '.join(allowed)})"
        )
        return filtered

    def is_tool_allowed(self, tool_name: str) -> bool:
        """检查单个工具是否允许使用"""
        if self.ALLOWED_TOOLS is None:
            return True
        return tool_name in self.ALLOWED_TOOLS

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理主Agent分派的任务（子类必须实现）

        Args:
            task: 任务描述，包含 user_query, context, user_id 等

        Returns:
            任务结果 dict，包含 response, success 等
        """
        raise NotImplementedError

    async def request_info_from_main(
        self, question: str, timeout: float = 30.0
    ) -> Optional[str]:
        """向主Agent请求缺失的信息

        模仿用户向主Agent消息队列添加一个问题。

        Args:
            question: 需要的信息描述
            timeout: 等待超时秒数

        Returns:
            主Agent的回复内容，超时返回None
        """
        self.logger.info(f"{self._log_tag} 向主Agent请求信息: {question[:80]}")

        reply = await self.send_request(
            receiver=self.main_agent_id,
            content={"question": question, "requester": self.agent_id},
            wait=True,
            timeout=timeout,
        )

        if reply and reply.content:
            answer = reply.content.get("answer", "")
            self.logger.info(f"{self._log_tag} 收到主Agent回复: {answer[:80]}")
            self._context_from_main.append({"question": question, "answer": answer})
            return answer
        else:
            self.logger.warning(f"{self._log_tag} 等待主Agent回复超时")
            return None

    async def check_for_updates(self) -> list:
        """主动检查inbox中是否有新消息

        子Agent完成某个动作后调用此方法，获取主Agent的异步通知。

        Returns:
            新消息列表
        """
        messages = await self.bus.drain_inbox(self.agent_id)
        for msg in messages:
            if msg.msg_type == MessageType.RESPONSE:
                answer = msg.content.get("answer", "")
                self._context_from_main.append({"msg_id": msg.msg_id, "answer": answer})
                self.logger.info(f"{self._log_tag} 从inbox获取到回复: {answer[:60]}")
        return messages

    def get_accumulated_context(self) -> str:
        """获取从主Agent积累的所有上下文信息"""
        if not self._context_from_main:
            return ""
        parts = ["[从主Agent获取的补充信息]"]
        for item in self._context_from_main:
            if "question" in item:
                parts.append(f"问: {item['question']}")
                parts.append(f"答: {item['answer']}")
            else:
                parts.append(f"通知: {item['answer']}")
        return "\n".join(parts)

    def clear_context(self):
        """清理积累的上下文（任务完成后）"""
        self._context_from_main.clear()

    async def run_task_loop(self, task_msg: Message) -> Dict[str, Any]:
        """执行完整的任务循环（接收→处理→返回结果）

        Args:
            task_msg: 任务分派消息

        Returns:
            任务执行结果
        """
        task = task_msg.content
        self.logger.info(f"{self._log_tag} 开始处理任务: {task.get('user_query', '')[:50]}")

        try:
            result = await self.handle_task(task)
            # 返回结果给主Agent
            await self.send_task_result(
                receiver=self.main_agent_id,
                content=result,
                reply_to=task_msg.msg_id,
            )
            return result
        except Exception as e:
            self.logger.error(f"{self._log_tag} 任务执行失败: {e}")
            error_result = {"success": False, "response": f"子Agent执行失败: {str(e)}"}
            await self.send_task_result(
                receiver=self.main_agent_id,
                content=error_result,
                reply_to=task_msg.msg_id,
            )
            return error_result
        finally:
            self.clear_context()
