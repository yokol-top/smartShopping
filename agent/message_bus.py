"""
异步消息总线：主Agent与子Agent之间的通讯机制

通讯模型：
- 每个Agent（主/子）都有自己的消息队列（inbox）
- 子Agent缺少信息时，向主Agent的inbox发送一个请求消息
- 主Agent处理后，向子Agent的inbox发送回复消息
- 子Agent完成某个动作后主动检查inbox获取回复
- 全程异步，互不阻塞
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"           # 子Agent向主Agent请求信息
    RESPONSE = "response"         # 主Agent回复子Agent
    TASK_DISPATCH = "task_dispatch"  # 主Agent分派任务给子Agent
    TASK_RESULT = "task_result"   # 子Agent向主Agent返回任务结果
    NOTIFY = "notify"             # 通知类消息（无需回复）


class MessageStatus(Enum):
    """消息状态"""
    PENDING = "pending"           # 待处理
    PROCESSING = "processing"     # 处理中
    COMPLETED = "completed"       # 已完成
    EXPIRED = "expired"           # 已过期


@dataclass
class Message:
    """消息体"""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    msg_type: MessageType = MessageType.REQUEST
    sender: str = ""              # 发送方Agent标识
    receiver: str = ""            # 接收方Agent标识
    reply_to: str = ""            # 关联的请求消息ID（用于匹配回复）
    content: Dict[str, Any] = field(default_factory=dict)
    status: MessageStatus = MessageStatus.PENDING
    created_at: float = field(default_factory=time.time)
    ttl: float = 60.0             # 消息有效期（秒）

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "reply_to": self.reply_to,
            "content": self.content,
            "status": self.status.value,
            "created_at": self.created_at,
        }


class AsyncMessageBus:
    """异步消息总线

    每个Agent注册后获得一个专属inbox（asyncio.Queue）。
    发送消息 = 投递到目标Agent的inbox。
    接收消息 = 从自己的inbox取出。

    支持两种使用模式：
    1. 主动轮询：agent完成一个动作后调用 check_inbox() 获取新消息
    2. 等待回复：agent发送请求后调用 wait_for_reply() 阻塞等待特定回复
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        # 注册的Agent ID集合（Queue在prepare()时创建，确保属于当前事件循环）
        self._registered_agents: set = set()
        self._inboxes: Dict[str, asyncio.Queue] = {}
        # 已完成的消息存档（用于关联查询）
        self._archive: Dict[str, Message] = {}
        # 回复通知：msg_id -> Event（用于 wait_for_reply）
        self._reply_events: Dict[str, asyncio.Event] = {}
        self._reply_messages: Dict[str, Message] = {}

    def register_agent(self, agent_id: str):
        """注册一个Agent（仅记录ID，Queue在prepare()时创建）"""
        if agent_id not in self._registered_agents:
            self._registered_agents.add(agent_id)
            self.logger.info(f"[MessageBus] 注册Agent: {agent_id}")

    def prepare(self):
        """在当前事件循环中创建/重建Queue

        必须在 asyncio.run() 或 async 函数内部调用，确保 Queue 绑定到当前事件循环。
        每次 dispatch 前调用一次即可。
        """
        self._inboxes = {aid: asyncio.Queue() for aid in self._registered_agents}
        self._reply_events.clear()
        self._reply_messages.clear()
        self.logger.debug(f"[MessageBus] 重建 {len(self._inboxes)} 个 inbox Queue")

    def _ensure_inbox(self, agent_id: str) -> asyncio.Queue:
        """确保 agent 的 inbox 存在（懒创建兼容）"""
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = asyncio.Queue()
        return self._inboxes[agent_id]

    async def send(self, message: Message):
        """发送消息到目标Agent的inbox"""
        receiver = message.receiver
        if receiver not in self._registered_agents:
            self.logger.warning(f"[MessageBus] 目标Agent未注册: {receiver}")
            return False

        inbox = self._ensure_inbox(receiver)
        await inbox.put(message)
        self._archive[message.msg_id] = message
        self.logger.info(
            f"[MessageBus] {message.sender} → {message.receiver} | "
            f"type={message.msg_type.value} | id={message.msg_id}"
        )

        # 如果是回复消息，通知等待方
        if message.msg_type == MessageType.RESPONSE and message.reply_to:
            event = self._reply_events.get(message.reply_to)
            if event:
                self._reply_messages[message.reply_to] = message
                event.set()

        return True

    async def check_inbox(self, agent_id: str, timeout: float = 0.1) -> Optional[Message]:
        """非阻塞地检查inbox是否有新消息

        Args:
            agent_id: Agent标识
            timeout: 最长等待秒数（默认0.1s，接近非阻塞）

        Returns:
            消息对象，如果没有新消息返回None
        """
        inbox = self._ensure_inbox(agent_id)

        try:
            message = await asyncio.wait_for(inbox.get(), timeout=timeout)
            if message.is_expired:
                self.logger.info(f"[MessageBus] 消息已过期丢弃: {message.msg_id}")
                return None
            return message
        except asyncio.TimeoutError:
            return None

    async def drain_inbox(self, agent_id: str) -> list:
        """一次性取出inbox中所有消息"""
        inbox = self._inboxes.get(agent_id)
        if not inbox:
            return []
        if inbox.empty():
            return []
        messages = []
        while not inbox.empty():
            try:
                msg = inbox.get_nowait()
                if not msg.is_expired:
                    messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    async def send_and_wait(
        self, message: Message, timeout: float = 30.0
    ) -> Optional[Message]:
        """发送请求并等待回复（阻塞直到收到回复或超时）

        适用于子Agent向主Agent请求必要信息的场景。

        Args:
            message: 请求消息
            timeout: 最长等待秒数

        Returns:
            回复消息，超时返回None
        """
        # 创建等待事件
        event = asyncio.Event()
        self._reply_events[message.msg_id] = event

        # 发送请求
        await self.send(message)

        # 等待回复
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            reply = self._reply_messages.pop(message.msg_id, None)
            return reply
        except asyncio.TimeoutError:
            self.logger.warning(
                f"[MessageBus] 等待回复超时: {message.msg_id} ({timeout}s)"
            )
            return None
        finally:
            self._reply_events.pop(message.msg_id, None)

    def get_inbox_size(self, agent_id: str) -> int:
        """查看inbox中的待处理消息数"""
        inbox = self._inboxes.get(agent_id)
        return inbox.qsize() if inbox else 0

    @property
    def registered_agents(self) -> set:
        return set(self._registered_agents)


class MessageHandler:
    """消息处理器基类

    子Agent和主Agent都可以继承此类来处理收到的消息。
    """

    def __init__(self, agent_id: str, bus: AsyncMessageBus, logger: logging.Logger = None):
        self.agent_id = agent_id
        self.bus = bus
        self.logger = logger or logging.getLogger(__name__)
        self.bus.register_agent(agent_id)

    async def send_request(
        self, receiver: str, content: Dict[str, Any], wait: bool = False, timeout: float = 30.0
    ) -> Optional[Message]:
        """发送请求消息

        Args:
            receiver: 目标Agent
            content: 消息内容
            wait: 是否等待回复
            timeout: 等待超时

        Returns:
            如果wait=True返回回复消息，否则返回None
        """
        msg = Message(
            msg_type=MessageType.REQUEST,
            sender=self.agent_id,
            receiver=receiver,
            content=content,
        )
        if wait:
            return await self.bus.send_and_wait(msg, timeout=timeout)
        else:
            await self.bus.send(msg)
            return None

    async def send_reply(self, original_msg: Message, content: Dict[str, Any]):
        """回复一个请求消息"""
        reply = Message(
            msg_type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=original_msg.sender,
            reply_to=original_msg.msg_id,
            content=content,
        )
        await self.bus.send(reply)

    async def send_task(self, receiver: str, content: Dict[str, Any]):
        """分派任务给子Agent"""
        msg = Message(
            msg_type=MessageType.TASK_DISPATCH,
            sender=self.agent_id,
            receiver=receiver,
            content=content,
        )
        await self.bus.send(msg)

    async def send_task_result(self, receiver: str, content: Dict[str, Any], reply_to: str = ""):
        """向主Agent返回任务执行结果"""
        msg = Message(
            msg_type=MessageType.TASK_RESULT,
            sender=self.agent_id,
            receiver=receiver,
            reply_to=reply_to,
            content=content,
        )
        await self.bus.send(msg)
