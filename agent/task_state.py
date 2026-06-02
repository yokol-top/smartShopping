"""
任务状态模型

维护一次复杂任务的完整生命周期状态（理解→规划→执行→整合→交付），
由主Agent持有，支持序列化/反序列化以实现会话恢复。

设计原则：
- 主Agent是状态的唯一所有者
- 子Agent只通过SubAgentResult向主Agent汇报"总结邮件"
- TaskState可序列化为JSON，由SessionManager持久化
- 会话恢复时，上一次未完成的TaskState也一并恢复
"""

import time
import uuid
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


# ================================================================
# 枚举
# ================================================================

class TaskPhase(str, Enum):
    """任务生命周期阶段"""
    UNDERSTAND = "understand"   # Phase 1: 解析意图、初步探索、判断复杂度
    PLAN = "plan"               # Phase 2: 分解子任务、决定委派、确定执行顺序
    EXECUTE = "execute"         # Phase 3: 执行子任务（自己做 + 委派子Agent）
    INTEGRATE = "integrate"     # Phase 4: 整合结果、验证质量、必要时重试
    DELIVER = "deliver"         # Phase 5: 总结交付、运行验证
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 失败


class SubTaskStatus(str, Enum):
    """子任务执行状态"""
    PENDING = "pending"         # 等待执行
    RUNNING = "running"         # 执行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 执行失败
    RETRYING = "retrying"       # 重试中
    CANCELLED = "cancelled"     # 已取消（依赖的子任务失败导致）


class DelegationReason(str, Enum):
    """委派给子Agent的原因"""
    PARALLELIZABLE = "parallelizable"       # 可并行加速
    CONTEXT_OVERFLOW = "context_overflow"    # 上下文快满/中间结果大
    OFF_TOPIC = "off_topic"                 # 与主线任务无关
    SPECIALIZED = "specialized"             # 需要特定领域能力
    MAIN_AGENT = "main_agent"               # 主Agent自己做（不委派）


# ================================================================
# 子Agent执行结果 —— 主Agent看到的"总结邮件"
# ================================================================

@dataclass
class SubAgentResult:
    """子Agent执行结果

    主Agent看不到子Agent内部做了什么（推理链、中间步骤、工具调用细节），
    只看到这个结构化的最终结论——就像同事发来的一封总结邮件。
    """
    task_id: str
    success: bool
    summary: str                            # 最终结论（主Agent唯一可见的内容）
    error: Optional[str] = None             # 失败原因
    retry_count: int = 0                    # 已重试次数
    execution_time: float = 0.0             # 执行耗时（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def __post_init__(self):
        """强制 summary 为 str，防止上游传入 dict/list 等导致切片报错"""
        if not isinstance(self.summary, str):
            self.summary = str(self.summary) if self.summary is not None else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "summary": self.summary,
            "error": self.error,
            "retry_count": self.retry_count,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubAgentResult':
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


# ================================================================
# 子任务定义
# ================================================================

@dataclass
class SubTask:
    """任务分解后的单个子任务

    每个子任务要么由主Agent自己完成，要么委派给动态创建的子Agent。
    委派时需要指定子Agent的角色、工具、上下文等配置。
    """
    id: str
    description: str
    assigned_to: str = "main"               # "main" 表示主Agent自己做
    delegation_reason: str = DelegationReason.MAIN_AGENT.value
    depends_on: List[str] = field(default_factory=list)
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[SubAgentResult] = None

    # ---- 子Agent配置（仅当 assigned_to != "main" 时有效）----
    agent_role: str = ""                    # 子Agent角色描述（系统提示词的核心）
    agent_tools: List[str] = field(default_factory=list)    # 工具白名单
    agent_context: str = ""                 # 裁剪后的精简上下文
    max_retries: int = 2                    # 最大重试次数
    timeout: float = 60.0                   # 执行超时（秒）

    # ---- 执行元数据 ----
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def mark_running(self):
        self.status = SubTaskStatus.RUNNING
        self.started_at = time.time()

    def mark_completed(self, result: SubAgentResult):
        self.status = SubTaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str, result: SubAgentResult = None):
        self.status = SubTaskStatus.FAILED
        if result:
            self.result = result
        else:
            self.result = SubAgentResult(
                task_id=self.id, success=False, summary="", error=error
            )
        self.completed_at = time.time()

    def mark_retrying(self):
        self.status = SubTaskStatus.RETRYING
        if self.result:
            self.result.retry_count += 1

    def mark_cancelled(self):
        self.status = SubTaskStatus.CANCELLED
        self.completed_at = time.time()

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            SubTaskStatus.COMPLETED, SubTaskStatus.FAILED, SubTaskStatus.CANCELLED
        )

    @property
    def can_retry(self) -> bool:
        retry_count = self.result.retry_count if self.result else 0
        return retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "delegation_reason": self.delegation_reason,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "agent_role": self.agent_role,
            "agent_tools": self.agent_tools,
            "agent_context": self.agent_context,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubTask':
        data = dict(data)  # shallow copy
        result_data = data.pop("result", None)
        status_str = data.pop("status", "pending")
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        obj = cls(**{k: v for k, v in data.items() if k in valid_keys})
        obj.status = SubTaskStatus(status_str)
        if result_data:
            obj.result = SubAgentResult.from_dict(result_data)
        return obj


# ================================================================
# 任务状态（主Agent持有的全局状态）
# ================================================================

@dataclass
class TaskState:
    """一次复杂任务的完整生命周期状态

    由主Agent维护，贯穿 理解→规划→执行→整合→交付 全过程。
    支持serialize/deserialize，由SessionManager持久化到数据库，
    会话恢复时重建上一次的任务状态。

    使用方式：
        state = TaskState(user_query="帮我推荐一套数码产品并下单")
        state.advance_phase(TaskPhase.PLAN)
        state.sub_tasks = [SubTask(...), ...]
        ...
        serialized = state.serialize()
        # 持久化到session
        session_manager.save_task_state(session_id, serialized)
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    phase: TaskPhase = TaskPhase.UNDERSTAND
    user_query: str = ""

    # ---- Phase 1 产出：理解 ----
    user_goal: str = ""                     # 用户真实目标（可能经过澄清后的）
    constraints: List[str] = field(default_factory=list)
    complexity: str = "simple"              # simple / medium / complex

    # ---- Phase 2 产出：规划 ----
    sub_tasks: List[SubTask] = field(default_factory=list)
    # 执行顺序分层：每层内的子任务可并行，层间串行
    # 例如 [["t1","t2"], ["t3"]] 表示 t1/t2可并行，完成后再执行t3
    execution_order: List[List[str]] = field(default_factory=list)

    # ---- Phase 3 状态：执行 ----
    current_layer: int = 0                  # 当前执行层

    # ---- Phase 4 产出：整合 ----
    integrated_result: str = ""

    # ---- Phase 5 产出：交付 ----
    final_response: str = ""

    # ---- 元数据 ----
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ================================================================
    # 状态转换
    # ================================================================

    def advance_phase(self, new_phase: TaskPhase):
        """推进到下一阶段"""
        self.phase = new_phase
        self.updated_at = time.time()

    # ================================================================
    # 子任务查询
    # ================================================================

    def get_ready_subtasks(self) -> List[SubTask]:
        """获取所有依赖已满足、可以执行的子任务"""
        completed_ids = {
            st.id for st in self.sub_tasks
            if st.status == SubTaskStatus.COMPLETED
        }
        return [
            st for st in self.sub_tasks
            if st.status == SubTaskStatus.PENDING
            and all(dep in completed_ids for dep in st.depends_on)
        ]

    def get_failed_subtasks(self) -> List[SubTask]:
        """获取所有失败的子任务"""
        return [st for st in self.sub_tasks if st.status == SubTaskStatus.FAILED]

    def get_retryable_subtasks(self) -> List[SubTask]:
        """获取所有可重试的失败子任务"""
        return [st for st in self.sub_tasks
                if st.status == SubTaskStatus.FAILED and st.can_retry]

    def get_subtask(self, task_id: str) -> Optional[SubTask]:
        """按ID查找子任务"""
        for st in self.sub_tasks:
            if st.id == task_id:
                return st
        return None

    def all_subtasks_done(self) -> bool:
        """所有子任务是否都已终结"""
        return all(st.is_terminal for st in self.sub_tasks)

    def has_failures(self) -> bool:
        """是否有失败的子任务"""
        return any(st.status == SubTaskStatus.FAILED for st in self.sub_tasks)

    def get_completed_results(self) -> Dict[str, SubAgentResult]:
        """获取所有已完成子任务的结果"""
        return {
            st.id: st.result
            for st in self.sub_tasks
            if st.status == SubTaskStatus.COMPLETED and st.result
        }

    def cancel_dependents(self, failed_task_id: str):
        """取消依赖于某个失败子任务的所有后续子任务"""
        to_cancel = set()
        to_cancel.add(failed_task_id)

        changed = True
        while changed:
            changed = False
            for st in self.sub_tasks:
                if st.id not in to_cancel and not st.is_terminal:
                    if any(dep in to_cancel for dep in st.depends_on):
                        to_cancel.add(st.id)
                        changed = True

        for st in self.sub_tasks:
            if st.id in to_cancel and st.status == SubTaskStatus.PENDING:
                st.mark_cancelled()

    # ================================================================
    # 序列化/反序列化
    # ================================================================

    def serialize(self) -> Dict[str, Any]:
        """序列化为可JSON化的dict，用于持久化"""
        return {
            "task_id": self.task_id,
            "phase": self.phase.value,
            "user_query": self.user_query,
            "user_goal": self.user_goal,
            "constraints": self.constraints,
            "complexity": self.complexity,
            "sub_tasks": [st.to_dict() for st in self.sub_tasks],
            "execution_order": self.execution_order,
            "current_layer": self.current_layer,
            "integrated_result": self.integrated_result,
            "final_response": self.final_response,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'TaskState':
        """从dict反序列化"""
        data = dict(data)
        sub_tasks_data = data.pop("sub_tasks", [])
        phase_str = data.pop("phase", "understand")
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        obj = cls(**{k: v for k, v in data.items() if k in valid_keys})
        obj.phase = TaskPhase(phase_str)
        obj.sub_tasks = [SubTask.from_dict(st) for st in sub_tasks_data]
        return obj

    def to_json(self) -> str:
        return json.dumps(self.serialize(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'TaskState':
        return cls.deserialize(json.loads(json_str))

    # ================================================================
    # 摘要（用于日志和调试）
    # ================================================================

    def summary(self) -> str:
        total = len(self.sub_tasks)
        completed = sum(1 for st in self.sub_tasks if st.status == SubTaskStatus.COMPLETED)
        failed = sum(1 for st in self.sub_tasks if st.status == SubTaskStatus.FAILED)
        return (
            f"TaskState[{self.task_id}] phase={self.phase.value} "
            f"subtasks={completed}/{total} done, {failed} failed"
        )
