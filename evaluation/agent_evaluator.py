"""
Agent整体效果评估模块 (Agent Evaluator)

多维度评估Agent性能：
1. 任务完成率 (Task Completion Rate): 任务是否成功完成
2. 鲁棒性 (Robustness): 面对异常输入和边界情况的表现
3. 效率 (Efficiency): 完成任务的步骤数和耗时
4. 用户满意度估计 (User Satisfaction): 基于回复质量的估算
5. RAG质量 (RAG Quality): 整合RAGEvaluator结果
"""
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


@dataclass
class TaskRecord:
    """单次任务记录"""
    task_id: str
    query: str
    intent_type: str
    complexity: str
    start_time: float
    end_time: float = 0.0
    success: bool = False
    steps_count: int = 0
    replan_count: int = 0
    error: str = ""
    rag_score: float = -1.0   # RAG评估分数，-1表示无RAG
    eval_passed: bool = True   # 任务评估是否通过


@dataclass
class AgentMetrics:
    """Agent指标汇总"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    completion_rate: float = 0.0
    avg_steps_per_task: float = 0.0
    avg_response_time_ms: float = 0.0
    avg_rag_score: float = 0.0
    replan_rate: float = 0.0        # 需要重新规划的比例
    robustness_score: float = 0.0   # 鲁棒性评分
    intent_distribution: Dict[str, int] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[str] = field(default_factory=list)
    evaluation_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "completion_rate": round(self.completion_rate, 3),
            "avg_steps_per_task": round(self.avg_steps_per_task, 2),
            "avg_response_time_ms": round(self.avg_response_time_ms, 1),
            "avg_rag_score": round(self.avg_rag_score, 3),
            "replan_rate": round(self.replan_rate, 3),
            "robustness_score": round(self.robustness_score, 3),
            "intent_distribution": self.intent_distribution,
            "complexity_distribution": self.complexity_distribution,
            "recent_errors": self.recent_errors[-5:],
            "evaluation_timestamp": self.evaluation_timestamp,
        }

    def summary(self) -> str:
        """生成可读的评估摘要"""
        return (
            f"📊 Agent评估报告\n"
            f"  总任务: {self.total_tasks} | 完成: {self.completed_tasks} | 失败: {self.failed_tasks}\n"
            f"  完成率: {self.completion_rate:.1%}\n"
            f"  平均步骤: {self.avg_steps_per_task:.1f} | 平均耗时: {self.avg_response_time_ms:.0f}ms\n"
            f"  重规划率: {self.replan_rate:.1%} | 鲁棒性: {self.robustness_score:.2f}\n"
            f"  RAG平均质量: {self.avg_rag_score:.2f}\n"
        )


class AgentEvaluator:
    """
    Agent整体效果评估器

    持续收集任务执行数据，提供多维度的Agent性能评估。
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        eval_config = config.get('evaluation', {}).get('agent', {})
        self.enabled = eval_config.get('enabled', True)
        self.max_records = eval_config.get('max_records', 1000)

        # 任务记录
        self._records: List[TaskRecord] = []
        # 错误统计
        self._error_counts: Dict[str, int] = defaultdict(int)
        # 运行中的任务
        self._running_tasks: Dict[str, TaskRecord] = {}

        self.logger.info("AgentEvaluator 初始化完成")

    # ================================================================
    # 任务生命周期追踪
    # ================================================================
    def task_started(self, task_id: str, query: str, intent_type: str, complexity: str):
        """记录任务开始"""
        if not self.enabled:
            return
        record = TaskRecord(
            task_id=task_id,
            query=query,
            intent_type=intent_type,
            complexity=complexity,
            start_time=time.time(),
        )
        self._running_tasks[task_id] = record
        self.logger.debug(f"[Agent评估] 任务开始: {task_id} | {intent_type}/{complexity}")

    def task_completed(
        self,
        task_id: str,
        success: bool = True,
        steps_count: int = 0,
        replan_count: int = 0,
        rag_score: float = -1.0,
        eval_passed: bool = True,
        error: str = "",
    ):
        """记录任务完成"""
        if not self.enabled:
            return

        record = self._running_tasks.pop(task_id, None)
        if not record:
            return

        record.end_time = time.time()
        record.success = success
        record.steps_count = steps_count
        record.replan_count = replan_count
        record.rag_score = rag_score
        record.eval_passed = eval_passed
        record.error = error

        self._records.append(record)

        # 维护最大记录数
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        if error:
            self._error_counts[error[:100]] += 1

        elapsed = (record.end_time - record.start_time) * 1000
        status = "成功" if success else "失败"
        self.logger.info(
            f"[Agent评估] 任务{status}: {task_id} | 步骤: {steps_count} | "
            f"耗时: {elapsed:.0f}ms | 重规划: {replan_count}"
        )

    # ================================================================
    # 指标计算
    # ================================================================
    def get_metrics(self, last_n: Optional[int] = None) -> AgentMetrics:
        """
        计算Agent指标

        Args:
            last_n: 只统计最近n条记录，None表示全部

        Returns:
            AgentMetrics
        """
        records = self._records[-last_n:] if last_n else self._records

        if not records:
            return AgentMetrics(evaluation_timestamp=datetime.now().isoformat())

        total = len(records)
        completed = sum(1 for r in records if r.success)
        failed = total - completed

        # 平均步骤
        steps_list = [r.steps_count for r in records if r.steps_count > 0]
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0

        # 平均响应时间
        time_list = [(r.end_time - r.start_time) * 1000 for r in records if r.end_time > 0]
        avg_time = sum(time_list) / len(time_list) if time_list else 0

        # RAG平均分
        rag_scores = [r.rag_score for r in records if r.rag_score >= 0]
        avg_rag = sum(rag_scores) / len(rag_scores) if rag_scores else 0

        # 重规划率
        replanned = sum(1 for r in records if r.replan_count > 0)
        replan_rate = replanned / total if total > 0 else 0

        # 鲁棒性评分（基于完成率和错误多样性）
        error_diversity = len(set(r.error for r in records if r.error)) / max(total, 1)
        robustness = max(0, (completed / total) - error_diversity * 0.2) if total > 0 else 0

        # 意图分布
        intent_dist = defaultdict(int)
        complexity_dist = defaultdict(int)
        for r in records:
            intent_dist[r.intent_type] += 1
            complexity_dist[r.complexity] += 1

        # 最近错误
        recent_errors = [r.error for r in records if r.error][-5:]

        return AgentMetrics(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            completion_rate=completed / total if total > 0 else 0,
            avg_steps_per_task=avg_steps,
            avg_response_time_ms=avg_time,
            avg_rag_score=avg_rag,
            replan_rate=replan_rate,
            robustness_score=robustness,
            intent_distribution=dict(intent_dist),
            complexity_distribution=dict(complexity_dist),
            recent_errors=recent_errors,
            evaluation_timestamp=datetime.now().isoformat(),
        )

    def print_metrics(self, last_n: Optional[int] = None):
        """打印Agent指标"""
        metrics = self.get_metrics(last_n)
        print(metrics.summary())
