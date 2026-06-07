"""
评估结果持久化存储 (Evaluation Store)

将评估结果持久化到 SQLite，支持：
1. AgentEvaluator 的任务执行记录
2. RegressionRunner 的测试报告
3. RAGEvaluator 的质量评分
"""
import json
import os
import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class EvalStore:
    """
    评估结果持久化存储

    使用 SQLite 存储所有评估数据，支持查询历史趋势。
    """

    def __init__(self, db_path: str = "./data/eval_store.db", logger: logging.Logger = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self.logger.info(f"EvalStore 初始化完成: {db_path}")

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- 任务执行记录（来自 AgentEvaluator）
                CREATE TABLE IF NOT EXISTS task_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    query TEXT,
                    intent_type TEXT,
                    complexity TEXT,
                    success INTEGER,
                    steps_count INTEGER DEFAULT 0,
                    replan_count INTEGER DEFAULT 0,
                    duration_ms REAL DEFAULT 0,
                    rag_score REAL DEFAULT -1,
                    eval_passed INTEGER DEFAULT 1,
                    error TEXT DEFAULT '',
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    llm_calls INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                -- 回归测试报告（来自 RegressionRunner）
                CREATE TABLE IF NOT EXISTS regression_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    total INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    skipped INTEGER,
                    errors INTEGER,
                    pass_rate REAL,
                    total_duration_ms REAL,
                    model TEXT,
                    report_json TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                -- RAG 质量评分（来自 RAGEvaluator）
                CREATE TABLE IF NOT EXISTS rag_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    retrieval_relevance REAL,
                    faithfulness REAL,
                    answer_relevance REAL,
                    context_utilization REAL,
                    overall_score REAL,
                    num_docs INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                -- 创建索引
                CREATE INDEX IF NOT EXISTS idx_task_created ON task_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_task_intent ON task_records(intent_type);
                CREATE INDEX IF NOT EXISTS idx_report_created ON regression_reports(created_at);
                CREATE INDEX IF NOT EXISTS idx_rag_created ON rag_scores(created_at);
            """)

    # ================================================================
    # 任务记录
    # ================================================================
    def save_task_record(self, record: Dict[str, Any]):
        """保存单条任务执行记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO task_records
                (task_id, query, intent_type, complexity, success, steps_count,
                 replan_count, duration_ms, rag_score, eval_passed, error,
                 input_tokens, output_tokens, llm_calls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get('task_id', ''),
                record.get('query', ''),
                record.get('intent_type', ''),
                record.get('complexity', ''),
                1 if record.get('success') else 0,
                record.get('steps_count', 0),
                record.get('replan_count', 0),
                record.get('duration_ms', 0),
                record.get('rag_score', -1),
                1 if record.get('eval_passed', True) else 0,
                record.get('error', ''),
                record.get('input_tokens', 0),
                record.get('output_tokens', 0),
                record.get('llm_calls', 0),
            ))

    def get_task_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取最近 N 天的任务统计"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(success) as completed,
                    COUNT(*) - SUM(success) as failed,
                    ROUND(AVG(CASE WHEN success=1 THEN 1.0 ELSE 0.0 END), 3) as completion_rate,
                    ROUND(AVG(duration_ms), 1) as avg_duration_ms,
                    ROUND(AVG(steps_count), 1) as avg_steps,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(llm_calls) as total_llm_calls
                FROM task_records
                WHERE created_at >= datetime('now', '-{days} days', 'localtime')
            """).fetchone()

            # 按意图类型分布
            intent_rows = conn.execute(f"""
                SELECT intent_type, COUNT(*) as count
                FROM task_records
                WHERE created_at >= datetime('now', '-{days} days', 'localtime')
                GROUP BY intent_type
                ORDER BY count DESC
            """).fetchall()

            return {
                "period_days": days,
                "total": row["total"],
                "completed": row["completed"],
                "failed": row["failed"],
                "completion_rate": row["completion_rate"],
                "avg_duration_ms": row["avg_duration_ms"],
                "avg_steps": row["avg_steps"],
                "total_input_tokens": row["total_input_tokens"],
                "total_output_tokens": row["total_output_tokens"],
                "total_llm_calls": row["total_llm_calls"],
                "intent_distribution": {r["intent_type"]: r["count"] for r in intent_rows},
            }

    # ================================================================
    # 回归测试报告
    # ================================================================
    def save_regression_report(self, report_dict: Dict[str, Any]):
        """保存回归测试报告"""
        summary = report_dict.get('summary', {})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO regression_reports
                (run_id, total, passed, failed, skipped, errors, pass_rate,
                 total_duration_ms, model, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_dict.get('run_id', ''),
                summary.get('total', 0),
                summary.get('passed', 0),
                summary.get('failed', 0),
                summary.get('skipped', 0),
                summary.get('errors', 0),
                summary.get('pass_rate', 0),
                summary.get('total_duration_ms', 0),
                report_dict.get('config_summary', {}).get('model', ''),
                json.dumps(report_dict, ensure_ascii=False),
            ))

    def get_regression_trend(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近 N 次回归测试的趋势"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT run_id, total, passed, failed, pass_rate, model, created_at
                FROM regression_reports
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [dict(r) for r in rows]

    # ================================================================
    # RAG 评分
    # ================================================================
    def save_rag_score(self, result: Dict[str, Any]):
        """保存 RAG 评估分数"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rag_scores
                (query, retrieval_relevance, faithfulness, answer_relevance,
                 context_utilization, overall_score, num_docs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.get('query', ''),
                result.get('retrieval_relevance', 0),
                result.get('faithfulness', 0),
                result.get('answer_relevance', 0),
                result.get('context_utilization', 0),
                result.get('overall_score', 0),
                result.get('details', {}).get('num_docs_retrieved', 0),
            ))

    def get_rag_trend(self, days: int = 30) -> Dict[str, Any]:
        """获取最近 N 天的 RAG 质量趋势"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(f"""
                SELECT
                    COUNT(*) as count,
                    ROUND(AVG(retrieval_relevance), 3) as avg_retrieval_relevance,
                    ROUND(AVG(faithfulness), 3) as avg_faithfulness,
                    ROUND(AVG(answer_relevance), 3) as avg_answer_relevance,
                    ROUND(AVG(context_utilization), 3) as avg_context_utilization,
                    ROUND(AVG(overall_score), 3) as avg_overall_score
                FROM rag_scores
                WHERE created_at >= datetime('now', '-{days} days', 'localtime')
            """).fetchone()

            return dict(row) if row else {}

    # ================================================================
    # 报告生成
    # ================================================================
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """生成仪表盘所需的汇总数据"""
        return {
            "task_stats_7d": self.get_task_stats(days=7),
            "task_stats_30d": self.get_task_stats(days=30),
            "regression_trend": self.get_regression_trend(limit=10),
            "rag_trend_30d": self.get_rag_trend(days=30),
            "generated_at": datetime.now().isoformat(),
        }
