"""
自动化回归测试运行器 (Regression Test Runner)

基于 Golden Test Suite 运行自动化评估，生成详细的测试报告。
支持三种测试模式：
1. 意图识别测试 — 仅测试意图识别准确性（快速，不启动完整 Agent）
2. 端到端测试 — 完整 Agent 流程（慢，但全面）
3. 鲁棒性测试 — 验证安全防护和边界处理

用法:
    python -m evaluation.regression_runner                    # 运行全部测试
    python -m evaluation.regression_runner --category intent  # 仅运行意图测试
    python -m evaluation.regression_runner --priority P0      # 仅运行 P0 测试
    python -m evaluation.regression_runner --tags rag,mcp     # 按标签筛选
    python -m evaluation.regression_runner --report           # 生成 JSON 报告
"""
import time
import json
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .golden_test_suite import (
    GoldenTestSuite,
    GoldenTestCase,
    TestCategory,
    ExpectedOutcome,
)


class TestStatus(str, Enum):
    """单条测试的执行状态"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """单条测试结果"""
    test_id: str
    test_name: str
    category: str
    status: TestStatus
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)  # 逐项检查结果

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
            "details": self.details,
            "error_message": self.error_message,
            "checks": self.checks,
        }


@dataclass
class TestReport:
    """测试报告汇总"""
    run_id: str
    timestamp: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_duration_ms: float = 0.0
    pass_rate: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "errors": self.errors,
                "pass_rate": round(self.pass_rate, 3),
                "total_duration_ms": round(self.total_duration_ms, 1),
            },
            "config_summary": self.config_summary,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        status_icon = "✅" if self.pass_rate >= 0.95 else ("⚠️" if self.pass_rate >= 0.8 else "❌")
        lines = [
            f"\n{'=' * 60}",
            f"{status_icon} 回归测试报告 | {self.timestamp}",
            f"{'=' * 60}",
            f"  总用例: {self.total} | ✅ 通过: {self.passed} | ❌ 失败: {self.failed} | ⏭️ 跳过: {self.skipped} | 💥 异常: {self.errors}",
            f"  通过率: {self.pass_rate:.1%} | 总耗时: {self.total_duration_ms:.0f}ms",
        ]

        # 列出失败的用例
        failed_results = [r for r in self.results if r.status == TestStatus.FAILED]
        if failed_results:
            lines.append(f"\n❌ 失败用例 ({len(failed_results)}):")
            for r in failed_results:
                lines.append(f"  - [{r.test_id}] {r.test_name}: {r.error_message}")

        error_results = [r for r in self.results if r.status == TestStatus.ERROR]
        if error_results:
            lines.append(f"\n💥 异常用例 ({len(error_results)}):")
            for r in error_results:
                lines.append(f"  - [{r.test_id}] {r.test_name}: {r.error_message}")

        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


class RegressionRunner:
    """
    回归测试运行器

    协调 GoldenTestSuite 和 Agent，执行自动化测试并生成报告。
    """

    def __init__(self, agent=None, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Args:
            agent: SmartAgent 实例（端到端测试需要）
            config: 配置字典
            logger: 日志记录器
        """
        self.agent = agent
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.suite = GoldenTestSuite(logger=self.logger)

    def run_all(
        self,
        categories: List[TestCategory] = None,
        priority: str = None,
        tags: List[str] = None,
    ) -> TestReport:
        """
        运行测试集

        Args:
            categories: 只运行指定类别的测试
            priority: 只运行指定优先级的测试
            tags: 只运行包含指定标签的测试

        Returns:
            TestReport
        """
        # 筛选测试用例
        cases = self.suite.all_cases

        if categories:
            cases = [c for c in cases if c.category in categories]
        if priority:
            cases = [c for c in cases if c.priority == priority]
        if tags:
            tag_set = set(tags)
            cases = [c for c in cases if tag_set.intersection(set(c.tags))]

        self.logger.info(f"准备运行 {len(cases)} 条测试用例")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = TestReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config_summary={
                "model": self.config.get('llm', {}).get('model', 'unknown'),
                "categories": [c.value for c in categories] if categories else ["all"],
                "priority": priority or "all",
                "tags": tags or [],
            },
        )

        total_start = time.time()

        for case in cases:
            result = self._run_single_case(case)
            report.results.append(result)

        report.total_duration_ms = (time.time() - total_start) * 1000
        report.total = len(report.results)
        report.passed = sum(1 for r in report.results if r.status == TestStatus.PASSED)
        report.failed = sum(1 for r in report.results if r.status == TestStatus.FAILED)
        report.skipped = sum(1 for r in report.results if r.status == TestStatus.SKIPPED)
        report.errors = sum(1 for r in report.results if r.status == TestStatus.ERROR)
        report.pass_rate = report.passed / report.total if report.total > 0 else 0.0

        self.logger.info(report.summary())
        return report

    def _run_single_case(self, case: GoldenTestCase) -> TestResult:
        """运行单条测试用例"""
        self.logger.info(f"运行测试: [{case.id}] {case.name}")
        start = time.time()

        try:
            if case.category == TestCategory.INTENT:
                result = self._run_intent_test(case)
            elif case.category == TestCategory.ROBUSTNESS:
                result = self._run_robustness_test(case)
            elif case.category in (TestCategory.END_TO_END, TestCategory.RAG_QUALITY):
                result = self._run_e2e_test(case)
            elif case.category == TestCategory.MULTI_TURN:
                result = self._run_multi_turn_test(case)
            else:
                result = TestResult(
                    test_id=case.id,
                    test_name=case.name,
                    category=case.category.value,
                    status=TestStatus.SKIPPED,
                    error_message=f"不支持的测试类别: {case.category.value}",
                )

            result.duration_ms = (time.time() - start) * 1000
            status_emoji = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "error": "💥"}
            self.logger.info(
                f"  {status_emoji.get(result.status.value, '?')} [{case.id}] {result.status.value} "
                f"({result.duration_ms:.0f}ms)"
            )
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.logger.error(f"  💥 [{case.id}] 测试异常: {e}")
            return TestResult(
                test_id=case.id,
                test_name=case.name,
                category=case.category.value,
                status=TestStatus.ERROR,
                duration_ms=duration,
                error_message=str(e),
            )

    # ================================================================
    # 意图识别测试
    # ================================================================
    def _run_intent_test(self, case: GoldenTestCase) -> TestResult:
        """测试意图识别准确性（不需要完整 Agent）"""
        if not self.agent:
            return TestResult(
                test_id=case.id, test_name=case.name, category=case.category.value,
                status=TestStatus.SKIPPED, error_message="未提供 Agent 实例",
            )

        checks = []

        # 调用意图识别
        intent_result = self.agent.intent_recognizer.recognize(case.input_query, case.context)

        # 检查意图类型
        if case.expected_intent:
            match = intent_result.intent_type.value == case.expected_intent
            checks.append({
                "check": "intent_type",
                "expected": case.expected_intent,
                "actual": intent_result.intent_type.value,
                "passed": match,
            })

        # 检查复杂度
        if case.expected_complexity:
            match = intent_result.complexity.value == case.expected_complexity
            checks.append({
                "check": "complexity",
                "expected": case.expected_complexity,
                "actual": intent_result.complexity.value,
                "passed": match,
            })

        # 检查工具名
        if case.expected_tool:
            match = intent_result.tool_name == case.expected_tool
            checks.append({
                "check": "tool_name",
                "expected": case.expected_tool,
                "actual": intent_result.tool_name,
                "passed": match,
            })

        # 判断整体是否通过
        all_passed = all(c["passed"] for c in checks) if checks else True
        failed_checks = [c for c in checks if not c["passed"]]

        return TestResult(
            test_id=case.id,
            test_name=case.name,
            category=case.category.value,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            checks=checks,
            details={
                "confidence": intent_result.confidence,
                "reason": intent_result.reason,
            },
            error_message=f"检查失败: {failed_checks}" if failed_checks else "",
        )

    # ================================================================
    # 鲁棒性测试
    # ================================================================
    def _run_robustness_test(self, case: GoldenTestCase) -> TestResult:
        """测试鲁棒性（安全防护、边界情况）"""
        if not self.agent:
            return TestResult(
                test_id=case.id, test_name=case.name, category=case.category.value,
                status=TestStatus.SKIPPED, error_message="未提供 Agent 实例",
            )

        checks = []

        # 通过输入验证器测试
        validation = self.agent.input_validator.validate(case.input_query)

        if case.expected_blocked:
            # 期望被拦截
            blocked = not validation.is_valid
            checks.append({
                "check": "input_blocked",
                "expected": True,
                "actual": blocked,
                "passed": blocked,
                "detail": validation.block_reason if not validation.is_valid else "未拦截",
            })
        else:
            # 不期望被拦截，但要确保不会崩溃
            try:
                response = self.agent.chat(case.input_query, verbose=False)
                checks.append({
                    "check": "no_crash",
                    "expected": "正常响应",
                    "actual": f"响应长度: {len(response)}",
                    "passed": True,
                })

                # 检查不应包含的关键词
                for keyword in case.expected_answer_not_contains:
                    found = keyword.lower() in response.lower()
                    checks.append({
                        "check": f"not_contains:{keyword}",
                        "expected": False,
                        "actual": found,
                        "passed": not found,
                    })
            except Exception as e:
                checks.append({
                    "check": "no_crash",
                    "expected": "正常响应",
                    "actual": f"异常: {e}",
                    "passed": False,
                })

        all_passed = all(c["passed"] for c in checks) if checks else True
        failed_checks = [c for c in checks if not c["passed"]]

        return TestResult(
            test_id=case.id,
            test_name=case.name,
            category=case.category.value,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            checks=checks,
            error_message=f"检查失败: {failed_checks}" if failed_checks else "",
        )

    # ================================================================
    # 端到端测试
    # ================================================================
    def _run_e2e_test(self, case: GoldenTestCase) -> TestResult:
        """端到端测试（完整 Agent 流程）"""
        if not self.agent:
            return TestResult(
                test_id=case.id, test_name=case.name, category=case.category.value,
                status=TestStatus.SKIPPED, error_message="未提供 Agent 实例",
            )

        checks = []

        try:
            # 运行 Agent
            response = self.agent.chat(case.input_query, verbose=False)

            # 检查应包含的关键词
            for keyword in case.expected_answer_contains:
                found = keyword.lower() in response.lower()
                checks.append({
                    "check": f"contains:{keyword}",
                    "expected": True,
                    "actual": found,
                    "passed": found,
                })

            # 检查不应包含的关键词
            for keyword in case.expected_answer_not_contains:
                found = keyword.lower() in response.lower()
                checks.append({
                    "check": f"not_contains:{keyword}",
                    "expected": False,
                    "actual": found,
                    "passed": not found,
                })

            # 检查非空响应
            checks.append({
                "check": "non_empty_response",
                "expected": True,
                "actual": bool(response and response.strip()),
                "passed": bool(response and response.strip()),
            })

        except Exception as e:
            checks.append({
                "check": "execution",
                "expected": "成功执行",
                "actual": f"异常: {e}",
                "passed": False,
            })

        all_passed = all(c["passed"] for c in checks) if checks else True
        failed_checks = [c for c in checks if not c["passed"]]

        return TestResult(
            test_id=case.id,
            test_name=case.name,
            category=case.category.value,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            checks=checks,
            error_message=f"检查失败: {failed_checks}" if failed_checks else "",
        )

    # ================================================================
    # 多轮对话测试
    # ================================================================
    def _run_multi_turn_test(self, case: GoldenTestCase) -> TestResult:
        """多轮对话测试

        模拟多轮对话历史，通过注入短期记忆实现上下文传递。
        """
        if not self.agent:
            return TestResult(
                test_id=case.id, test_name=case.name, category=case.category.value,
                status=TestStatus.SKIPPED, error_message="未提供 Agent 实例",
            )

        checks = []

        try:
            # 将历史对话注入短期记忆
            for turn in case.conversation_turns[:-1]:  # 排除最后一轮（待测试的输入）
                self.agent.short_term_memory.add_message(turn["role"], turn["content"])

            # 执行最后一轮
            response = self.agent.chat(case.input_query, verbose=False)

            # 检查应包含的关键词
            for keyword in case.expected_answer_contains:
                found = keyword.lower() in response.lower()
                checks.append({
                    "check": f"contains:{keyword}",
                    "expected": True,
                    "actual": found,
                    "passed": found,
                })

            # 检查不应包含的关键词
            for keyword in case.expected_answer_not_contains:
                found = keyword.lower() in response.lower()
                checks.append({
                    "check": f"not_contains:{keyword}",
                    "expected": False,
                    "actual": found,
                    "passed": not found,
                })

        except Exception as e:
            checks.append({
                "check": "multi_turn_execution",
                "expected": "成功执行",
                "actual": f"异常: {e}",
                "passed": False,
            })

        all_passed = all(c["passed"] for c in checks) if checks else True
        failed_checks = [c for c in checks if not c["passed"]]

        return TestResult(
            test_id=case.id,
            test_name=case.name,
            category=case.category.value,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            checks=checks,
            error_message=f"检查失败: {failed_checks}" if failed_checks else "",
        )

    # ================================================================
    # 报告导出
    # ================================================================
    def save_report(self, report: TestReport, output_dir: str = "./data/eval_reports"):
        """保存测试报告到 JSON 文件"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"regression_{report.run_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        self.logger.info(f"测试报告已保存: {filepath}")
        return filepath


# ================================================================
# CLI 入口
# ================================================================
def main():
    """命令行入口"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Agent 回归测试运行器")
    parser.add_argument("--category", type=str, help="测试类别: intent, end_to_end, robustness, multi_turn, rag_quality")
    parser.add_argument("--priority", type=str, help="优先级: P0, P1, P2")
    parser.add_argument("--tags", type=str, help="标签过滤（逗号分隔）")
    parser.add_argument("--report", action="store_true", help="保存 JSON 报告")
    parser.add_argument("--config", type=str, default="./config/settings.yaml", help="配置文件路径")
    parser.add_argument("--intent-only", action="store_true", help="仅运行意图识别测试（快速模式，不需要 MCP 服务）")
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("RegressionRunner")

    # 解析参数
    categories = None
    if args.category:
        try:
            categories = [TestCategory(args.category)]
        except ValueError:
            logger.error(f"无效的测试类别: {args.category}")
            sys.exit(1)

    if args.intent_only:
        categories = [TestCategory.INTENT]

    tags = args.tags.split(",") if args.tags else None

    # 初始化 Agent
    logger.info("初始化 SmartAgent...")
    try:
        # 将项目根目录加入 sys.path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from agent import SmartAgent
        agent = SmartAgent(config_path=args.config)
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        logger.info("将跳过需要 Agent 的测试")
        agent = None

    # 运行测试
    runner = RegressionRunner(agent=agent, config=agent.config if agent else {}, logger=logger)
    report = runner.run_all(categories=categories, priority=args.priority, tags=tags)

    # 输出报告
    print(report.summary())

    if args.report:
        filepath = runner.save_report(report)
        print(f"📄 报告已保存: {filepath}")

    # 返回退出码（供 CI 使用）
    sys.exit(0 if report.pass_rate >= 0.95 else 1)


if __name__ == "__main__":
    main()
