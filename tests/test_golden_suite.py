"""
pytest 集成测试 — 基于 Golden Test Suite

运行方式:
    # 运行全部黄金测试
    pytest tests/test_golden_suite.py -v

    # 仅运行意图识别测试（快速）
    pytest tests/test_golden_suite.py -v -k "intent"

    # 仅运行 P0 级别测试
    pytest tests/test_golden_suite.py -v -k "P0"

    # 仅运行鲁棒性测试
    pytest tests/test_golden_suite.py -v -k "robustness"

注意:
    - 意图识别测试和鲁棒性测试不需要 MCP 服务运行
    - 端到端测试需要完整环境（MCP 服务 + 数据库）
"""
import os
import sys
import pytest
import time

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.golden_test_suite import (
    GoldenTestSuite,
    GoldenTestCase,
    TestCategory,
    GOLDEN_TEST_CASES,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture(scope="session")
def agent():
    """创建一个共享的 SmartAgent 实例（整个测试会话复用）"""
    try:
        from agent import SmartAgent
        return SmartAgent(config_path="./config/settings.yaml")
    except Exception as e:
        pytest.skip(f"Agent 初始化失败，跳过需要 Agent 的测试: {e}")
        return None


@pytest.fixture(scope="session")
def suite():
    """创建 Golden Test Suite"""
    return GoldenTestSuite()


# ================================================================
# 测试辅助
# ================================================================

def _get_test_cases_by_category(category: TestCategory):
    """获取指定类别的测试用例"""
    return [tc for tc in GOLDEN_TEST_CASES if tc.category == category]


def _get_intent_cases():
    return _get_test_cases_by_category(TestCategory.INTENT)


def _get_robustness_cases():
    return _get_test_cases_by_category(TestCategory.ROBUSTNESS)


def _get_e2e_cases():
    return _get_test_cases_by_category(TestCategory.END_TO_END)


def _get_rag_cases():
    return _get_test_cases_by_category(TestCategory.RAG_QUALITY)


# ================================================================
# 意图识别测试
# ================================================================

class TestIntentRecognition:
    """意图识别准确性测试"""

    @pytest.mark.parametrize(
        "case",
        _get_intent_cases(),
        ids=[c.id for c in _get_intent_cases()],
    )
    def test_intent_recognition(self, agent, case: GoldenTestCase):
        """验证意图识别的准确性"""
        if agent is None:
            pytest.skip("Agent 不可用")

        intent_result = agent.intent_recognizer.recognize(case.input_query, case.context)

        # 检查意图类型
        if case.expected_intent:
            assert intent_result.intent_type.value == case.expected_intent, (
                f"[{case.id}] 意图类型不匹配: "
                f"期望 {case.expected_intent}, 实际 {intent_result.intent_type.value} "
                f"(原因: {intent_result.reason})"
            )

        # 检查复杂度
        if case.expected_complexity:
            assert intent_result.complexity.value == case.expected_complexity, (
                f"[{case.id}] 复杂度不匹配: "
                f"期望 {case.expected_complexity}, 实际 {intent_result.complexity.value}"
            )

        # 检查工具名
        if case.expected_tool:
            assert intent_result.tool_name == case.expected_tool, (
                f"[{case.id}] 工具名不匹配: "
                f"期望 {case.expected_tool}, 实际 {intent_result.tool_name}"
            )


# ================================================================
# 鲁棒性测试
# ================================================================

class TestRobustness:
    """安全防护和边界处理测试"""

    @pytest.mark.parametrize(
        "case",
        _get_robustness_cases(),
        ids=[c.id for c in _get_robustness_cases()],
    )
    def test_robustness(self, agent, case: GoldenTestCase):
        """验证 Agent 对异常输入的处理能力"""
        if agent is None:
            pytest.skip("Agent 不可用")

        if case.expected_blocked:
            # 期望被输入验证器拦截
            validation = agent.input_validator.validate(case.input_query)
            assert not validation.is_valid, (
                f"[{case.id}] 应被拦截但未拦截: {case.input_query[:50]}"
            )
        else:
            # 不应崩溃
            response = agent.chat(case.input_query, verbose=False)
            assert response is not None, f"[{case.id}] 响应为 None"
            assert isinstance(response, str), f"[{case.id}] 响应不是字符串"

            # 检查不应包含的关键词
            for keyword in case.expected_answer_not_contains:
                assert keyword.lower() not in response.lower(), (
                    f"[{case.id}] 响应不应包含 '{keyword}'"
                )


# ================================================================
# 端到端测试
# ================================================================

class TestEndToEnd:
    """端到端功能测试"""

    @pytest.mark.parametrize(
        "case",
        _get_e2e_cases(),
        ids=[c.id for c in _get_e2e_cases()],
    )
    def test_end_to_end(self, agent, case: GoldenTestCase):
        """验证完整 Agent 流程"""
        if agent is None:
            pytest.skip("Agent 不可用")

        start = time.time()
        response = agent.chat(case.input_query, verbose=False)
        duration_ms = (time.time() - start) * 1000

        # 基本检查
        assert response is not None, f"[{case.id}] 响应为 None"
        assert response.strip(), f"[{case.id}] 响应为空"

        # 关键词检查
        for keyword in case.expected_answer_contains:
            assert keyword.lower() in response.lower(), (
                f"[{case.id}] 响应应包含 '{keyword}'，实际响应: {response[:200]}"
            )

        for keyword in case.expected_answer_not_contains:
            assert keyword.lower() not in response.lower(), (
                f"[{case.id}] 响应不应包含 '{keyword}'"
            )

        # 延迟检查
        if case.max_latency_ms:
            assert duration_ms <= case.max_latency_ms, (
                f"[{case.id}] 响应超时: {duration_ms:.0f}ms > {case.max_latency_ms}ms"
            )


# ================================================================
# RAG 质量测试
# ================================================================

class TestRAGQuality:
    """RAG 检索和回答质量测试"""

    @pytest.mark.parametrize(
        "case",
        _get_rag_cases(),
        ids=[c.id for c in _get_rag_cases()],
    )
    def test_rag_quality(self, agent, case: GoldenTestCase):
        """验证 RAG 链路质量"""
        if agent is None:
            pytest.skip("Agent 不可用")

        response = agent.chat(case.input_query, verbose=False)

        # 基本检查
        assert response is not None and response.strip(), f"[{case.id}] 响应为空"

        # 关键词检查
        for keyword in case.expected_answer_contains:
            assert keyword.lower() in response.lower(), (
                f"[{case.id}] 响应应包含 '{keyword}'"
            )

        for keyword in case.expected_answer_not_contains:
            assert keyword.lower() not in response.lower(), (
                f"[{case.id}] 响应不应包含 '{keyword}'"
            )


# ================================================================
# 测试套件摘要
# ================================================================

class TestSuiteMetadata:
    """验证测试套件自身的完整性"""

    def test_suite_has_cases(self, suite):
        """测试套件不能为空"""
        assert len(suite.all_cases) > 0

    def test_unique_ids(self, suite):
        """所有测试用例的 ID 必须唯一"""
        ids = [tc.id for tc in suite.all_cases]
        assert len(ids) == len(set(ids)), f"存在重复 ID: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_categories_covered(self, suite):
        """至少覆盖主要测试类别"""
        categories = {tc.category for tc in suite.all_cases}
        required = {TestCategory.INTENT, TestCategory.ROBUSTNESS, TestCategory.END_TO_END}
        missing = required - categories
        assert not missing, f"缺少必要测试类别: {missing}"

    def test_p0_cases_exist(self, suite):
        """至少有 P0 级别的测试用例"""
        p0 = suite.get_by_priority("P0")
        assert len(p0) >= 5, f"P0 测试用例不足（当前: {len(p0)}，最少: 5）"
