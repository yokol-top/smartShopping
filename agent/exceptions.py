"""
Agent 模块自定义异常
"""


class NeedUserInputException(Exception):
    """执行中途 LLM 判断需要用户补充信息时抛出。

    传播路径：
        UnifiedReActExecutor → TaskPlanner → Orchestrator.handle_request()
        → agent.py chat() 中返回给用户（与前置澄清走相同路径）

    并行子 Agent 中触发时：
        SubAgentFactory 捕获，将对应子任务标记为 PAUSED，
        等待其余并行任务完成后将 question 返回给用户。
    """

    def __init__(self, question: str):
        self.question = question
        super().__init__(question)
