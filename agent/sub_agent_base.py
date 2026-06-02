"""
子Agent基类

定义子Agent的统一接口，供 DynamicSubAgent 继承。
"""

import logging
from typing import Dict, Any


class SubAgentBase:
    """子Agent基类

    子类只需实现 handle_task()，工厂直接 await agent.handle_task(payload) 调用。
    """

    AGENT_TYPE = "base"    # 子类覆盖
    AGENT_NAME = "SubAgent" # 子类覆盖，用于日志前缀

    # 工具白名单：空列表=禁用所有工具，None=不限制
    ALLOWED_TOOLS: list = None

    def __init__(
        self,
        agent_id: str,
        llm_client,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None,
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self._log_tag = f"[{self.AGENT_NAME}]"

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理主Agent分派的任务（子类必须实现）

        Returns:
            {"success": bool, "response": str, "agent_result": dict}
        """
        raise NotImplementedError
