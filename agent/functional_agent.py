"""
功能处理子Agent

职责：
- 执行业务操作（创建用户、创建订单、查询订单、修改用户信息等）
- 调用MCP工具完成实际的数据库操作
- 缺少必要参数时向主Agent请求

工作模式：
1. 解析用户意图，确定需要调用的MCP工具和参数
2. 检查参数完整性，缺少时向主Agent请求补充
3. 调用MCP工具执行操作
4. 格式化结果返回
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from .sub_agent_base import SubAgentBase
from .message_bus import AsyncMessageBus
from utils import LLMClient


class FunctionalAgent(SubAgentBase):
    """功能处理子Agent"""

    AGENT_TYPE = "functional"
    AGENT_NAME = "FunctionalAgent"

    # 功能子Agent允许使用用户管理和订单管理相关的MCP工具
    ALLOWED_TOOLS = [
        "create_user",
        "get_user_detail",
        "update_user_profile_refined",
        "create_complex_order",
        "query_order_detail",
        "list_all_orders",
    ]

    def __init__(
        self,
        bus: AsyncMessageBus,
        llm_client: LLMClient,
        mcp_manager=None,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None,
    ):
        super().__init__(
            agent_id="functional_agent",
            bus=bus,
            llm_client=llm_client,
            config=config,
            logger=logger,
        )
        self.mcp_manager = mcp_manager

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理功能性任务

        Args:
            task: {
                "user_query": 用户请求,
                "context": 已组装的上下文（由SubAgentContextBuilder构建）,
                "context_sections": 分区上下文dict,
                "user_id": 当前用户ID,
                "username": 当前用户名,
                "tool_name": 意图识别出的目标工具（可选）,
                "_ctx_builder": 上下文构建器引用
            }

        Returns:
            {"success": bool, "response": str}
        """
        user_query = task.get("user_query", "")
        context = task.get("context", "")
        user_id = task.get("user_id", "")
        username = task.get("username", "")
        tool_name = task.get("tool_name", "")
        ctx_sections = task.get("context_sections", {})
        ctx_builder = task.get("_ctx_builder")

        self.logger.info(f"{self._log_tag} 处理任务: {user_query[:60]} | 目标工具: {tool_name}")

        # 动态注入工具列表到领域上下文
        if ctx_builder and ctx_sections is not None:
            tool_list = self._get_available_tools()
            ctx_sections['domain_context'] = ctx_builder.build_domain_context(
                tool_list=tool_list,
            )

        # Step 1: 确定要调用的工具
        if not tool_name:
            tool_name = self._select_tool(user_query, context)
        if not tool_name:
            return {"success": False, "response": "无法确定需要执行的操作，请提供更多信息。"}

        # Step 2: 基于 schema 提取参数（与主Agent TaskPlanner._extract_and_call_mcp 一致）
        parameters = self._extract_and_fill_params(
            tool_name, user_query, context, user_id, username
        )

        # Step 3: 检查必填参数，缺少时向主Agent请求
        parameters = await self._fill_missing_params(
            tool_name, parameters, user_query, user_id, username
        )
        # 动态追加主Agent回复到上下文
        if ctx_builder and ctx_sections is not None:
            extra_ctx = self.get_accumulated_context()
            if extra_ctx:
                ctx_sections['main_replies'] = extra_ctx

        # Step 4: 执行MCP工具调用
        result = await self._execute_tool(tool_name, parameters)

        # Step 5: 检查执行后是否有主Agent的额外通知
        await self.check_for_updates()

        # Step 6: 格式化结果
        response = self._format_result(user_query, tool_name, parameters, result)

        return {"success": result.get("success", False), "response": response}

    # ================================================================
    # Schema 驱动的工具调用（复用主Agent TaskPlanner 的设计模式）
    # ================================================================

    def _get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """从MCP动态获取工具的 inputSchema（与主Agent TaskPlanner._get_tool_schema 一致）"""
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_available_tools(use_cache=True):
            if tool.get("name") == tool_name:
                return tool.get("inputSchema", {})
        return None

    def _select_tool(self, query: str, context: str) -> str:
        """当意图识别未确定工具时，用LLM从可用工具中选择"""
        available_tools = self._get_available_tools()
        prompt = f"""根据用户请求，从可用工具中选择最合适的一个。

用户请求：{query}

可用工具：
{available_tools}

只返回工具名称，不要其他内容。"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.1).strip()
            if resp and self.is_tool_allowed(resp):
                return resp
            self.logger.warning(f"{self._log_tag} LLM选择了无效工具: {resp}")
        except Exception as e:
            self.logger.error(f"{self._log_tag} 工具选择失败: {e}")
        return ""

    def _extract_and_fill_params(
            self, tool_name: str, user_query: str, context: str,
            user_id: str, username: str,
    ) -> Dict[str, Any]:
        """基于 MCP inputSchema 提取参数

        与主Agent TaskPlanner._extract_and_call_mcp 采用相同的设计模式：
        1. 懒加载：运行时从MCP获取工具的 JSON Schema
        2. Schema 作为 prompt 的一部分传给 LLM
        3. LLM 严格按照 schema 结构输出参数
        4. 已知值（user_id/operator_id）在 prompt 和后处理中双重保障
        """
        schema = self._get_tool_schema(tool_name)
        if not schema:
            self.logger.warning(f"{self._log_tag} 未获取到工具 {tool_name} 的 schema")
            return {}

        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

        prompt = f"""你是一个工具调用助手。请根据用户请求，从上下文中提取工具参数。

用户请求: {user_query}
对话历史: {context}

当前用户: user_id={user_id}, username={username}

工具名: {tool_name}
工具参数结构（JSON Schema）:
{schema_json}

**关键规则**:
1. 严格按照 schema 结构提取参数，字段名和嵌套层级必须与 schema 完全一致。
2. 必需参数（required）必须提供，可选参数在用户未提及时不填。
3. 如果 schema 中有 operator_id 或 user_id 字段，使用当前用户的 user_id: {user_id}。
4. 参数优先级：用户请求中的明确信息 > 对话上下文中的信息 > 默认值。
5. 根据上下文智能推断操作类型：如果 schema 中包含 action 字段（如 add/update/remove），需根据用户意图判断。
6. 不要猜测或编造用户未明确提供的值。

只返回JSON。"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.2)
            params = json.loads(self._clean_json_response(resp))

            # 后处理：确保 user_id / operator_id 被正确注入（防止 LLM 漏填或填错）
            self._inject_known_values(params, schema, user_id)

            self.logger.info(
                f"{self._log_tag} Schema驱动参数提取: "
                f"{json.dumps(params, ensure_ascii=False)[:500]}"
            )
            return params
        except Exception as e:
            self.logger.error(f"{self._log_tag} 参数提取失败: {e}")
            return {}

    @staticmethod
    def _clean_json_response(resp: str) -> str:
        """清理LLM返回的JSON（处理常见的LLM输出污染）

        处理场景：
        1. ```json ... ``` markdown代码块
        2. JSON后面跟了额外文本说明 → "Extra data" 错误
        3. 多行JSON中混入解释文字
        """
        import re
        resp = resp.strip()

        # 场景1：提取 ```json ... ``` 中的内容
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', resp, re.DOTALL)
        if code_block:
            resp = code_block.group(1).strip()

        # 场景2：提取第一个完整的 JSON 对象 {...}
        # 用栈匹配花括号，找到第一个完整对象
        brace_start = resp.find('{')
        if brace_start >= 0:
            depth = 0
            in_string = False
            escape = False
            for i in range(brace_start, len(resp)):
                c = resp[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        resp = resp[brace_start:i + 1]
                        break

        return resp.strip()

    def _inject_known_values(
            self, params: Dict[str, Any], schema: Dict[str, Any], user_id: str,
    ):
        """将已知值注入到参数中（递归处理嵌套对象）

        遍历 schema properties，找到名为 operator_id / user_id 的字段，
        无论在顶层还是通过 $ref / allOf 嵌套，都确保填入当前用户值。
        """
        known = {"operator_id": user_id, "user_id": user_id}
        properties = schema.get("properties", {})
        defs = schema.get("$defs", {})

        for field_name, field_schema in properties.items():
            # 顶层字段直接注入
            if field_name in known:
                params[field_name] = known[field_name]
                continue

            # 解析嵌套对象的 properties（支持 $ref 和 allOf）
            nested_props = self._resolve_nested_properties(field_schema, defs)
            if nested_props and any(k in nested_props for k in known):
                if field_name not in params:
                    params[field_name] = {}
                if isinstance(params[field_name], dict):
                    for k, v in known.items():
                        if k in nested_props:
                            params[field_name][k] = v

    @staticmethod
    def _resolve_nested_properties(
            field_schema: Dict[str, Any], defs: Dict[str, Any]
    ) -> Optional[Dict]:
        """解析字段的嵌套 properties（处理 $ref / allOf / anyOf）"""
        # 直接 $ref
        ref = field_schema.get("$ref", "")
        if ref:
            def_name = ref.split("/")[-1]
            return defs.get(def_name, {}).get("properties")

        # allOf（Pydantic optional model 常见模式）
        for item in field_schema.get("allOf", []):
            ref = item.get("$ref", "")
            if ref:
                def_name = ref.split("/")[-1]
                props = defs.get(def_name, {}).get("properties")
                if props:
                    return props

        # anyOf（Optional[Model] 另一种模式）
        for item in field_schema.get("anyOf", []):
            ref = item.get("$ref", "")
            if ref:
                def_name = ref.split("/")[-1]
                props = defs.get(def_name, {}).get("properties")
                if props:
                    return props

        # 直接 object
        return field_schema.get("properties")

    async def _fill_missing_params(
            self, tool_name: str, params: Dict[str, Any],
            query: str, user_id: str, username: str,
    ) -> Dict[str, Any]:
        """检查必填参数（从 schema 动态读取 required），缺失时向主Agent请求"""
        schema = self._get_tool_schema(tool_name)
        if not schema:
            return params

        required = schema.get("required", [])
        missing = [p for p in required if p not in params or not params[p]]

        if not missing:
            return params

        self.logger.info(f"{self._log_tag} 工具 {tool_name} 缺少必填参数: {missing}")

        # 从 schema 提取缺失字段的描述，让主Agent知道需要什么
        properties = schema.get("properties", {})
        missing_desc = []
        for p in missing:
            desc = properties.get(p, {}).get("description", p)
            missing_desc.append(f"- {p}: {desc}")

        question = (
            f"执行 {tool_name} 操作时缺少以下必要参数:\n"
            + "\n".join(missing_desc) + "\n"
            f"用户原始请求: '{query}'。"
            f"请从对话上下文或用户信息中补充这些参数。"
            f"以JSON格式返回补充的参数值。"
        )

        reply = await self.request_info_from_main(question, timeout=15.0)
        if reply:
            try:
                cleaned = self._clean_json_response(reply)
                supplementary = json.loads(cleaned)
                if isinstance(supplementary, dict):
                    # 过滤掉"无额外信息"等无效值
                    supplementary = {
                        k: v for k, v in supplementary.items()
                        if v and v != "无额外信息" and v != "无"
                    }
                    params.update(supplementary)
            except (json.JSONDecodeError, Exception):
                self.logger.info(f"{self._log_tag} 主Agent回复非JSON: {reply[:100]}")

        return params

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行MCP工具调用

        注意：MCPManager.call_tool 内部可能使用 asyncio.get_event_loop().run_until_complete()
        （SSE协议），而我们已经在异步事件循环中。因此需要在独立线程中调用以避免嵌套循环冲突。
        """
        if not self.mcp_manager:
            return {"success": False, "error": "MCP管理器未初始化"}

        try:
            # 权限检查：只能调用白名单内的工具
            if not self.is_tool_allowed(tool_name):
                self.logger.warning(f"{self._log_tag} 工具 {tool_name} 不在权限白名单内")
                return {"success": False, "error": f"权限不足：功能处理子Agent无权调用工具 {tool_name}"}

            self.logger.info(f"{self._log_tag} 调用工具: {tool_name}, 参数: {params}")

            # 从 get_available_tools 中查找工具所在的服务器
            all_tools = self.mcp_manager.get_available_tools(use_cache=True)
            server_name = None
            for t in all_tools:
                if t.get("name") == tool_name:
                    server_name = t.get("server")
                    break

            if not server_name:
                return {"success": False, "error": f"工具 {tool_name} 不存在"}

            # 在线程池中调用，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.mcp_manager.call_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    parameters=params,
                )
            )

            # 检查返回结果是否包含 error
            if isinstance(result, dict) and "error" in result:
                return {"success": False, "error": result["error"]}

            self.logger.info(f"{self._log_tag} 工具调用成功: {tool_name}")
            return {"success": True, "result": result}

        except Exception as e:
            self.logger.error(f"{self._log_tag} 工具调用失败: {tool_name} - {e}")
            return {"success": False, "error": str(e)}

    def _format_result(
        self, query: str, tool_name: str, params: Dict[str, Any], result: Dict[str, Any]
    ) -> str:
        """格式化工具执行结果为用户友好的回答"""
        if not result.get("success"):
            error = result.get("error", "未知错误")
            return f"操作执行失败: {error}"

        tool_result = result.get("result", "")
        if isinstance(tool_result, dict):
            tool_result = tool_result.get("result", str(tool_result))

        # 使用LLM将工具结果转化为自然语言回答
        prompt = f"""将以下工具执行结果转化为面向用户的友好回答。

用户请求：{query}
执行的操作：{tool_name}
执行结果：{tool_result}

要求：
1. 用自然语言回答，不要暴露工具名
2. 突出关键信息（ID、状态、数据等）
3. 简洁明了"""

        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.3)
            return response.strip()
        except Exception:
            return str(tool_result)

    def _get_available_tools(self) -> str:
        """获取可用工具列表描述"""
        if not self.mcp_manager:
            return "无可用工具"

        all_tools = self.mcp_manager.get_available_tools(use_cache=True)
        # 只暴露本子Agent有权使用的工具
        allowed_tools = self.filter_tools(all_tools)
        lines = []
        for t in allowed_tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
