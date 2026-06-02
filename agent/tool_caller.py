"""
工具调用器 (ToolCaller)

从 TaskPlanner 中提取的 MCP 工具执行逻辑，职责单一：
- 查找工具归属服务器
- 获取并压缩工具 Schema
- LLM 辅助参数提取
- 实际工具调用（MCP + 本地工具）
"""

import json
import logging
from typing import Dict, Any, List, Optional


class ToolCaller:
    """MCP/本地工具调用器

    TaskPlanner 通过 self._tool_caller 委托所有工具执行操作，
    自身只负责计划生成和步骤编排。
    """

    def __init__(
        self,
        llm_client,
        mcp_manager=None,
        tool_manager=None,
        context_manager=None,
        logger: logging.Logger = None,
    ):
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.tool_manager = tool_manager
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)

    # ================================================================
    # 工具调用主入口
    # ================================================================

    def call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """执行工具调用（MCP 或本地工具）

        Returns:
            统一格式字符串："工具执行成功: {结果}" 或 "工具执行失败: {原因}"
        """
        # 频率限制检查
        if self.tool_manager and self.tool_manager.rate_limiter:
            check = self.tool_manager.rate_limiter.check(tool_name)
            if not check.allowed:
                self.logger.warning(f"工具调用被频率限制: {check.reason}")
                return f"工具执行失败: {check.reason}"

        try:
            # 优先本地工具
            if self.tool_manager:
                tool_info = self.tool_manager.get_tool(tool_name)
                if tool_info and tool_info.source == "local":
                    self.logger.info(f"调用本地工具: {tool_name}")
                    result = self.tool_manager.call_local_tool(tool_name, parameters)
                    return f"工具执行成功: {json.dumps(result, ensure_ascii=False)}"

            # MCP 工具
            server_name = self.find_server(tool_name)
            if not server_name:
                return f"工具执行失败: 找不到工具 {tool_name} 所属的服务"
            self.logger.info(f"调用MCP - 服务: {server_name}, 工具: {tool_name}")
            result = self.mcp_manager.call_tool(
                server_name=server_name, tool_name=tool_name, parameters=parameters
            )
            if isinstance(result, dict) and 'error' in result:
                return f"工具执行失败: {result['error']}"

            if self.tool_manager:
                self.tool_manager.record_call(tool_name)

            # 提取 MCP 返回值中的 result 字段（自然语言内容），避免把原始 dict 暴露给用户
            result_text = result.get('result', json.dumps(result, ensure_ascii=False)) if isinstance(result, dict) else str(result)
            return f"工具执行成功: {result_text}"
        except Exception as e:
            self.logger.error(f"工具调用异常 [{tool_name}]: {e}")
            return f"工具执行失败: {str(e)}"

    def extract_params_and_call(
        self,
        tool_name: str,
        step_description: str,
        user_query: str,
        context: str,
        long_term_context: str,
        execution_history: list,
        verbose: bool = False,
    ) -> str:
        """使用 LLM 从上下文中提取参数，然后调用工具（懒加载 Schema）"""
        schema = self.get_schema(tool_name)
        if not schema:
            return f"未找到工具 {tool_name} 的schema"

        schema_json = json.dumps(self.compress_schema(schema), ensure_ascii=False)
        hist = "\n".join(
            f"- {h['description']}: {h.get('result', '')[:500]}"
            for h in execution_history
        )

        # 上下文窗口预算分配
        if self.context_manager:
            active = ['short_term_memory', 'tool_results']
            if long_term_context:
                active.append('long_term_memory')
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section('short_term_memory', context)
            long_term_context = self.context_manager.manage_section('long_term_memory', long_term_context)
            if hist:
                hist = self.context_manager.manage_section('tool_results', hist)

        prompt_parts = []
        if long_term_context:
            prompt_parts.append(long_term_context)
        prompt_parts.append(f"""你是一个工具调用助手。请根据当前步骤的目标，从上下文中提取工具参数。

当前步骤目标: {step_description}

用户请求: {user_query}
对话历史: {context}
前置步骤结果:
{hist if hist else "（无）"}

工具名: {tool_name}
工具参数结构（JSON Schema）:
{schema_json}

**关键规则**:
1. 只提取与"当前步骤目标"相关的参数。
2. 【最重要】参数名必须与"工具参数结构（JSON Schema）"中定义的字段名完全一致，禁止使用上下文中的实体类型名（如 addr_id、type 等）作为参数名。
   - 例如：schema 定义了 address_id，就必须用 address_id，不得写成 addr_id
   - 例如：schema 定义了 customer_name，就必须用 customer_name，不得写成 username
3. 从上下文中查找值时的映射规则：
   - address_id → 查找 ADDR-xxx 格式的ID（来自地址信息）
   - card_id     → 查找 CARD-xxx 格式的ID（来自银行卡信息）
   - customer_name → 查找用户名（username / 收货人姓名）
   - user_id     → 查找 UID-xxx 格式的ID
4. 如果前置步骤结果中包含了本次调用所需的动态值（如用户ID、订单号等），必须使用前置步骤返回的实际值，不要编造或使用默认值。
5. 参数优先级：前置步骤结果 > 用户请求中的信息 > 默认值。
6. 根据上下文智能推断操作类型：如果schema中包含操作类型字段（如action: add/update/delete），需根据实际情况选择。
7. 如果某个必需的ID字段在上下文中不存在，说明应使用创建/新增操作而非更新操作。
8. quantity（数量）若上下文未明确说明，默认填 1。

只返回JSON。""")

        try:
            param_json = self.llm_client.generate(
                prompt="\n".join(prompt_parts), temperature=0.3
            )
            # 复用 TaskPlanner 的 JSON 清理逻辑
            cleaned = self._clean_json(param_json)
            parameters = json.loads(cleaned)
            if verbose:
                print(f"📋 提取参数: {json.dumps(parameters, ensure_ascii=False, indent=2)}\n")
            return self.call(tool_name, parameters)
        except Exception as e:
            return f"参数提取失败: {e}"

    # ================================================================
    # Schema 工具
    # ================================================================

    def find_server(self, tool_name: str) -> Optional[str]:
        """查找工具所属的服务器名称"""
        if self.tool_manager:
            return self.tool_manager.find_tool_server(tool_name)
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_available_tools(use_cache=True):
            if tool.get('name') == tool_name:
                return tool.get('server')
        return None

    def get_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具的 Input Schema"""
        if not self.mcp_manager:
            return None
        for tool in self.mcp_manager.get_available_tools(use_cache=True):
            if tool.get('name') == tool_name:
                return tool.get('inputSchema', {})
        return None

    @staticmethod
    def compress_schema(schema: dict) -> dict:
        """压缩 Schema，去除非必要字段以节省 token"""
        if not schema:
            return {}
        result = {}
        if 'properties' in schema:
            result['properties'] = {
                k: ToolCaller.compress_property(v)
                for k, v in schema['properties'].items()
            }
        if 'required' in schema:
            result['required'] = schema['required']
        return result

    @staticmethod
    def compress_property(prop: dict) -> dict:
        """压缩单个属性定义"""
        if not prop:
            return {}
        compressed = {}
        for key in ('type', 'description', 'enum', 'items', 'properties', 'required'):
            if key in prop:
                val = prop[key]
                if key == 'properties' and isinstance(val, dict):
                    compressed[key] = {
                        k: ToolCaller.compress_property(v) for k, v in val.items()
                    }
                elif key == 'items' and isinstance(val, dict):
                    compressed[key] = ToolCaller.compress_property(val)
                else:
                    compressed[key] = val
        return compressed

    # ================================================================
    # 内部工具
    # ================================================================

    @staticmethod
    def _clean_json(response: str) -> str:
        """从 LLM 响应中提取第一个完整 JSON 对象"""
        import re
        response = response.strip()
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_block:
            response = code_block.group(1).strip()
        brace_start = response.find('{')
        if brace_start >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(brace_start, len(response)):
                c = response[i]
                if esc:
                    esc = False
                    continue
                if c == '\\':
                    esc = True
                    continue
                if c == '"' and not esc:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return response[brace_start:i + 1]
        return response
