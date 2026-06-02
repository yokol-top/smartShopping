import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Any, List, Optional

import requests
import yaml
from mcp import ClientSession
from mcp.client.sse import sse_client

from observability import get_tracer


class MCPManager:
    """MCP (Model Context Protocol) 管理器"""

    @staticmethod
    def _run_async(coro):
        """在同步上下文中运行async协程，兼容已有事件循环（如uvicorn）的场景"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 已有事件循环在跑（uvicorn / jupyter 等），在新线程中执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def __init__(self, config_file: str, logger: logging.Logger = None):
        self.config_file = config_file
        self.logger = logger or logging.getLogger(__name__)
        
        self.servers = []
        self.enabled_servers = []
        self._server_tools_cache = {}  # 缓存从MCP服务器获取的工具列表
        self._server_info_cache = {}  # 缓存从MCP服务器获取的服务器信息
        self._tool_cat_cache: Dict[str, List[str]] = {}  # 工具名 → 类别列表

        self._load_config()
        
        self.logger.info(f"初始化MCP管理器，启用 {len(self.enabled_servers)} 个服务")
    
    def _load_config(self):
        """加载MCP配置"""
        if not os.path.exists(self.config_file):
            self.logger.warning(f"MCP配置文件不存在: {self.config_file}")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.servers = config.get('servers', [])
            self.enabled_servers = [s for s in self.servers if s.get('enabled', False)]
            
            self.logger.info(f"加载 {len(self.servers)} 个MCP服务配置，其中 {len(self.enabled_servers)} 个已启用")
            
            for server in self.enabled_servers:
                self.logger.info(f"  - {server['name']}: {server.get('description', 'No description')}")
        except Exception as e:
            self.logger.error(f"加载MCP配置失败: {e}")
    
    def reload_config(self):
        """重新加载配置"""
        self.logger.info("重新加载MCP配置")
        self._load_config()
    
    def get_enabled_servers(self) -> List[Dict[str, Any]]:
        """获取已启用的服务列表"""
        return self.enabled_servers
    
    def get_server_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取服务配置"""
        for server in self.enabled_servers:
            if server['name'] == name:
                return server
        return None
    
    async def _fetch_server_info_and_tools_sse(self, endpoint: str) -> Dict[str, Any]:
        """通过SSE协议从MCP服务器获取服务器信息和工具列表"""
        try:
            async with sse_client(endpoint) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化并获取服务器信息
                    init_result = await session.initialize()
                    
                    # 提取服务器信息
                    server_info = {
                        'name': init_result.serverInfo.name if hasattr(init_result, 'serverInfo') else '',
                        'version': init_result.serverInfo.version if hasattr(init_result, 'serverInfo') else ''
                    }
                    
                    # 获取工具列表
                    tools_result = await session.list_tools()
                    
                    # 转换为标准格式
                    tools = []
                    for tool in tools_result.tools:
                        tools.append({
                            'name': tool.name,
                            'description': tool.description if hasattr(tool, 'description') else '',
                            'inputSchema': tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        })
                    
                    return {
                        'server_info': server_info,
                        'tools': tools
                    }
        except Exception as e:
            self.logger.error(f"从MCP服务器获取信息失败: {e}")
            return {'server_info': {}, 'tools': []}
    
    def _fetch_server_info_and_tools(self, server_name: str, endpoint: str) -> Dict[str, Any]:
        """从MCP服务器获取服务器信息和工具列表"""
        try:
            # 检测是否为SSE端点
            if '/sse' in endpoint.lower():
                # 使用SSE客户端（asyncio.run在当前线程创建新事件循环）
                result = self._run_async(
                    asyncio.wait_for(
                        self._fetch_server_info_and_tools_sse(endpoint),
                        timeout=10
                    )
                )
                return result
            else:
                # 先获取工具列表
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tools/list",
                    "params": {}
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                response = requests.post(
                    endpoint,
                    json=rpc_request,
                    headers=headers,
                    timeout=10,
                    stream=False
                )
                
                response.raise_for_status()
                rpc_response = response.json()
                
                if "error" in rpc_response:
                    self.logger.error(f"MCP错误: {rpc_response['error']}")
                    return {'server_info': {}, 'tools': []}
                
                result = rpc_response.get("result", {})
                tools = result.get("tools", [])
                
                # HTTP JSON-RPC不返回服务器信息，只返回工具列表
                return {
                    'server_info': {},
                    'tools': tools
                }
        except Exception as e:
            self.logger.error(f"从服务器 {server_name} 获取信息失败: {e}")
            return {'server_info': {}, 'tools': []}
    
    def get_available_tools(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        获取所有可用工具的详细信息（从MCP服务器动态获取）
        
        Args:
            use_cache: 是否使用缓存的工具列表
        
        Returns:
            工具信息列表，每个工具包含: name, server, description, inputSchema
        """
        tools = []
        for server in self.enabled_servers:
            server_name = server['name']
            endpoint = server.get('endpoint')
            
            if not endpoint:
                self.logger.warning(f"服务 {server_name} 未配置端点")
                continue
            
            # 检查缓存
            if use_cache and server_name in self._server_tools_cache:
                tool_details = self._server_tools_cache[server_name]
            else:
                # 从服务器获取信息和工具列表
                result = self._fetch_server_info_and_tools(server_name, endpoint)
                self._server_info_cache[server_name] = result['server_info']
                self._server_tools_cache[server_name] = result['tools']
                tool_details = result['tools']
                
                self.logger.info(f"从服务器 {server_name} 获取到 {len(tool_details)} 个工具")
            
            # 构建工具信息
            for tool in tool_details:
                tools.append({
                    'name': tool['name'],
                    'server': server_name,
                    'description': tool.get('description', ''),
                    'inputSchema': tool.get('inputSchema', {})
                })
        
        return tools

    # ================================================================
    # 按阶段/意图过滤工具
    # ================================================================

    # 工具类别关键词规则（工具名或描述中包含这些词则归入对应类别）
    _TOOL_CATEGORY_RULES: Dict[str, Dict[str, List[str]]] = {
        'search': {
            'name': ['search', 'knowledge'],
            'desc': ['搜索', '检索', '知识库', 'faq', '商品信息', 'search'],
        },
        'order': {
            'name': ['order', 'create_complex'],
            'desc': ['下单', '订单', '购买', '创建订单', '批量创建'],
        },
        'user': {
            'name': ['user', 'profile'],
            'desc': ['用户', '地址', '银行卡', '创建用户', '修改用户', '查询用户'],
        },
    }

    # 意图类型 → 允许的工具类别（None 表示返回全量，[] 表示不需要工具）
    _INTENT_CATEGORIES: Dict[str, Optional[List[str]]] = {
        'greeting': [],
        'simple_chat': [],
        'rag_simple': ['search'],
        'rag_advanced': ['search'],
        'mcp_ask_info': None,
    }

    def _classify_tool(self, tool: Dict[str, Any]) -> List[str]:
        """根据工具名称和描述关键词将工具归入一个或多个类别"""
        name = tool.get('name', '').lower()
        desc = tool.get('description', '').lower()
        categories = []
        for cat, rules in self._TOOL_CATEGORY_RULES.items():
            if any(k in name for k in rules['name']) or any(k in desc for k in rules['desc']):
                categories.append(cat)
        return categories or ['other']

    def _ensure_tool_cat_cache(self, all_tools: List[Dict[str, Any]]) -> None:
        """（惰性）构建工具类别缓存"""
        if self._tool_cat_cache:
            return
        for tool in all_tools:
            name = tool.get('name', '')
            if name:
                self._tool_cat_cache[name] = self._classify_tool(tool)

    def _infer_categories_from_query(self, query: str) -> List[str]:
        """根据查询关键词推断需要哪些工具类别（用于 mcp_execute 意图）"""
        q = query.lower()
        cats: set = {'search'}  # 搜索工具作为安全基线

        order_kws = ['下单', '订单', '购买', '买', 'order', '创建订单', '支付', '结算', '确认下单']
        user_kws = ['用户', '地址', '银行卡', '账户', '收货', '我的信息', '登录', '注册']
        search_kws = ['搜索', '推荐', '商品', 'search', '找', '哪款', '对比', '查找']

        if any(kw in q for kw in order_kws):
            cats.add('order')
            cats.add('user')   # 下单操作通常需要用户信息（地址/银行卡）
        if any(kw in q for kw in user_kws):
            cats.add('user')
        if any(kw in q for kw in search_kws):
            cats.add('search')

        return list(cats)

    def get_tools_for_context(
        self,
        intent_type: str = None,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """按当前阶段/意图返回过滤后的工具列表。

        - 导购阶段（rag_simple/rag_advanced）：只返回搜索类工具
        - 问候/闲聊：不返回工具
        - mcp_execute：根据 query 关键词推断所需类别
        - None 或未知意图：返回全量（安全兜底）

        Args:
            intent_type: 意图类型字符串（如 'rag_simple'、'mcp_execute'）
            query: 用户查询文本，用于 mcp_execute 时进一步推断

        Returns:
            过滤后的工具列表
        """
        all_tools = self.get_available_tools(use_cache=True)
        if not all_tools:
            return []

        # 构建/复用工具类别缓存
        self._ensure_tool_cat_cache(all_tools)

        # 确定允许的类别
        if intent_type in self._INTENT_CATEGORIES:
            allowed = self._INTENT_CATEGORIES[intent_type]
        elif intent_type == 'mcp_execute':
            allowed = self._infer_categories_from_query(query)
        else:
            return all_tools  # 未知意图，返回全量

        if allowed is None:
            return all_tools  # None = 全量

        if not allowed:
            return []  # 空列表 = 无需工具

        # 按类别过滤
        filtered = [
            t for t in all_tools
            if any(cat in allowed for cat in self._tool_cat_cache.get(t.get('name', ''), ['other']))
        ]

        # 兜底：过滤结果为空时返回全量
        if not filtered:
            self.logger.warning(
                f"[MCPManager] 按意图 '{intent_type}' 过滤后工具为空，回退到全量"
            )
            return all_tools

        self.logger.debug(
            f"[MCPManager] get_tools_for_context | intent={intent_type} | "
            f"total={len(all_tools)} → filtered={len(filtered)}"
        )
        return filtered

    async def _call_tool_sse(
        self,
        endpoint: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """通过SSE协议调用MCP工具"""
        try:
            async with sse_client(endpoint) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # 调用工具
                    result = await session.call_tool(tool_name, arguments=parameters)
                    
                    # 提取结果
                    if result.content and len(result.content) > 0:
                        return {"result": result.content[0].text}
                    return {"result": "执行成功，无返回内容"}
        except Exception as e:
            raise e
    
    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        调用MCP工具

        Args:
            server_name: 服务名称
            tool_name: 工具名称
            parameters: 参数
            timeout: 超时时间

        Returns:
            工具执行结果
        """
        tracer = get_tracer()

        with tracer.start_span("mcp.call_tool", {
            "mcp.server_name": server_name,
            "mcp.tool_name": tool_name,
            "mcp.parameters": json.dumps(parameters, ensure_ascii=False)[:500],
        }):
            self.logger.info(f"调用MCP工具 - 服务: {server_name}, 工具: {tool_name}")
            self.logger.info(f"MCP调用入参: {parameters}")

            server = self.get_server_by_name(server_name)
            if not server:
                error_msg = f"服务未找到或未启用: {server_name}"
                self.logger.error(error_msg)
                tracer.set_span_attributes({"mcp.error": error_msg})
                return {"error": error_msg}

            # 验证工具是否存在（从缓存或服务器获取）
            server_tools = self._server_tools_cache.get(server_name)
            if server_tools is None:
                # 尝试从服务器获取信息和工具列表
                endpoint = server.get('endpoint')
                if endpoint:
                    result = self._fetch_server_info_and_tools(server_name, endpoint)
                    self._server_info_cache[server_name] = result['server_info']
                    self._server_tools_cache[server_name] = result['tools']
                    server_tools = result['tools']

            if server_tools is not None:
                tool_names = [t['name'] for t in server_tools]
                if tool_name not in tool_names:
                    error_msg = f"工具不存在: {tool_name}（可用工具: {', '.join(tool_names)}）"
                    self.logger.error(error_msg)
                    tracer.set_span_attributes({"mcp.error": error_msg})
                    return {"error": error_msg}

            endpoint = server.get('endpoint')
            if not endpoint:
                error_msg = f"服务端点未配置: {server_name}"
                self.logger.error(error_msg)
                tracer.set_span_attributes({"mcp.error": error_msg})
                return {"error": error_msg}

            try:
                # 检测是否为SSE端点
                if '/sse' in endpoint.lower():
                    # 使用SSE客户端（asyncio.run在当前线程创建新事件循环）
                    result = self._run_async(
                        asyncio.wait_for(
                            self._call_tool_sse(endpoint, tool_name, parameters),
                            timeout=timeout
                        )
                    )
                    self.logger.info(f"MCP工具调用成功: {tool_name}")
                    self.logger.info(f"MCP响应结果: {result}")
                    tracer.set_span_attributes({"mcp.result": json.dumps(result, ensure_ascii=False)[:500]})
                    tracer.set_span_ok()
                    return result
                else:
                    # 使用标准HTTP JSON-RPC
                    rpc_request = {
                        "jsonrpc": "2.0",
                        "id": str(uuid.uuid4()),
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": parameters
                        }
                    }

                    self.logger.debug(f"发送MCP请求到 {endpoint}: {rpc_request}")

                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }

                    response = requests.post(
                        endpoint,
                        json=rpc_request,
                        headers=headers,
                        timeout=timeout,
                        stream=False
                    )

                    response.raise_for_status()
                    rpc_response = response.json()

                    if "error" in rpc_response:
                        error_msg = f"MCP错误: {rpc_response['error']}"
                        self.logger.error(error_msg)
                        tracer.set_span_attributes({"mcp.error": error_msg})
                        return {"error": error_msg}

                    result = rpc_response.get("result", {})
                    self.logger.info(f"MCP工具调用成功: {tool_name}")
                    self.logger.info(f"MCP响应结果: {result}")
                    tracer.set_span_attributes({"mcp.result": json.dumps(result, ensure_ascii=False)[:500]})
                    tracer.set_span_ok()
                    return result
            except asyncio.TimeoutError:
                error_msg = f"MCP调用超时: {server_name}.{tool_name}"
                self.logger.error(error_msg)
                tracer.set_span_attributes({"mcp.error": error_msg})
                return {"error": error_msg}
            except requests.exceptions.Timeout:
                error_msg = f"MCP调用超时: {server_name}.{tool_name}"
                self.logger.error(error_msg)
                tracer.set_span_attributes({"mcp.error": error_msg})
                return {"error": error_msg}
            except requests.exceptions.RequestException as e:
                error_msg = f"MCP调用失败: {e}"
                self.logger.error(error_msg)
                tracer.record_exception(e)
                return {"error": error_msg}
            except Exception as e:
                error_msg = f"MCP调用异常: {e}"
                self.logger.error(error_msg)
                tracer.record_exception(e)
                return {"error": error_msg}
    
    def is_enabled(self) -> bool:
        """检查MCP功能是否启用"""
        return len(self.enabled_servers) > 0
    
    def add_server(
        self,
        name: str,
        endpoint: str,
        description: str = "",
        enabled: bool = True
    ):
        """
        添加新的MCP服务（工具列表将自动从服务器获取）
        
        Args:
            name: 服务名称
            endpoint: 服务端点
            description: 描述
            enabled: 是否启用
        """
        server_config = {
            "name": name,
            "enabled": enabled,
            "endpoint": endpoint,
            "description": description
        }
        
        # 检查是否已存在
        existing = self.get_server_by_name(name)
        if existing:
            self.logger.warning(f"服务已存在，将更新配置: {name}")
            self.servers = [s for s in self.servers if s['name'] != name]
        
        self.servers.append(server_config)
        
        if enabled:
            self.enabled_servers.append(server_config)
        
        self._save_config()
        
        self.logger.info(f"添加MCP服务: {name}")
    
    def get_server_info(self, server_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取服务器信息（从MCP服务器动态获取）
        
        Args:
            server_name: 服务名称
            use_cache: 是否使用缓存
        
        Returns:
            服务器信息，包含name, version等
        """
        server = self.get_server_by_name(server_name)
        if not server:
            return {}
        
        endpoint = server.get('endpoint')
        if not endpoint:
            return {}
        
        # 检查缓存
        if use_cache and server_name in self._server_info_cache:
            return self._server_info_cache[server_name]
        
        # 从服务器获取信息
        result = self._fetch_server_info_and_tools(server_name, endpoint)
        self._server_info_cache[server_name] = result['server_info']
        self._server_tools_cache[server_name] = result['tools']
        
        return result['server_info']
    
    def remove_server(self, name: str):
        """删除MCP服务"""
        self.servers = [s for s in self.servers if s['name'] != name]
        self.enabled_servers = [s for s in self.enabled_servers if s['name'] != name]
        
        # 清除缓存
        if name in self._server_tools_cache:
            del self._server_tools_cache[name]
        if name in self._server_info_cache:
            del self._server_info_cache[name]
        
        self._save_config()
        
        self.logger.info(f"删除MCP服务: {name}")
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            config = {"servers": self.servers}
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(f"MCP配置已保存: {self.config_file}")
        except Exception as e:
            self.logger.error(f"保存MCP配置失败: {e}")
