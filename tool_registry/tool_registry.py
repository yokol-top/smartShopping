"""
工具层 (Tool Registry)

类似Java注册中心的设计：
1. 工具注册 / 卸载
2. 工具发现：关键词粗过滤 + LLM语义精过滤
3. MCP工具懒加载：查询时只返回名称+描述，使用时才加载完整schema
4. 本地工具（如search_knowledge）与MCP工具统一管理
"""
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from observability import get_tracer


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    description: str
    source: str                          # "local" / "mcp"
    server_name: str = ""                # MCP所属服务名
    keywords: List[str] = field(default_factory=list)  # 粗过滤关键词
    input_schema: Optional[Dict] = None  # 完整schema（懒加载时可能为None）
    handler: Optional[Callable] = None   # 本地工具的执行函数
    enabled: bool = True

    def brief(self) -> Dict[str, str]:
        """返回轻量摘要（名称+描述），用于减少上下文token"""
        return {"name": self.name, "description": self.description, "source": self.source}

    def full(self) -> Dict[str, Any]:
        """返回完整信息（含schema）"""
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "server_name": self.server_name,
            "inputSchema": self.input_schema or {},
            "keywords": self.keywords,
        }


class ToolRegistry:
    """
    统一工具注册中心

    管理本地工具和MCP工具，提供分级工具发现能力：
    - Level 1 (粗过滤): 关键词匹配，快速缩小候选范围
    - Level 2 (精过滤): LLM语义匹配，从候选中选出最佳工具
    - MCP懒加载: 查询阶段仅返回名称+描述，实际调用时才加载完整schema
    """

    def __init__(self, config: Dict[str, Any], mcp_manager=None,
                 llm_client=None, logger: logging.Logger = None):
        self.config = config
        self.mcp_manager = mcp_manager
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

        # 工具注册表：name → ToolInfo
        self._registry: Dict[str, ToolInfo] = {}
        # MCP工具schema缓存（懒加载）
        self._mcp_schema_cache: Dict[str, Dict] = {}

        # 注册内置工具
        self._register_builtin_tools()
        # 从MCP加载工具摘要
        self._sync_mcp_tools()

        self.logger.info(f"ToolRegistry 初始化完成 | 已注册工具: {len(self._registry)}")

    # ================================================================
    # 工具注册 / 卸载
    # ================================================================
    def register(self, tool: ToolInfo):
        """注册工具"""
        self._registry[tool.name] = tool
        self.logger.info(f"[工具注册] {tool.name} ({tool.source})")

    def unregister(self, tool_name: str):
        """卸载工具"""
        if tool_name in self._registry:
            del self._registry[tool_name]
            self.logger.info(f"[工具卸载] {tool_name}")
        else:
            self.logger.warning(f"[工具卸载] 工具不存在: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[ToolInfo]:
        """获取工具信息"""
        return self._registry.get(tool_name)

    def list_tools_brief(self) -> List[Dict[str, str]]:
        """列出所有工具的轻量摘要（名称+描述）"""
        return [t.brief() for t in self._registry.values() if t.enabled]

    def list_tools_full(self) -> List[Dict[str, Any]]:
        """列出所有工具的完整信息"""
        result = []
        for t in self._registry.values():
            if not t.enabled:
                continue
            # 如果MCP工具还没加载schema，先加载
            if t.source == "mcp" and t.input_schema is None:
                self._load_mcp_schema(t.name)
            result.append(t.full())
        return result

    # ================================================================
    # 工具发现（分级过滤）
    # ================================================================
    def find_tools(self, query: str, top_k: int = 5) -> List[ToolInfo]:
        """
        工具发现主入口：关键词粗过滤 → LLM语义精过滤

        Args:
            query: 用户需求描述
            top_k: 返回的最大工具数

        Returns:
            匹配的工具列表（按相关度排序）
        """
        tracer = get_tracer()
        ctx = tracer.start_span("tool_registry.find_tools", {
            "query": query, "total_tools": len(self._registry),
        }) if tracer else _noop()

        with ctx:
            # Level 1: 关键词粗过滤
            candidates = self._keyword_filter(query)
            self.logger.info(f"[工具发现] 关键词粗过滤: {len(candidates)} 个候选")

            if not candidates:
                # 关键词未命中任何工具，返回全量供LLM选择
                candidates = [t for t in self._registry.values() if t.enabled]

            if len(candidates) <= top_k:
                return candidates

            # Level 2: LLM语义精过滤
            if self.llm_client:
                selected = self._semantic_filter(query, candidates, top_k)
                self.logger.info(f"[工具发现] LLM语义精过滤: {len(selected)} 个最终选择")
                return selected

            return candidates[:top_k]

    def _keyword_filter(self, query: str) -> List[ToolInfo]:
        """Level 1: 关键词粗过滤"""
        query_lower = query.lower()
        matched = []
        for tool in self._registry.values():
            if not tool.enabled:
                continue
            # 匹配工具名、描述、关键词
            searchable = f"{tool.name} {tool.description} {' '.join(tool.keywords)}".lower()
            # 双向匹配：查询词出现在工具信息中，或工具关键词出现在查询中
            if any(kw in query_lower for kw in tool.keywords if kw):
                matched.append(tool)
            elif any(word in searchable for word in query_lower.split() if len(word) > 1):
                matched.append(tool)
        return matched

    def _semantic_filter(self, query: str, candidates: List[ToolInfo],
                         top_k: int) -> List[ToolInfo]:
        """Level 2: LLM语义精过滤"""
        tools_desc = "\n".join(
            f"{i+1}. {t.name}: {t.description}"
            for i, t in enumerate(candidates)
        )

        prompt = f"""你是一个工具选择助手。根据用户需求，从候选工具中选出最相关的工具。

用户需求: {query}

候选工具:
{tools_desc}

请返回最相关的工具名称列表（最多{top_k}个），按相关度从高到低排列。
只返回JSON数组，例如: ["tool_a", "tool_b"]
"""
        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.1)
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            selected_names = json.loads(response.strip())

            name_to_tool = {t.name: t for t in candidates}
            return [name_to_tool[n] for n in selected_names if n in name_to_tool][:top_k]
        except Exception as e:
            self.logger.warning(f"LLM语义过滤失败: {e}，返回关键词过滤结果")
            return candidates[:top_k]

    # ================================================================
    # MCP 懒加载
    # ================================================================
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """
        获取工具完整schema（MCP工具会按需加载）

        Args:
            tool_name: 工具名称

        Returns:
            inputSchema字典，未找到返回None
        """
        tool = self._registry.get(tool_name)
        if not tool:
            return None

        if tool.source == "mcp" and tool.input_schema is None:
            self._load_mcp_schema(tool_name)

        return tool.input_schema

    def _load_mcp_schema(self, tool_name: str):
        """懒加载MCP工具的完整schema"""
        if not self.mcp_manager:
            return

        tool = self._registry.get(tool_name)
        if not tool or tool.source != "mcp":
            return

        # 从MCP管理器获取完整工具列表（含schema）
        all_tools = self.mcp_manager.get_available_tools(use_cache=True)
        for mcp_tool in all_tools:
            if mcp_tool['name'] == tool_name:
                tool.input_schema = mcp_tool.get('inputSchema', {})
                self._mcp_schema_cache[tool_name] = tool.input_schema
                self.logger.info(f"[懒加载] 已加载MCP工具schema: {tool_name}")
                return

        self.logger.warning(f"[懒加载] MCP工具未找到schema: {tool_name}")

    def find_tool_server(self, tool_name: str) -> Optional[str]:
        """查找工具所属的MCP服务名"""
        tool = self._registry.get(tool_name)
        if tool and tool.source == "mcp":
            return tool.server_name
        return None

    # ================================================================
    # 内部初始化
    # ================================================================
    def _register_builtin_tools(self):
        """注册内置工具"""
        self.register(ToolInfo(
            name="search_knowledge",
            description="从知识库中搜索相关信息",
            source="local",
            keywords=["搜索", "查询", "查找", "知识", "文档", "检索", "search", "query", "find"],
        ))
        self.register(ToolInfo(
            name="finish",
            description="完成任务并返回最终答案",
            source="local",
            keywords=["完成", "结束", "返回", "finish", "done"],
        ))

    def _sync_mcp_tools(self):
        """从MCP管理器同步工具摘要（仅名称+描述，不含schema）"""
        if not self.mcp_manager:
            return

        try:
            all_tools = self.mcp_manager.get_available_tools(use_cache=True)
            for mcp_tool in all_tools:
                name = mcp_tool.get('name', '')
                if not name or name in self._registry:
                    continue
                # 从描述中提取关键词
                desc = mcp_tool.get('description', '')
                keywords = self._extract_keywords(name, desc)

                self.register(ToolInfo(
                    name=name,
                    description=desc,
                    source="mcp",
                    server_name=mcp_tool.get('server', ''),
                    keywords=keywords,
                    input_schema=None,  # 懒加载，不立即加载schema
                ))
            self.logger.info(f"[MCP同步] 已同步 {len(all_tools)} 个MCP工具摘要")
        except Exception as e:
            self.logger.error(f"同步MCP工具失败: {e}")

    def refresh_mcp_tools(self):
        """刷新MCP工具列表"""
        # 先清除现有MCP工具
        mcp_names = [n for n, t in self._registry.items() if t.source == "mcp"]
        for name in mcp_names:
            del self._registry[name]
        self._mcp_schema_cache.clear()
        # 重新同步
        self._sync_mcp_tools()

    @staticmethod
    def _extract_keywords(name: str, description: str) -> List[str]:
        """从工具名和描述中提取关键词"""
        keywords = []
        # 从工具名提取（驼峰/下划线分割）
        import re
        parts = re.split(r'[_\-]', name)
        for part in parts:
            # 驼峰拆分
            sub_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', part)
            keywords.extend(p.lower() for p in sub_parts if len(p) > 1)
            if part.lower() not in keywords:
                keywords.append(part.lower())

        # 从描述中提取关键实体词
        desc_keywords = re.findall(r'[\u4e00-\u9fff]{2,}', description)  # 中文词
        desc_keywords += re.findall(r'[a-zA-Z]{3,}', description)  # 英文词
        keywords.extend(w.lower() for w in desc_keywords[:10])

        return list(set(keywords))


class _noop:
    """空上下文管理器"""
    def __enter__(self): return self
    def __exit__(self, *a): pass
