"""
工具管理器（Tool Manager）

类似服务注册中心的设计，统一管理本地工具与 MCP 远程工具：

1. 注册与生命周期
   - 本地工具：通过 register_local_tool() 注册 Python 可调用对象
   - MCP 工具：通过 sync_mcp_tools() 从 MCPManager 批量导入
   - 支持卸载（unregister）、启用/禁用、更新

2. 自动分类与关键词提取
   - 注册时根据工具名称和描述自动归类（order/user/product/address/card/query/...）
   - 提取中英文关键词，用于后续粗过滤

3. 多层过滤（get_tools_for_query）
   - Layer 1: 关键词/分类粗过滤 —— 基于用户查询的关键词匹配与分类匹配
   - Layer 2: 调用频率过滤 —— 集成 RateLimiter，排除已达限额的工具
   - Layer 3: 返回精简工具列表给 LLM（仅名称+描述）

4. 懒加载 Schema
   - 选择阶段仅暴露 name + description，大幅节省上下文 token
   - 参数填充阶段（调用时）才返回完整 inputSchema
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable


class ToolCategory(str, Enum):
    """工具分类"""
    ORDER = "order"          # 订单相关
    USER = "user"            # 用户信息
    PRODUCT = "product"      # 商品/产品
    ADDRESS = "address"      # 地址
    CARD = "card"            # 银行卡/支付
    QUERY = "query"          # 查询类
    MANAGE = "manage"        # 管理/运维
    GENERAL = "general"      # 通用


@dataclass
class ToolInfo:
    """工具元信息"""
    name: str
    description: str
    source: str                          # "local" | "mcp"
    server_name: str = ""                # MCP 工具所属服务器
    category: ToolCategory = ToolCategory.GENERAL
    keywords: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    call_count: int = 0                  # 累计调用次数

    # 仅本地工具使用
    handler: Optional[Callable] = field(default=None, repr=False)


# ================================================================
# 分类与关键词规则（轻量级，无需 LLM）
# ================================================================

_CATEGORY_RULES: List[tuple] = [
    (ToolCategory.ORDER,   ["order", "订单", "下单", "购买", "退单"]),
    (ToolCategory.USER,    ["user", "用户", "login", "注册", "登录", "profile", "个人"]),
    (ToolCategory.PRODUCT, ["product", "商品", "搜索商品", "browse", "catalog", "库存"]),
    (ToolCategory.ADDRESS, ["address", "地址", "收货", "addr"]),
    (ToolCategory.CARD,    ["card", "银行卡", "支付", "payment", "pay"]),
    (ToolCategory.QUERY,   ["query", "search", "list", "get", "查询", "查看", "搜索"]),
    (ToolCategory.MANAGE,  ["delete", "remove", "update", "create", "add", "manage",
                            "删除", "修改", "添加", "管理"]),
]

# 中文分词简易模式：提取连续汉字、英文单词、含下划线标识符
_TOKEN_RE = re.compile(r'[\u4e00-\u9fff]+|[a-zA-Z_][a-zA-Z0-9_]*')


def _extract_keywords(name: str, description: str) -> List[str]:
    """从工具名和描述中提取关键词"""
    text = f"{name} {description}".lower()
    tokens = _TOKEN_RE.findall(text)
    # 去重并去掉过短 token
    seen = set()
    result = []
    for t in tokens:
        t_lower = t.lower()
        if len(t_lower) >= 2 and t_lower not in seen:
            seen.add(t_lower)
            result.append(t_lower)
    return result


def _classify(name: str, description: str) -> ToolCategory:
    """基于规则对工具自动分类"""
    text = f"{name} {description}".lower()
    for category, patterns in _CATEGORY_RULES:
        for p in patterns:
            if p in text:
                return category
    return ToolCategory.GENERAL


class ToolManager:
    """
    工具管理器

    统一管理本地工具和 MCP 工具的注册、发现、过滤与调用。
    """

    def __init__(self, config: Dict[str, Any] = None,
                 mcp_manager=None, rate_limiter=None,
                 logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.mcp_manager = mcp_manager
        self.rate_limiter = rate_limiter

        # 核心注册表：name -> ToolInfo
        self._registry: Dict[str, ToolInfo] = {}

        # 分类索引：category -> [tool_name, ...]
        self._category_index: Dict[ToolCategory, List[str]] = {c: [] for c in ToolCategory}

        # 关键词倒排索引：keyword -> [tool_name, ...]
        self._keyword_index: Dict[str, List[str]] = {}

        # 自动从 MCP 导入工具
        if self.mcp_manager:
            self.sync_mcp_tools()

        self.logger.info(
            f"ToolManager 初始化完成 | 工具总数: {len(self._registry)} | "
            f"分类: {self._category_summary()}"
        )

    # ================================================================
    # 注册 / 卸载 / 更新
    # ================================================================

    def register_local_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Dict[str, Any] = None,
        category: ToolCategory = None,
        keywords: List[str] = None,
    ) -> ToolInfo:
        """
        注册本地工具（Python 可调用对象）

        Args:
            name: 工具名（唯一标识）
            description: 工具描述
            handler: 可调用对象 (parameters: dict) -> Any
            input_schema: 参数 JSON Schema（可选）
            category: 手动指定分类（None 则自动推断）
            keywords: 手动指定关键词（None 则自动提取）

        Returns:
            注册后的 ToolInfo
        """
        cat = category or _classify(name, description)
        kws = keywords or _extract_keywords(name, description)

        tool = ToolInfo(
            name=name,
            description=description,
            source="local",
            category=cat,
            keywords=kws,
            input_schema=input_schema or {},
            handler=handler,
        )

        self._add_to_registry(tool)
        self.logger.info(f"注册本地工具: {name} | 分类: {cat.value} | 关键词: {kws[:5]}")
        return tool

    def sync_mcp_tools(self):
        """从 MCPManager 同步所有可用的 MCP 工具"""
        if not self.mcp_manager:
            return

        mcp_tools = self.mcp_manager.get_available_tools(use_cache=False)
        added = 0
        for t in mcp_tools:
            name = t.get('name', '')
            if not name:
                continue
            # 已注册则跳过（除非需要更新）
            if name in self._registry and self._registry[name].source == "mcp":
                # 更新 schema（MCP 可能有新版本）
                self._registry[name].input_schema = t.get('inputSchema', {})
                continue

            cat = _classify(name, t.get('description', ''))
            kws = _extract_keywords(name, t.get('description', ''))

            tool = ToolInfo(
                name=name,
                description=t.get('description', ''),
                source="mcp",
                server_name=t.get('server', ''),
                category=cat,
                keywords=kws,
                input_schema=t.get('inputSchema', {}),
            )
            self._add_to_registry(tool)
            added += 1

        if added:
            self.logger.info(f"从 MCP 同步了 {added} 个工具")

    def unregister(self, name: str) -> bool:
        """卸载工具"""
        tool = self._registry.pop(name, None)
        if not tool:
            return False

        # 清理索引
        cat_list = self._category_index.get(tool.category, [])
        if name in cat_list:
            cat_list.remove(name)

        for kw in tool.keywords:
            kw_list = self._keyword_index.get(kw, [])
            if name in kw_list:
                kw_list.remove(name)

        self.logger.info(f"卸载工具: {name}")
        return True

    def set_enabled(self, name: str, enabled: bool):
        """启用/禁用工具"""
        tool = self._registry.get(name)
        if tool:
            tool.enabled = enabled
            self.logger.info(f"工具 {name} {'启用' if enabled else '禁用'}")

    # ================================================================
    # 多层过滤：查询 → 关键词/分类过滤 → 频率过滤 → 返回列表
    # ================================================================

    def get_tools_for_query(self, query: str = "", top_k: int = 15) -> List[ToolInfo]:
        """
        根据用户查询多层过滤，返回候选工具列表

        过滤流程：
        1. 关键词/分类粗过滤：用户查询的关键词匹配工具关键词和分类
        2. 频率过滤：排除已达调用频率上限的工具（通过 RateLimiter）
        3. 取 top_k 返回

        Args:
            query: 用户查询（空字符串返回全部已启用工具）
            top_k: 最大返回数量

        Returns:
            过滤后的 ToolInfo 列表（仅已启用）
        """
        # 全量启用工具
        all_enabled = [t for t in self._registry.values() if t.enabled]

        if not query:
            candidates = all_enabled
        else:
            # Layer 1: 关键词 + 分类粗过滤
            candidates = self._keyword_category_filter(query, all_enabled)
            # 如果过滤后结果太少（<3），补充全量工具（避免漏掉）
            if len(candidates) < 3:
                seen = {t.name for t in candidates}
                for t in all_enabled:
                    if t.name not in seen:
                        candidates.append(t)

        # Layer 2: 频率过滤（排除已达限额的工具）
        if self.rate_limiter:
            candidates = [
                t for t in candidates
                if self.rate_limiter.is_allowed(t.name)
            ]

        return candidates[:top_k]

    def get_tools_desc(self, query: str = "", target_tool: str = None) -> str:
        """
        获取工具列表的文本描述（给 LLM 使用）

        懒加载策略：
        - 默认仅返回 name + description（节省 token）
        - target_tool 指定的工具额外加载完整 schema

        Args:
            query: 用户查询（用于过滤）
            target_tool: 需要加载完整 schema 的工具名

        Returns:
            格式化的工具列表字符串
        """
        desc = ["1. search_knowledge - 从知识库中搜索相关信息"]
        idx = 2

        tools = self.get_tools_for_query(query)
        for tool in tools:
            td = f"{idx}. {tool.name} - {tool.description}"

            # 仅为目标工具加载完整 schema
            if target_tool and tool.name == target_tool and tool.input_schema:
                import json
                sj = json.dumps(
                    self._compress_schema(tool.input_schema),
                    ensure_ascii=False, indent=2
                )
                td += f"\n   Input Schema:\n" + '\n'.join('   ' + l for l in sj.split('\n'))

            desc.append(td)
            idx += 1

        desc.append(f"\n{idx}. finish - 完成任务并返回最终答案")
        return "\n".join(desc)

    # ================================================================
    # 工具查找与 Schema 获取
    # ================================================================

    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """按名称获取工具"""
        return self._registry.get(name)

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具的完整 inputSchema（懒加载第二阶段）"""
        tool = self._registry.get(name)
        return tool.input_schema if tool else None

    def find_tool_server(self, name: str) -> Optional[str]:
        """查找 MCP 工具所属服务器"""
        tool = self._registry.get(name)
        if tool and tool.source == "mcp":
            return tool.server_name
        # 降级：返回第一个可用的 MCP 服务器
        if self.mcp_manager:
            servers = self.mcp_manager.get_enabled_servers()
            if servers:
                return servers[0].get('name') if isinstance(servers, list) else list(servers.keys())[0]
        return None

    def record_call(self, name: str):
        """记录一次工具调用（更新计数 + 通知 RateLimiter）"""
        tool = self._registry.get(name)
        if tool:
            tool.call_count += 1
        if self.rate_limiter:
            self.rate_limiter.record_call(name)

    def call_local_tool(self, name: str, parameters: Dict[str, Any]) -> Any:
        """调用本地工具"""
        tool = self._registry.get(name)
        if not tool:
            raise ValueError(f"工具不存在: {name}")
        if tool.source != "local":
            raise ValueError(f"工具 {name} 不是本地工具，请通过 MCP 调用")
        if not tool.handler:
            raise ValueError(f"工具 {name} 没有注册 handler")
        if not tool.enabled:
            raise ValueError(f"工具 {name} 已被禁用")

        self.record_call(name)
        return tool.handler(parameters)

    def list_tools_brief(self) -> List[Dict[str, Any]]:
        """列出所有工具的简要信息"""
        return [
            {
                "name": t.name,
                "source": t.source,
                "category": t.category.value,
                "description": t.description[:80],
                "enabled": t.enabled,
                "call_count": t.call_count,
            }
            for t in self._registry.values()
        ]

    # ================================================================
    # 内部方法
    # ================================================================

    def _add_to_registry(self, tool: ToolInfo):
        """注册工具并建立索引"""
        self._registry[tool.name] = tool

        # 分类索引
        self._category_index[tool.category].append(tool.name)

        # 关键词倒排索引
        for kw in tool.keywords:
            self._keyword_index.setdefault(kw, []).append(tool.name)

    def _keyword_category_filter(
        self, query: str, tools: List[ToolInfo]
    ) -> List[ToolInfo]:
        """关键词 + 分类粗过滤"""
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return tools

        scored: List[tuple] = []
        for tool in tools:
            score = 0

            # 关键词匹配得分
            tool_kw_set = set(tool.keywords)
            overlap = query_tokens & tool_kw_set
            score += len(overlap) * 3  # 精确匹配权重高

            # 子串匹配（"查订单" → "订单" 在关键词中）
            for qt in query_tokens:
                for tk in tool.keywords:
                    if qt in tk or tk in qt:
                        score += 1

            # 分类匹配加分
            for cat, patterns in _CATEGORY_RULES:
                if tool.category == cat:
                    for p in patterns:
                        if p in query.lower():
                            score += 2
                            break

            # 调用频率加分（常用工具优先）
            if tool.call_count > 0:
                score += min(tool.call_count, 5) * 0.5

            if score > 0:
                scored.append((score, tool))

        # 按得分降序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]

    def _category_summary(self) -> str:
        """生成分类摘要"""
        parts = []
        for cat, names in self._category_index.items():
            if names:
                parts.append(f"{cat.value}({len(names)})")
        return ", ".join(parts) if parts else "无"

    @staticmethod
    def _compress_schema(schema: dict) -> dict:
        """压缩 JSON Schema，减少 token 占用"""
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            if key in ('additionalProperties', 'title', '$defs'):
                continue
            if key == 'properties' and isinstance(value, dict):
                result[key] = {
                    pname: ToolManager._compress_property(pschema)
                    for pname, pschema in value.items()
                }
            else:
                result[key] = value
        return result

    @staticmethod
    def _compress_property(prop: dict) -> dict:
        """压缩单个属性的 schema"""
        if not isinstance(prop, dict):
            return prop
        if 'anyOf' in prop:
            non_null = [s for s in prop['anyOf'] if s.get('type') != 'null']
            if len(non_null) == 1:
                compressed = dict(non_null[0])
                compressed['optional'] = True
                for k in ('description', 'default', 'enum'):
                    if k in prop:
                        compressed[k] = prop[k]
                return compressed
        cleaned = {}
        for k, v in prop.items():
            if k not in ('title',):
                cleaned[k] = v
        return cleaned
