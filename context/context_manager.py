"""
上下文工程模块 (Context Engineering Manager)

企业级LLM上下文窗口管理，实现三大核心能力：

1. 分区预算管理
   LLM窗口被划分为多个分区，各分区有独立预算：
   - system_prompt:    系统提示词（优先级最高，几乎不压缩）
   - planning_steps:   规划步骤
   - tools:            工具列表（支持懒加载）
   - short_term_memory: 短期记忆（最近对话）
   - rag_results:      RAG检索结果
   - long_term_memory:  长期记忆
   - tool_results:     工具执行结果（优先级最低，最先被裁剪）

2. 工具懒加载 (Lazy Loading)
   - 选择阶段：仅加载工具名称+描述（关键词粗过滤 → 权限过滤 → 工具列表）
   - 参数填充阶段：按需加载完整 Input Schema
   - 通过延迟加载可减少约60-80%的工具描述token占用

3. 预算溢出分级处理（按优先级依次执行）
   - 裁剪(Trim):       丢弃历史N轮之前的工具执行结果、减少保留对话轮次
   - 微压缩(Micro-compress): 程序化去除JSON冗余、合并空行、截断长值
   - 投影(Project):     将大块内容存入文件，上下文中仅保留引用
   - 自动压缩(Auto-compress): 使用LLM进行智能摘要
"""

import json
import os
import re
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ContextSection:
    """上下文分区

    Attributes:
        name:     分区名称，与 section_budgets 配置中的 key 对应
        content:  分区原始内容
        priority: 优先级（1=最高/最不可能被裁剪，数值越大越容易被裁剪）
    """
    name: str
    content: str
    priority: int

    @property
    def length(self) -> int:
        return len(self.content)


class ContextWindowManager:
    """
    企业级上下文窗口管理器

    提供两种使用方式：
    1. manage_section(name, content) - 对单个分区应用预算管理，适合嵌入现有代码
    2. build_context(sections)       - 对完整上下文应用全局预算管理，适合全新组装
    """

    # 分区默认优先级（数字越小越重要，越不会被裁剪）
    DEFAULT_PRIORITIES = {
        'system_prompt': 1,
        'planning_steps': 2,
        'tools': 3,
        'short_term_memory': 4,
        'rag_results': 5,
        'long_term_memory': 6,
        'tool_results': 7,
    }

    def __init__(self, config: Dict[str, Any], tool_registry=None,
                 llm_client=None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        ctx_config = config.get('context_window', {})

        # 总预算（字符数）
        self.total_budget = ctx_config.get('total_budget', 12000)

        # 各分区预算
        self.section_budgets: Dict[str, int] = ctx_config.get('section_budgets', {
            'system_prompt': 2000,
            'long_term_memory': 1000,
            'short_term_memory': 2000,
            'rag_results': 2000,
            'planning_steps': 1500,
            'tools': 2000,
            'tool_results': 2000,
        })

        # 裁剪配置：保留最近N轮工具执行结果
        self.trim_tool_results_after_rounds = ctx_config.get(
            'trim_tool_results_after_rounds', 5
        )

        # 投影配置
        self.projection_threshold = ctx_config.get('projection_threshold', 3000)
        self.projection_dir = ctx_config.get(
            'projection_dir', './data/context_projections'
        )
        os.makedirs(self.projection_dir, exist_ok=True)

        self.tool_registry = tool_registry
        self.llm_client = llm_client

        # 动态预算：未使用分区的预算可重分配给活跃分区
        self._dynamic_budgets: Optional[Dict[str, int]] = None

        self.logger.info(
            f"ContextWindowManager 初始化 | 总预算: {self.total_budget} 字符 | "
            f"分区数: {len(self.section_budgets)}"
        )

    # ================================================================
    # 动态预算分配
    # ================================================================

    def set_active_sections(self, active_sections: List[str]):
        """
        声明当前任务实际使用的分区，动态重分配预算

        例如当前任务没有RAG操作，则rag_results的预算会按比例
        分配给其他活跃分区，充分利用上下文窗口。

        在每次prompt组装前调用，后续 manage_section() 会自动使用动态预算。

        Args:
            active_sections: 当前任务实际使用的分区名称列表
        """
        self._dynamic_budgets = self._calculate_dynamic_budgets(active_sections)
        self.logger.debug(
            f"[动态预算] 活跃分区: {active_sections} | "
            f"动态预算: {self._dynamic_budgets}"
        )

    def _get_section_budget(self, section_name: str) -> int:
        """获取分区预算（优先使用动态预算，否则回退到静态配置）"""
        if self._dynamic_budgets:
            return self._dynamic_budgets.get(
                section_name, self.section_budgets.get(section_name, 2000)
            )
        return self.section_budgets.get(section_name, 2000)

    def _calculate_dynamic_budgets(
        self, active_sections: List[str]
    ) -> Dict[str, int]:
        """
        动态计算预算：将未使用分区的预算按比例分配给活跃分区

        例如：
        - 静态预算: rag_results=2000, tools=2000, short_term_memory=2000
        - 当前任务无RAG -> rag_results的2000字符按比例分给tools和short_term_memory
        - tools动态预算: 2000 + 2000*(2000/4000) = 3000
        - short_term_memory动态预算: 2000 + 2000*(2000/4000) = 3000
        """
        unused_budget = 0
        active_base_total = 0

        for name, budget in self.section_budgets.items():
            if name in active_sections:
                active_base_total += budget
            else:
                unused_budget += budget

        if unused_budget == 0 or active_base_total == 0:
            return dict(self.section_budgets)

        dynamic = {}
        for name in active_sections:
            base = self.section_budgets.get(name, 2000)
            extra = int(unused_budget * (base / active_base_total))
            dynamic[name] = base + extra

        if unused_budget > 0:
            self.logger.info(
                f"[动态预算] 未使用分区释放 {unused_budget} 字符预算，"
                f"重分配给 {len(active_sections)} 个活跃分区"
            )

        return dynamic

    def reset_dynamic_budgets(self):
        """重置动态预算，回退到静态配置"""
        self._dynamic_budgets = None

    # ================================================================
    # 核心接口
    # ================================================================

    def manage_section(self, section_name: str, content: str) -> str:
        """
        对单个分区内容应用预算管理（最小侵入式集成方式）

        在现有代码中，只需在拼接prompt前对各部分调用此方法：
            context = context_manager.manage_section('short_term_memory', context)

        预算会自动使用动态分配结果（如果已调用 set_active_sections），
        否则回退到静态配置。

        Args:
            section_name: 分区名称，对应 section_budgets 中的 key
            content:      分区原始内容

        Returns:
            预算管理后的内容（可能被裁剪、压缩或投影）
        """
        if not content or not content.strip():
            return content

        budget = self._get_section_budget(section_name)

        if len(content) <= budget:
            return content

        self.logger.info(
            f"[预算控制] 分区 '{section_name}' 超出预算 "
            f"({len(content)}/{budget})，启动分级压缩"
        )

        section = ContextSection(
            name=section_name,
            content=content,
            priority=self.DEFAULT_PRIORITIES.get(section_name, 5),
        )
        return self._apply_overflow_strategy(section, budget)

    def build_context(self, sections: List[ContextSection]) -> str:
        """
        组装完整上下文，应用全局预算管理

        流程：
        1. 对每个分区应用独立预算上限
        2. 检查总预算，超出时按优先级从低到高压缩各分区
        3. 按优先级排序拼接最终上下文

        Args:
            sections: 上下文分区列表

        Returns:
            组装后的上下文字符串
        """
        # Step 1: 动态预算分配 — 未使用分区的预算按比例重分配给活跃分区
        active_names = [s.name for s in sections if s.content.strip()]
        dynamic_budgets = self._calculate_dynamic_budgets(active_names)

        # Step 2: 各分区独立预算控制
        for section in sections:
            budget = dynamic_budgets.get(
                section.name, self.section_budgets.get(section.name, 2000)
            )
            if section.length > budget:
                self.logger.info(
                    f"[预算控制] 分区 '{section.name}' 超出动态预算 "
                    f"({section.length}/{budget})"
                )
                section.content = self._apply_overflow_strategy(section, budget)

        # Step 3: 全局预算控制
        total_len = sum(s.length for s in sections)
        if total_len > self.total_budget:
            self.logger.info(
                f"[全局预算] 总上下文超出预算 ({total_len}/{self.total_budget})，"
                f"启动全局压缩"
            )
            sections = self._global_budget_control(sections)

        # Step 4: 按优先级排序后拼接
        sections.sort(key=lambda s: s.priority)
        result_parts = [s.content for s in sections if s.content.strip()]

        final_context = "\n\n".join(result_parts)
        self.logger.info(
            f"[上下文组装] 最终长度: {len(final_context)} 字符, "
            f"分区数: {len(result_parts)}"
        )
        return final_context

    # ================================================================
    # 工具懒加载
    # ================================================================

    def get_tools_brief_desc(self, query: str = None) -> str:
        """
        获取工具的轻量描述（仅名称+描述，不含 Input Schema）

        这是懒加载的第一阶段：让LLM基于简要信息选择工具。
        完整schema仅在 _extract_and_call_mcp 中按需加载。

        Args:
            query: 用户查询，用于关键词粗过滤缩小候选范围

        Returns:
            格式化的工具列表字符串
        """
        desc_lines = ["1. search_knowledge - 从知识库中搜索相关信息"]
        idx = 2

        if self.tool_registry:
            if query:
                # 关键词粗过滤 → LLM语义精过滤
                tools = self.tool_registry.find_tools(query)
            else:
                tools = [
                    t for t in self.tool_registry._registry.values()
                    if t.enabled and t.source == "mcp"
                ]

            for tool in tools:
                desc_lines.append(f"{idx}. {tool.name} - {tool.description}")
                idx += 1

        desc_lines.append(f"\n{idx}. finish - 完成任务并返回最终答案")
        return "\n".join(desc_lines)

    def get_tool_full_desc(self, tool_name: str) -> str:
        """
        获取工具的完整描述（含 Input Schema）

        懒加载的第二阶段：仅在LLM需要填充参数时才调用。

        Args:
            tool_name: 工具名称

        Returns:
            包含完整schema的工具描述字符串
        """
        if not self.tool_registry:
            return f"未找到工具 {tool_name}"

        schema = self.tool_registry.get_tool_schema(tool_name)
        if not schema:
            return f"未找到工具 {tool_name} 的schema"

        tool = self.tool_registry.get_tool(tool_name)
        desc = tool.description if tool else ''
        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

        return (
            f"工具: {tool_name}\n"
            f"描述: {desc}\n"
            f"Input Schema:\n{schema_json}"
        )

    # ================================================================
    # 溢出分级处理
    # ================================================================

    def _apply_overflow_strategy(self, section: ContextSection,
                                 budget: int) -> str:
        """
        对单个分区应用溢出策略（按优先级依次执行）：
        裁剪 → 微压缩 → 投影 → 自动压缩 → 硬截断（兜底）
        """
        content = section.content

        # 策略1：裁剪
        content = self._trim(section.name, content, budget)
        if len(content) <= budget:
            self.logger.debug(f"[{section.name}] 裁剪后满足预算")
            return content

        # 策略2：微压缩
        content = self._micro_compress(content)
        if len(content) <= budget:
            self.logger.debug(f"[{section.name}] 微压缩后满足预算")
            return content

        # 策略3：投影（将大块内容存入文件，上下文仅保留引用）
        if len(content) > self.projection_threshold:
            content = self._project_to_file(section.name, content, budget)
            if len(content) <= budget:
                self.logger.debug(f"[{section.name}] 投影后满足预算")
                return content

        # 策略4：自动压缩（LLM摘要）
        if self.llm_client and len(content) > budget:
            content = self._auto_compress(content, budget)
            if len(content) <= budget:
                self.logger.debug(f"[{section.name}] 自动压缩后满足预算")
                return content

        # 兜底：硬截断
        if len(content) > budget:
            self.logger.warning(
                f"[{section.name}] 所有策略后仍超预算，硬截断 "
                f"({len(content)} → {budget})"
            )
            content = content[:budget - 20] + "\n...(已截断)"

        return content

    def _global_budget_control(
        self, sections: List[ContextSection]
    ) -> List[ContextSection]:
        """全局预算控制：按优先级从低到高压缩各分区，直到总量满足预算"""
        total = sum(s.length for s in sections)
        overflow = total - self.total_budget

        # 按优先级从低到高排序（先压缩不重要的分区）
        by_priority = sorted(sections, key=lambda s: s.priority, reverse=True)

        for section in by_priority:
            if overflow <= 0:
                break

            current_len = section.length
            target_len = max(200, current_len - overflow)

            if current_len > target_len:
                section.content = self._apply_overflow_strategy(
                    section, target_len
                )
                saved = current_len - section.length
                overflow -= saved
                self.logger.info(
                    f"[全局压缩] 分区 '{section.name}' 压缩节省 {saved} 字符"
                )

        return sections

    # ================================================================
    # 策略1：裁剪 (Trim)
    # ================================================================

    def _trim(self, section_name: str, content: str, budget: int) -> str:
        """
        裁剪策略：根据分区类型采用不同方式

        - tool_results:      丢弃历史N轮之前的详细执行结果
        - short_term_memory: 保留摘要 + 最近对话，裁剪早期对话
        - rag_results:       减少返回文档数量，保留最相关的
        - 其他:              保留头尾，裁剪中间
        """
        dispatch = {
            'tool_results': self._trim_tool_results,
            'short_term_memory': self._trim_memory,
            'rag_results': self._trim_rag_results,
            'tools': self._trim_tools,
        }

        handler = dispatch.get(section_name)
        if handler:
            return handler(content, budget)

        # 通用裁剪：保留头尾
        if len(content) > budget:
            half = budget // 2 - 15
            return content[:half] + "\n...(中间部分已裁剪)...\n" + content[-half:]
        return content

    def _trim_tool_results(self, content: str, budget: int) -> str:
        """裁剪工具执行结果：丢弃较早轮次的详细结果，仅保留摘要行"""
        lines = content.split('\n')

        # 按"步骤N"分块
        step_blocks: List[str] = []
        current_block: List[str] = []
        for line in lines:
            if re.match(r'^步骤\s*\d+', line) and current_block:
                step_blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            step_blocks.append('\n'.join(current_block))

        if len(step_blocks) <= self.trim_tool_results_after_rounds:
            # 步骤数不多，尝试截断每个步骤的结果长度
            return self._truncate_step_results(content, budget)

        # 较早的步骤只保留描述行，最近N轮保留完整结果
        trimmed = []
        cutoff = len(step_blocks) - self.trim_tool_results_after_rounds
        for i, block in enumerate(step_blocks):
            if i < cutoff:
                first_line = block.split('\n')[0]
                trimmed.append(f"{first_line} (详细结果已省略)")
            else:
                trimmed.append(block)

        return '\n'.join(trimmed)

    @staticmethod
    def _truncate_step_results(content: str, budget: int) -> str:
        """截断每个步骤的结果，使总长度满足预算"""
        if len(content) <= budget:
            return content
        lines = content.split('\n')
        result = []
        remaining = budget
        for line in lines:
            if remaining <= 50:
                result.append("...(后续结果已省略)")
                break
            if len(line) > remaining:
                result.append(line[:remaining - 20] + "...(已截断)")
                remaining = 0
            else:
                result.append(line)
                remaining -= len(line) + 1
        return '\n'.join(result)

    def _trim_memory(self, content: str, budget: int) -> str:
        """裁剪短期记忆：保留历史摘要 + 最近对话，裁剪早期详细对话"""
        lines = content.split('\n')

        # 分离摘要部分和对话部分
        summary_lines: List[str] = []
        conversation_lines: List[str] = []
        in_summary = False

        for line in lines:
            if '[历史对话摘要]' in line:
                in_summary = True
                summary_lines.append(line)
            elif '[最近对话]' in line:
                in_summary = False
                conversation_lines.append(line)
            elif in_summary:
                summary_lines.append(line)
            else:
                conversation_lines.append(line)

        summary_text = '\n'.join(summary_lines)
        remaining_budget = budget - len(summary_text) - 10

        conv_text = '\n'.join(conversation_lines)
        if len(conv_text) > remaining_budget and remaining_budget > 50:
            # 从尾部保留（最近对话更重要）
            conv_text = (
                "...(早期对话已裁剪)\n"
                + conv_text[-(remaining_budget - 30):]
            )

        if summary_text:
            return summary_text + '\n' + conv_text
        return conv_text

    @staticmethod
    def _trim_rag_results(content: str, budget: int) -> str:
        """裁剪RAG结果：减少文档数量，保留最相关的前几个"""
        if len(content) <= budget:
            return content

        docs = re.split(r'(\[文档\s*\d+\])', content)
        if len(docs) <= 3:
            return content[:budget - 20] + "\n...(已截断)"

        # 重组：保留前2个文档
        result = docs[0]  # 前缀文本
        kept = 0
        i = 1
        while i < len(docs) - 1 and kept < 2:
            result += docs[i] + docs[i + 1]  # 标签 + 内容
            kept += 1
            i += 2

        total_docs = (len(docs) - 1) // 2
        trimmed_count = total_docs - kept
        if trimmed_count > 0:
            result += f"\n(还有 {trimmed_count} 个相关文档已省略以节省上下文空间)"

        # 如果仍超预算，截断
        if len(result) > budget:
            result = result[:budget - 20] + "\n...(已截断)"

        return result

    @staticmethod
    def _trim_tools(content: str, budget: int) -> str:
        """裁剪工具列表：去掉 Input Schema 详情，仅保留工具名称+描述

        工具列表格式示例:
            1. search_knowledge - 从知识库中搜索相关信息
               参数: {"query": "搜索关键词"}
            2. get_user_detail - 查询用户详情
               Input Schema:
               { ... }
            3. finish - 完成任务并返回最终答案

        裁剪策略：逐步去掉 Input Schema 块，直到满足预算。
        确保每个工具至少保留 "序号. 名称 - 描述" 这一行。
        """
        if len(content) <= budget:
            return content

        lines = content.split('\n')
        # 第一轮：去掉所有 Input Schema 块（含缩进行），保留工具名称行
        brief_lines = []
        skip_schema = False
        for line in lines:
            stripped = line.strip()
            # 检测 Input Schema 开始
            if stripped.startswith('Input Schema:'):
                skip_schema = True
                continue
            # 检测新的工具条目（以数字+点开头）或 finish 条目 → 结束 schema 跳过
            if skip_schema and re.match(r'^\d+\.', stripped):
                skip_schema = False
            # 跳过 schema 内的缩进行（JSON内容）
            if skip_schema:
                continue
            # 跳过单独的参数行（"   参数: ..."），这也是详情
            if stripped.startswith('参数:') and len(brief_lines) > 0:
                continue
            brief_lines.append(line)

        result = '\n'.join(brief_lines)
        if len(result) <= budget:
            return result

        # 第二轮：如果仍超预算，截断工具条目数量（从后往前移除，保留前面的工具）
        tool_lines = []
        other_lines = []
        for line in brief_lines:
            if re.match(r'^\s*\d+\.\s+', line):
                tool_lines.append(line)
            else:
                other_lines.append(line)

        while tool_lines and len('\n'.join(tool_lines + other_lines)) > budget:
            # 保留第一个(search_knowledge)和最后一个(finish)，从倒数第二个开始移除
            if len(tool_lines) > 2:
                tool_lines.pop(-2)
            else:
                break

        result = '\n'.join(tool_lines)
        if len(result) > budget:
            result = result[:budget - 30] + "\n...(部分工具已省略)"
        return result

    # ================================================================
    # 策略2：微压缩 (Micro-compress)
    # ================================================================

    def _micro_compress(self, content: str) -> str:
        """
        微压缩：通过工程代码去除冗余数据

        处理方式：
        1. 合并连续空行
        2. 压缩JSON块中的空值和冗余字段
        3. 去除重复段落
        4. 截断超长单行
        """
        # 1. 合并连续空行（3个以上空行 → 2个）
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 2. 压缩内容中的JSON块
        content = self._compress_json_blocks(content)

        # 3. 去除重复段落
        content = self._deduplicate_paragraphs(content)

        # 4. 截断超长单行（>500字符）
        lines = content.split('\n')
        compressed = []
        for line in lines:
            if len(line) > 500:
                line = line[:500] + "...(已截断)"
            compressed.append(line)
        content = '\n'.join(compressed)

        return content

    def _compress_json_blocks(self, content: str) -> str:
        """压缩内容中的JSON块：移除null值、空字段、缩短长字符串值"""

        def _compress_match(match):
            try:
                obj = json.loads(match.group(0))
                compressed = self._compress_json_obj(obj)
                return json.dumps(
                    compressed, ensure_ascii=False, separators=(',', ':')
                )
            except (json.JSONDecodeError, Exception):
                return match.group(0)

        # 匹配 50+ 字符的 JSON 对象
        content = re.sub(r'\{[^{}]{50,}\}', _compress_match, content)
        return content

    def _compress_json_obj(self, obj: Any) -> Any:
        """递归压缩JSON对象：移除空值、截断长字符串、限制数组长度"""
        if isinstance(obj, dict):
            compressed = {}
            for k, v in obj.items():
                if v is None or v == "" or v == [] or v == {}:
                    continue
                compressed[k] = self._compress_json_obj(v)
            return compressed
        elif isinstance(obj, list):
            # 数组最多保留10项
            items = [self._compress_json_obj(item) for item in obj[:10]]
            if len(obj) > 10:
                items.append(f"...(共{len(obj)}项)")
            return items
        elif isinstance(obj, str) and len(obj) > 200:
            return obj[:200] + "..."
        return obj

    @staticmethod
    def _deduplicate_paragraphs(content: str) -> str:
        """去除重复段落（基于前100字符的指纹去重）"""
        paragraphs = content.split('\n\n')
        seen = set()
        unique = []
        for para in paragraphs:
            normalized = para.strip()
            if not normalized:
                continue
            fingerprint = normalized[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(para)
            # else: 跳过重复段落
        return '\n\n'.join(unique)

    # ================================================================
    # 策略3：投影 (Project)
    # ================================================================

    def _project_to_file(self, section_name: str, content: str,
                         budget: int) -> str:
        """
        投影策略：将复杂任务的过程输入、思考和结果从上下文中
        移动到一个文件中，上下文仅保留摘要引用。

        需要完整信息时可从文件中读取。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content[:200].encode()).hexdigest()[:8]
        filename = f"{section_name}_{timestamp}_{content_hash}.txt"
        filepath = os.path.join(self.projection_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== 投影内容: {section_name} ===\n")
                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"原始长度: {len(content)} 字符\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(content)

            self.logger.info(
                f"[投影] 分区 '{section_name}' 已投影至文件: {filename} "
                f"(原始 {len(content)} 字符)"
            )

            # 在上下文中保留部分摘要 + 文件引用
            summary_budget = budget - 100
            summary = content[:summary_budget]
            reference = (
                f"\n[完整信息已投影至文件: {filename}，"
                f"如需完整内容可从该文件读取]"
            )
            return summary + reference

        except Exception as e:
            self.logger.error(f"[投影] 写入文件失败: {e}")
            return content[:budget - 20] + "\n...(已截断)"

    def read_projection(self, filename: str) -> Optional[str]:
        """从投影文件中读取完整内容（供后续需要时调用）"""
        filepath = os.path.join(self.projection_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"[投影] 文件不存在: {filepath}")
            return None
        except Exception as e:
            self.logger.error(f"[投影] 读取失败: {e}")
            return None

    # ================================================================
    # 策略4：自动压缩 (Auto-compress)
    # ================================================================

    def _auto_compress(self, content: str, target_length: int) -> str:
        """
        自动压缩：使用LLM对内容进行智能摘要

        这是最后的压缩手段，在裁剪、微压缩、投影都无法满足预算时使用。
        通过LLM提取关键信息，生成不超过目标长度的摘要。
        """
        if not self.llm_client:
            return content[:target_length - 20] + "\n...(已截断)"

        try:
            # 限制发送给LLM的内容长度，避免自身也超出上下文
            input_content = content[:4000]

            prompt = (
                f"请将以下内容压缩为不超过{target_length}字符的摘要，"
                f"保留所有关键信息：\n"
                f"1. 必须保留所有ID标识符（用户ID、订单号、商品ID等）\n"
                f"2. 保留核心数据、结论和操作结果\n"
                f"3. 去除冗余描述和重复内容\n"
                f"4. 保持信息准确性和可理解性\n\n"
                f"原始内容：\n{input_content}\n\n"
                f"压缩后的内容："
            )

            compressed = self.llm_client.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=target_length
            ).strip()

            self.logger.info(
                f"[自动压缩] 原长度: {len(content)}, "
                f"压缩后: {len(compressed)}"
            )
            return compressed

        except Exception as e:
            self.logger.error(f"[自动压缩] LLM摘要失败: {e}")
            return content[:target_length - 20] + "\n...(已截断)"

    # ================================================================
    # 便捷方法：组装特定场景的上下文
    # ================================================================

    def assemble_full_context(
        self,
        system_prompt: str = "",
        short_term_memory: str = "",
        long_term_memory: str = "",
        rag_results: str = "",
        planning_steps: str = "",
        tools: str = "",
        tool_results: str = "",
    ) -> str:
        """
        组装完整的LLM上下文窗口

        将所有分区按照企业级上下文工程方式组装，自动应用
        预算管理和溢出处理。

        Args:
            system_prompt:    系统提示词
            short_term_memory: 短期记忆（最近对话历史）
            long_term_memory:  长期记忆（历史对话语义检索结果）
            rag_results:      RAG检索结果
            planning_steps:   规划步骤
            tools:            工具列表描述
            tool_results:     工具执行结果

        Returns:
            组装后的上下文字符串
        """
        sections = []
        section_data = {
            'system_prompt': system_prompt,
            'short_term_memory': short_term_memory,
            'long_term_memory': long_term_memory,
            'rag_results': rag_results,
            'planning_steps': planning_steps,
            'tools': tools,
            'tool_results': tool_results,
        }

        for name, content in section_data.items():
            if content and content.strip():
                sections.append(ContextSection(
                    name=name,
                    content=content,
                    priority=self.DEFAULT_PRIORITIES.get(name, 5),
                ))

        return self.build_context(sections)
