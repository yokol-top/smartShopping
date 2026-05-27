"""
子Agent上下文工程模块

企业级上下文管理在主/子Agent架构中的实践：

┌──────────────────────────────────────────────────────────┐
│                  主Agent (SmartAgent)                     │
│  ┌──────────────────────────────────────────────────┐    │
│  │ 完整上下文窗口 (12000 chars)                       │    │
│  │  system_prompt | short_term_memory | long_term   │    │
│  │  tools | rag_results | planning_steps            │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│              dispatch → 上下文裁剪/投影                    │
│                         │                                │
│     ┌───────────────────┴───────────────────┐            │
│     ▼                                       ▼            │
│  ┌──────────────┐                  ┌──────────────┐      │
│  │ 售前子Agent    │                  │ 功能子Agent    │      │
│  │ (4000 chars)  │                  │ (4000 chars)  │      │
│  │               │                  │               │      │
│  │ task_brief    │                  │ task_brief    │      │
│  │ conv_context  │                  │ conv_context  │      │
│  │ rag_context   │                  │ tool_context  │      │
│  │ user_profile  │                  │ user_profile  │      │
│  │ main_replies  │                  │ main_replies  │      │
│  └──────────────┘                  └──────────────┘      │
└──────────────────────────────────────────────────────────┘

上下文分层：
1. task_brief     (P1) 任务摘要：用户当前请求 + 约束条件（不可裁剪）
2. conv_context   (P2) 对话上下文：最近对话精简版（按轮次淘汰）
3. domain_context (P3) 领域上下文：RAG/工具相关（按相关度裁剪）
4. user_profile   (P4) 用户画像：user_id, username, 偏好（结构化，极少淘汰）
5. main_replies   (P5) 主Agent补充：从主Agent获取的回复（任务结束后淘汰）

注入时机：
- task_brief:     dispatch 时一次性注入
- conv_context:   dispatch 时从主Agent短期记忆裁剪注入
- domain_context: 子Agent执行过程中按需构建（RAG检索/工具列表）
- user_profile:   dispatch 时注入，任务期间不变
- main_replies:   子Agent请求信息后动态追加

淘汰策略：
- 任务结束后清理 main_replies
- conv_context 从主Agent传入时已按预算裁剪（保留最近3-5轮）
- domain_context 在子Agent内部按需构建，不缓存跨任务
"""

import logging
from typing import Dict, Any, List, Optional


class SubAgentContextBuilder:
    """子Agent上下文构建器

    负责将主Agent的完整上下文裁剪、投影为子Agent适用的精简上下文。
    每个子Agent类型有不同的分区预算和注入策略。
    """

    # 子Agent上下文分区优先级（数字越小越重要）
    SECTION_PRIORITIES = {
        'task_brief': 1,       # 任务摘要（不可裁剪）
        'user_profile': 2,     # 用户画像
        'conv_context': 3,     # 对话上下文
        'domain_context': 4,   # 领域上下文（RAG/工具）
        'main_replies': 5,     # 主Agent补充信息
    }

    # 子Agent类型的默认预算分配
    DEFAULT_BUDGETS = {
        'presale': {
            'total': 4000,
            'task_brief': 500,
            'user_profile': 300,
            'conv_context': 1200,
            'domain_context': 1500,
            'main_replies': 500,
        },
        'functional': {
            'total': 4000,
            'task_brief': 500,
            'user_profile': 300,
            'conv_context': 800,
            'domain_context': 1800,
            'main_replies': 600,
        },
    }

    def __init__(
            self,
            agent_type: str = "presale",
            config: Dict[str, Any] = None,
            logger: logging.Logger = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.agent_type = agent_type

        # 从配置或默认值获取预算
        sub_config = (config or {}).get('sub_agents', {}).get(
            'context_budgets', {}
        ).get(agent_type, {})

        defaults = self.DEFAULT_BUDGETS.get(agent_type, self.DEFAULT_BUDGETS['presale'])
        self.total_budget = sub_config.get('total', defaults['total'])
        self.section_budgets = {
            k: sub_config.get(k, defaults[k])
            for k in defaults if k != 'total'
        }

        self.logger.info(
            f"[SubAgentContext] {agent_type} 上下文预算: "
            f"总量={self.total_budget}, 分区={self.section_budgets}"
        )

    # ================================================================
    # 注入时机1：dispatch 时构建初始上下文
    # ================================================================

    def build_task_context(
            self,
            user_query: str,
            context: str,
            user_id: str = "",
            username: str = "",
            long_term_context: str = "",
            constraints: List[str] = None,
            tool_name: str = "",
    ) -> Dict[str, str]:
        """在 dispatch 时构建子Agent的初始上下文

        从主Agent的完整上下文中提取、裁剪出子Agent需要的精简版。

        Args:
            user_query: 用户原始请求
            context: 主Agent的短期记忆上下文（完整版）
            user_id: 当前用户ID
            username: 当前用户名
            long_term_context: 主Agent的长期记忆上下文
            constraints: 目标理解提取的约束条件
            tool_name: 意图识别出的工具名

        Returns:
            分区上下文 dict，key 是分区名，value 是裁剪后的内容
        """
        sections = {}

        # === 分区1: task_brief（任务摘要，不可裁剪） ===
        sections['task_brief'] = self._build_task_brief(
            user_query, constraints, tool_name
        )

        # === 分区2: user_profile（用户画像） ===
        sections['user_profile'] = self._build_user_profile(
            user_id, username, long_term_context
        )

        # === 分区3: conv_context（对话上下文，从主Agent裁剪） ===
        sections['conv_context'] = self._trim_conversation_context(context)

        # === 分区4: domain_context（领域上下文，此时为空，子Agent执行时按需填充） ===
        sections['domain_context'] = ""

        # === 分区5: main_replies（主Agent补充，初始为空） ===
        sections['main_replies'] = ""

        self.logger.info(
            f"[SubAgentContext] 初始上下文构建完成 | "
            + " | ".join(f"{k}={len(v)}" for k, v in sections.items() if v)
        )
        return sections

    def _build_task_brief(
            self,
            user_query: str,
            constraints: List[str] = None,
            tool_name: str = "",
    ) -> str:
        """构建任务摘要（P1，不可裁剪）"""
        budget = self.section_budgets.get('task_brief', 500)
        parts = [f"[当前任务] {user_query}"]
        if constraints:
            parts.append(f"[约束条件] {'; '.join(constraints)}")
        if tool_name:
            parts.append(f"[建议工具] {tool_name}")
        brief = "\n".join(parts)
        return brief[:budget]

    def _build_user_profile(
            self,
            user_id: str,
            username: str,
            long_term_context: str = "",
    ) -> str:
        """构建用户画像（P2，结构化，极少淘汰）

        从长期记忆中提取用户偏好、历史行为特征。
        """
        budget = self.section_budgets.get('user_profile', 300)
        parts = []
        if user_id:
            parts.append(f"user_id={user_id}")
        if username:
            parts.append(f"username={username}")

        profile = "[用户信息] " + ", ".join(parts) if parts else ""

        # 从长期记忆中提取偏好关键词
        if long_term_context:
            # 裁剪长期记忆：只保留预算允许的长度
            remaining = budget - len(profile) - 20
            if remaining > 50:
                lt_trimmed = long_term_context[:remaining]
                profile += f"\n[历史偏好] {lt_trimmed}"

        return profile[:budget]

    def _trim_conversation_context(self, context: str) -> str:
        """裁剪主Agent的对话上下文，保留对子Agent最有价值的部分

        策略：
        1. 保留历史摘要（如果有）
        2. 对话只保留最近3轮（6条消息），从尾部截取
        3. 去掉系统消息和重复的"任务已完成"
        """
        budget = self.section_budgets.get('conv_context', 1200)

        if not context:
            return ""

        lines = context.split('\n')

        # 分离不同部分
        summary_lines = []
        user_info_lines = []
        conv_lines = []
        in_summary = False

        for line in lines:
            if '[当前登录用户]' in line:
                # 用户信息已由 user_profile 分区处理，跳过避免重复
                continue
            elif '[历史对话摘要]' in line:
                in_summary = True
                summary_lines.append(line)
            elif '[最近对话]' in line:
                in_summary = False
            elif in_summary:
                summary_lines.append(line)
            else:
                # 过滤无信息量的行
                stripped = line.strip()
                if stripped and stripped != "ASSISTANT: 任务已完成":
                    conv_lines.append(line)

        # 摘要部分：保留但裁剪
        summary = "\n".join(summary_lines)
        summary_budget = min(len(summary), budget // 3)
        summary = summary[:summary_budget]

        # 对话部分：保留最近3轮（从尾部截取USER/ASSISTANT对）
        remaining_budget = budget - len(summary) - 10
        conv_text = "\n".join(conv_lines)
        if len(conv_text) > remaining_budget:
            # 从尾部保留
            conv_text = conv_text[-remaining_budget:]
            # 找到第一个完整消息的开头
            first_role = conv_text.find('\nUSER:')
            if first_role == -1:
                first_role = conv_text.find('\nASSISTANT:')
            if first_role > 0:
                conv_text = "...(早期对话已省略)" + conv_text[first_role:]

        result = ""
        if summary:
            result = summary + "\n"
        if conv_text.strip():
            result += conv_text

        return result.strip()[:budget]

    # ================================================================
    # 注入时机2：子Agent执行过程中动态构建领域上下文
    # ================================================================

    def build_domain_context(
            self,
            rag_results: list = None,
            mcp_data: str = "",
            tool_list: str = "",
    ) -> str:
        """构建领域上下文（P4，子Agent执行过程中按需调用）

        Args:
            rag_results: RAG检索结果
            mcp_data: MCP工具返回的实时数据
            tool_list: 可用工具列表描述

        Returns:
            格式化的领域上下文
        """
        budget = self.section_budgets.get('domain_context', 1500)
        parts = []

        # RAG结果
        if rag_results:
            rag_text = self._format_rag_results(rag_results)
            parts.append(f"[知识库信息]\n{rag_text}")

        # MCP实时数据
        if mcp_data:
            parts.append(f"[实时数据]\n{mcp_data}")

        # 工具列表
        if tool_list:
            parts.append(f"[可用工具]\n{tool_list}")

        domain = "\n\n".join(parts)

        # 超预算时按优先级裁剪：保留RAG > MCP > 工具列表
        if len(domain) > budget:
            domain = domain[:budget - 20] + "\n...(部分领域信息已省略)"

        return domain

    def _format_rag_results(self, results: list) -> str:
        """格式化RAG检索结果为精简文本"""
        parts = []
        for i, r in enumerate(results, 1):
            doc = r.get("document", r.get("text", ""))
            product = r.get("metadata", {}).get("product_name", "")
            # 每条结果限制在300字符
            if len(doc) > 300:
                doc = doc[:300] + "..."
            prefix = f"[{product}] " if product else ""
            parts.append(f"{i}. {prefix}{doc}")
        return "\n".join(parts)

    # ================================================================
    # 注入时机3：主Agent回复后追加
    # ================================================================

    def append_main_reply(
            self,
            current_replies: str,
            question: str,
            answer: str,
    ) -> str:
        """追加主Agent的回复到 main_replies 分区

        淘汰策略：如果累积回复超预算，保留最新的回复。
        """
        budget = self.section_budgets.get('main_replies', 500)
        new_entry = f"问: {question}\n答: {answer}"

        if current_replies:
            updated = current_replies + "\n\n" + new_entry
        else:
            updated = "[主Agent补充信息]\n" + new_entry

        # 超预算时淘汰最早的回复
        if len(updated) > budget:
            # 保留标题 + 最近的回复
            entries = updated.split("\n\n")
            while len("\n\n".join(entries)) > budget and len(entries) > 2:
                entries.pop(1)  # 保留标题[0]，删除最早的回复
            updated = "\n\n".join(entries)
            if len(updated) > budget:
                updated = updated[:budget - 20] + "\n...(早期回复已省略)"

        return updated

    # ================================================================
    # 组装完整上下文（给LLM的prompt使用）
    # ================================================================

    def assemble(self, sections: Dict[str, str]) -> str:
        """将所有分区组装为最终上下文字符串

        按优先级顺序拼接，确保不超总预算。

        Args:
            sections: 各分区内容 dict

        Returns:
            组装后的上下文字符串
        """
        ordered = sorted(
            [(k, v) for k, v in sections.items() if v and v.strip()],
            key=lambda x: self.SECTION_PRIORITIES.get(x[0], 99)
        )

        result_parts = []
        total_len = 0
        for name, content in ordered:
            if total_len + len(content) > self.total_budget:
                # 超总预算，裁剪当前分区
                remaining = self.total_budget - total_len - 20
                if remaining > 50:
                    result_parts.append(content[:remaining] + "...(已裁剪)")
                break
            result_parts.append(content)
            total_len += len(content) + 2  # +2 for \n\n

        assembled = "\n\n".join(result_parts)
        self.logger.debug(
            f"[SubAgentContext] 组装完成 | 总长度={len(assembled)} | "
            f"分区={[k for k, _ in ordered]}"
        )
        return assembled

    # ================================================================
    # 淘汰策略：任务结束时清理
    # ================================================================

    @staticmethod
    def cleanup_after_task(sections: Dict[str, str]) -> Dict[str, str]:
        """任务结束后清理上下文

        淘汰策略：
        - main_replies: 完全清理（任务级生命周期）
        - domain_context: 完全清理（任务级生命周期）
        - task_brief: 完全清理
        - conv_context: 保留（由主Agent管理生命周期）
        - user_profile: 保留（会话级生命周期）
        """
        sections['main_replies'] = ""
        sections['domain_context'] = ""
        sections['task_brief'] = ""
        return sections
