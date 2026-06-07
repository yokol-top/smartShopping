"""
统一 ReAct 执行引擎 (Unified ReAct Executor)

将 TaskPlanner._react_loop 和 DynamicSubAgent._execute_with_tools 中
重复的 ReAct 循环逻辑提取到此处，消除双重实现。

使用方：
- TaskPlanner._react_loop  → 创建 UnifiedReActExecutor，调用 execute()
- DynamicSubAgent._execute_with_tools → 创建 UnifiedReActExecutor（带 allowed_tools），调用 execute()

设计原则：
- 工具白名单由调用方通过 ReActConfig.allowed_tools 传入（None = 全部工具）
- RAG 搜索能力可选（enable_rag=False 时不暴露 search_knowledge 动作）
- 上下文窗口管理可选（传入 context_manager 时启用）
- 最大迭代次数、温度均可配置
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .exceptions import NeedUserInputException


@dataclass
class ReActConfig:
    """ReAct 循环配置"""
    max_iterations: int = 5
    temperature: float = 0.3
    allowed_tools: Optional[List[str]] = None   # None = 不限制（全部工具可用）
    enable_rag: bool = True                      # 是否开放 search_knowledge 动作
    system_role: str = ""                        # 子Agent 角色描述（影响 system prompt）


@dataclass
class ReActStep:
    """单轮 ReAct 步骤记录"""
    iteration: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""


class UnifiedReActExecutor:
    """
    统一 ReAct 执行引擎

    替代 TaskPlanner._react_loop 和 DynamicSubAgent._execute_with_tools。
    核心逻辑只在此一处维护。
    """

    def __init__(
        self,
        llm_client,
        mcp_manager=None,
        rag_engine=None,
        context_manager=None,
        logger: logging.Logger = None,
    ):
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.rag_engine = rag_engine
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)

    # ================================================================
    # 主入口
    # ================================================================

    def execute(
        self,
        task_desc: str,
        context: str = "",
        long_term_context: str = "",
        verbose: bool = False,
        config: ReActConfig = None,
    ) -> str:
        """
        执行 ReAct 循环直到任务完成或达到最大迭代次数。

        Args:
            task_desc: 任务描述（用户的原始请求或子任务描述）
            context: 对话上下文（短期记忆）
            long_term_context: 长期记忆上下文
            verbose: 是否打印过程信息
            config: ReAct 配置（None 时使用默认值）

        Returns:
            最终答案字符串
        """
        cfg = config or ReActConfig()
        steps: List[ReActStep] = []

        # 预处理上下文（上下文窗口管理）
        context, long_term_context = self._manage_context(
            context, long_term_context, cfg
        )

        # 获取可用工具描述（按任务描述过滤，减少无关工具噪音）
        tools_desc = self._get_tools_desc(cfg, task_desc=task_desc)

        for iteration in range(cfg.max_iterations):
            if verbose:
                print(f"\n🔄 第 {iteration + 1} 轮思考\n")

            step = ReActStep(iteration=iteration + 1)

            # 构建 prompt
            history_text = self._format_history(steps)
            prompt = self._build_prompt(
                task_desc, context, long_term_context,
                tools_desc, history_text, cfg,
            )

            # 调用 LLM
            try:
                raw = self.llm_client.generate(
                    prompt=prompt, temperature=cfg.temperature
                ).strip()
            except Exception as e:
                self.logger.error(f"[UnifiedReAct] LLM 调用失败: {e}")
                break

            # 解析响应
            step.thought, step.action, step.action_input = self._parse_response(raw)

            if verbose:
                print(f"💭 思考: {step.thought}")
                print(f"⚡ 行动: {step.action}")
                print(f"📥 输入: {step.action_input}\n")

            # finish 动作 → 直接返回
            if step.action == "finish":
                answer = step.action_input.get("answer", "")
                if not answer:
                    answer = self._summarize_steps(task_desc, steps)
                return answer

            # 重复动作检测
            if self._is_duplicate(step, steps):
                self.logger.warning("[UnifiedReAct] 检测到重复动作，强制结束")
                if verbose:
                    print("⚠️ 检测到重复动作，自动结束\n")
                return self._summarize_steps(task_desc, steps)

            # 执行动作，获取 Observation
            step.observation = self._execute_action(step, task_desc, context, cfg)

            if verbose:
                preview = step.observation[:200] + ("..." if len(step.observation) > 200 else "")
                print(f"👀 观察: {preview}\n")

            steps.append(step)

            # 上下文管理：更新 tool_results 预算
            if self.context_manager and steps:
                history_text = self._format_history(steps)
                history_text = self.context_manager.manage_section(
                    "tool_results", history_text
                )

        # 达到最大迭代，汇总返回
        return self._summarize_steps(task_desc, steps)

    # ================================================================
    # Prompt 构建
    # ================================================================

    def _build_prompt(
        self,
        task_desc: str,
        context: str,
        long_term_context: str,
        tools_desc: str,
        history_text: str,
        cfg: ReActConfig,
    ) -> str:
        """构建 ReAct 循环的 prompt"""
        parts = []

        if long_term_context:
            parts.append(long_term_context)
            parts.append("")

        # 角色设定
        role_line = f"你是{cfg.system_role}。" if cfg.system_role else "你是一个能够使用工具完成任务的 AI 助手。"

        # 可用动作说明
        actions_desc = ""
        if cfg.enable_rag and self.rag_engine:
            actions_desc += "- search_knowledge: {\"query\": \"搜索关键词\"}\n"
        actions_desc += "- call_mcp_tool: {\"tool_name\": \"工具名\", \"parameters\": {...}}\n"
        actions_desc += "- need_input: {\"question\": \"需要用户提供的具体信息\"}\n"
        actions_desc += "- finish: {\"answer\": \"最终答案\"}"

        parts.append(f"""{role_line}

用户任务: {task_desc}

对话历史:
{context or "(无)"}

可用工具:
{tools_desc or "(无可用工具)"}

已完成的步骤:
{history_text if history_text else "（尚未开始）"}

**关键规则**:
1. 检查"已完成的步骤"中的观察结果，如果任务已完成，立即使用 Action: finish。
2. 不要重复执行已经成功的操作。
3. 缺少必要参数时，用 Action: finish 告知用户需要补充什么。
4. 调用工具时必须提供工具要求的**全部必填参数**，从对话历史和已完成步骤中查找这些值。
5. 使用 finish 时，answer 是面向用户的自然语言回答；**商品ID（P001等格式）、订单号（ORD-XXX）、地址ID（ADDR-XXX）、银行卡ID（CARD-XXX）等关键标识符必须原样保留在回答中**，不可省略，不得用名称替代ID。
6. 若需要用户补充关键信息（如选择商品、确认地址）才能继续执行，使用 Action: need_input 告知。

严格按以下格式回复：
Thought: [总结已有结果，分析下一步]
Action: [search_knowledge / call_mcp_tool / need_input / finish]
Action Input: [JSON格式]

Action Input 格式：
{actions_desc}

如果任务已完成或缺少参数，使用 Action: finish。""")

        return "\n".join(parts)

    # ================================================================
    # 动作执行
    # ================================================================

    def _execute_action(
        self,
        step: ReActStep,
        task_desc: str,
        context: str,
        cfg: ReActConfig,
    ) -> str:
        """执行 ReAct 步骤的动作，返回 Observation"""
        action = step.action
        action_input = step.action_input

        if action == "need_input":
            question = action_input.get("question", "需要您提供更多信息，请补充后再试。")
            self.logger.info(f"[UnifiedReAct] 执行中途需要用户输入: {question[:80]}")
            raise NeedUserInputException(question)

        if action == "search_knowledge" and cfg.enable_rag and self.rag_engine:
            return self._exec_rag(action_input.get("query", task_desc), context)

        if action == "call_mcp_tool":
            tool_name = action_input.get("tool_name", "")
            params = action_input.get("parameters", {})

            # 工具白名单检查
            if cfg.allowed_tools is not None and tool_name not in cfg.allowed_tools:
                return f"工具 '{tool_name}' 不在此Agent的权限范围内（白名单: {cfg.allowed_tools}）"

            return self._exec_mcp(tool_name, params)

        # 未知动作 → 尝试从响应中提取 finish 答案
        return f"未知动作: {action}，请使用 finish 返回结果"

    def _exec_rag(self, query: str, context: str) -> str:
        """执行 RAG 检索"""
        try:
            results = self.rag_engine.query(query, context=context, top_k=3)
            if not results:
                return "知识库中未找到相关内容"
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(f"{i}. {r.get('content', '')[:500]}")
            return "\n".join(parts)
        except Exception as e:
            self.logger.error(f"[UnifiedReAct] RAG 搜索失败: {e}")
            return f"知识库搜索失败: {e}"

    def _exec_mcp(self, tool_name: str, params: Dict[str, Any]) -> str:
        """执行 MCP 工具调用"""
        if not self.mcp_manager:
            return "MCP 管理器未初始化"
        if not tool_name:
            return "工具名不能为空"
        try:
            # 查找工具所在服务器
            all_tools = self.mcp_manager.get_available_tools(use_cache=True)
            server_name = None
            for t in all_tools:
                if t.get("name") == tool_name:
                    server_name = t.get("server")
                    break
            if not server_name:
                return f"工具 '{tool_name}' 不可用"
            result = self.mcp_manager.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                parameters=params,
            )
            if isinstance(result, dict) and "error" in result:
                return f"工具调用失败: {result['error']}"
            # 提取 MCP 返回值中的 result 字段（自然语言内容），避免把原始 dict 暴露给用户
            if isinstance(result, dict):
                return result.get("result", str(result))
            return str(result)
        except Exception as e:
            self.logger.error(f"[UnifiedReAct] 工具 {tool_name} 调用失败: {e}")
            return f"工具调用异常: {e}"

    # ================================================================
    # 工具列表
    # ================================================================

    def _get_tools_desc(self, cfg: ReActConfig, task_desc: str = "") -> str:
        """获取工具描述（先按任务描述做阶段过滤，再按白名单过滤）"""
        lines = []

        # RAG 搜索（可选）
        if cfg.enable_rag and self.rag_engine:
            lines.append("- search_knowledge: 从知识库检索商品信息/FAQ  参数: query*: 搜索关键词")

        if not self.mcp_manager:
            return "\n".join(lines) if lines else "(无可用工具)"

        # 按任务描述过滤（减少无关工具噪音）
        all_tools = self.mcp_manager.get_tools_for_context(
            intent_type='mcp_execute', query=task_desc
        )
        for t in all_tools:
            name = t.get("name", "")
            # 白名单过滤
            if cfg.allowed_tools is not None and name not in cfg.allowed_tools:
                continue
            desc = t.get("description", "")[:80]
            # 提取参数：先展示必填，再展示可选
            # 必须让 LLM 看到可选参数的名字，否则会凭描述文字猜测参数名导致调用失败
            schema = t.get("inputSchema", {})
            required_params = schema.get("required", [])
            all_properties = schema.get("properties", {})
            param_hints = []
            for p in required_params:
                prop_info = all_properties.get(p, {})
                param_hints.append(f"{p}*: {prop_info.get('description', prop_info.get('type', ''))[:40]}")
            for p, prop_info in all_properties.items():
                if p not in required_params:
                    param_hints.append(f"{p}: {prop_info.get('description', prop_info.get('type', ''))[:40]}")
            params_str = ("  参数: " + "; ".join(param_hints)) if param_hints else ""
            lines.append(f"- {name}: {desc}{params_str}")

        return "\n".join(lines) if lines else "(无可用工具)"

    # ================================================================
    # 解析
    # ================================================================

    def _parse_response(self, raw: str):
        """解析 LLM 响应，提取 Thought/Action/Action Input。

        兼容两种格式：
        1. 文本格式（期望）：Thought: ...\nAction: ...\nAction Input: {...}
        2. JSON 格式（LLM 有时返回）：{"Thought":"...","Action":"...","Action Input":{...}}
        """
        thought = ""
        action = "finish"
        action_input = {"answer": raw}

        # ── 先尝试 JSON 格式 ──────────────────────────────────────────
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                thought = parsed.get("Thought") or parsed.get("thought") or ""
                raw_action = parsed.get("Action") or parsed.get("action") or "finish"
                action = str(raw_action).strip().lower()
                ai = (
                    parsed.get("Action Input")
                    or parsed.get("action_input")
                    or parsed.get("ActionInput")
                )
                if isinstance(ai, dict):
                    action_input = ai
                elif ai is not None:
                    action_input = {"answer": str(ai)}
                elif action == "finish":
                    # LLM 有时把 answer 放在顶层而不是 action_input 里
                    # 如: {"action":"finish","answer":"..."} 而非标准格式
                    top_answer = parsed.get("answer") or parsed.get("Answer")
                    if top_answer:
                        action_input = {"answer": str(top_answer)}
                return thought, action, action_input
            except (json.JSONDecodeError, Exception):
                pass  # JSON 解析失败，回退到正则

        # ── 正则解析（文本格式）────────────────────────────────────────
        # 提取 Thought
        t_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", raw, re.DOTALL | re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).strip()

        # 提取 Action
        a_match = re.search(r"Action:\s*(\w+)", raw, re.IGNORECASE)
        if a_match:
            action = a_match.group(1).strip().lower()

        # 提取 Action Input（支持嵌套 JSON，按括号深度追踪）
        ai_header = re.search(r"Action Input:\s*", raw, re.IGNORECASE)
        if ai_header:
            brace_start = raw.find("{", ai_header.end())
            if brace_start >= 0:
                depth, in_str, esc = 0, False, False
                json_str = None
                for i in range(brace_start, len(raw)):
                    c = raw[i]
                    if esc:
                        esc = False
                        continue
                    if c == "\\" and in_str:
                        esc = True
                        continue
                    if c == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = raw[brace_start: i + 1]
                            break
                if json_str:
                    try:
                        action_input = json.loads(json_str)
                    except json.JSONDecodeError:
                        try:
                            import ast
                            action_input = ast.literal_eval(json_str)
                        except Exception:
                            pass

        return thought, action, action_input

    # ================================================================
    # 辅助
    # ================================================================

    def _format_history(self, steps: List[ReActStep]) -> str:
        """格式化已完成步骤的历史"""
        if not steps:
            return ""
        parts = []
        for s in steps:
            parts.append(
                f"第{s.iteration}轮 - Thought: {s.thought[:100]}\n"
                f"  Action: {s.action} | Input: {str(s.action_input)[:100]}\n"
                f"  Observation: {s.observation[:300]}"
            )
        return "\n\n".join(parts)

    def _summarize_steps(self, task_desc: str, steps: List[ReActStep]) -> str:
        """将执行步骤汇总为最终回答"""
        if not steps:
            return "未能完成任务"

        # 提取最后一个有实质内容的 observation
        for s in reversed(steps):
            if s.observation and not s.observation.startswith("未知动作"):
                obs = s.observation
                break
        else:
            obs = steps[-1].observation if steps else ""

        # 尝试用 LLM 生成用户友好的汇总
        try:
            obs_parts = "\n".join(
                f"步骤{s.iteration}: {s.observation[:200]}"
                for s in steps if s.observation
            )
            prompt = (
                f"用户任务：{task_desc}\n\n"
                f"执行结果（内部日志，仅供参考）：\n{obs_parts[:1000]}\n\n"
                "请将以上执行结果转换为面向用户的自然语言回复。要求：\n"
                "1. 语气友好，使用日常口语，不暴露内部工具名称或JSON格式\n"
                "2. 如果是订单操作，突出订单号、商品、金额、地址等关键信息\n"
                "3. 如果操作成功，直接告知结果；如果失败，说明原因并给出建议\n"
                "4. 简洁清晰，不超过200字\n\n"
                "用户友好的回复："
            )
            return self.llm_client.generate(
                prompt=prompt, temperature=0.3, max_tokens=300
            ).strip()
        except Exception:
            return obs or "任务执行完成"

    def _is_duplicate(self, step: ReActStep, history: List[ReActStep]) -> bool:
        """检测是否与历史步骤完全重复"""
        for prev in history:
            if (prev.action == step.action
                    and prev.action_input == step.action_input
                    and prev.action != "finish"):
                return True
        return False

    def _manage_context(
        self,
        context: str,
        long_term_context: str,
        cfg: ReActConfig,
    ):
        """通过 context_manager 管理上下文窗口预算（可选）"""
        if not self.context_manager:
            return context, long_term_context
        try:
            active = ["short_term_memory", "tools", "tool_results"]
            if long_term_context:
                active.append("long_term_memory")
            self.context_manager.set_active_sections(active)
            context = self.context_manager.manage_section("short_term_memory", context)
            long_term_context = self.context_manager.manage_section(
                "long_term_memory", long_term_context
            )
        except Exception as e:
            self.logger.debug(f"[UnifiedReAct] context_manager 调用失败（忽略）: {e}")
        return context, long_term_context
