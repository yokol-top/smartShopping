import uuid
import time
from typing import Dict, Any, List  # noqa: F401 - used in type hints

from mcp_manager_module import MCPManager
from memory import ShortTermMemory, LongTermMemory
from observability import AgentTracer, get_tracer
from rag.rag_engine import RAGEngine
from .intent_recognizer import IntentRecognizer, TaskComplexity
from .goal_understanding import GoalUnderstanding
from .task_planner import TaskPlanner
from .task_evaluator import TaskEvaluator
from utils import setup_logger, ConfigLoader, LLMClient
from input_gate import InputValidator
from security import OutputGuard, RateLimiter
from tool_manager import ToolManager
from evaluation import AgentEvaluator
from context import ContextWindowManager
from session import SessionManager
from .agent_router import AgentRouter
from .product_knowledge import build_product_knowledge_base


class SmartAgent:
    """智能Agent：支持多轮对话、记忆、RAG和MCP"""

    def __init__(self, config_path: str = "./config/settings.yaml",
                 user_id: str = None, username: str = None, session_id: str = None):
        """
        Args:
            config_path: 配置文件路径
            user_id: 登录用户ID（登录后传入，如 UID-8888）
            username: 登录用户名（如 张三）
            session_id: 会话ID（恢复历史会话时传入，新会话不传）
        """
        # 加载配置
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.user_id = user_id
        self.username = username

        # 设置日志
        self.logger = setup_logger(
            name="SmartAgent",
            log_file=self.config.get('logging', {}).get('file', './logs/agent.log'),
            level=self.config.get('logging', {}).get('level', 'INFO'),
            console=self.config.get('logging', {}).get('console', True)
        )

        self.logger.info("=" * 60)
        self.logger.info("初始化 SmartAgent")
        self.logger.info("=" * 60)

        # 初始化可观测性追踪器
        obs_config = self.config.get('observability', {})
        self.agent_tracer = AgentTracer(obs_config)
        if obs_config.get('enabled', False):
            self.logger.info(f"可观测性追踪已启用 - exporter: {obs_config.get('exporter', 'console')}")
        else:
            self.logger.info("可观测性追踪未启用")

        # 初始化会话管理器
        self.session_manager = SessionManager(config=self.config, logger=self.logger)

        # 会话ID：优先使用传入的 session_id（恢复历史会话）
        if session_id:
            self.session_id = session_id
            self._is_resumed_session = True
        else:
            self.session_id = str(uuid.uuid4())
            self._is_resumed_session = False
        self.logger.info(f"会话ID: {self.session_id} | 用户: {self.user_id or 'anonymous'}")

        # 加载系统提示词
        system_prompt_config = self.config.get('system_prompt', {})
        system_prompt_enabled = system_prompt_config.get('enabled', True)
        system_prompt_content = system_prompt_config.get('content', '') if system_prompt_enabled else None

        # 初始化LLM客户端（传入全局系统提示词）
        self.logger.info("初始化LLM客户端...")
        self.llm_client = LLMClient(
            api_key=self.config.get('llm', {}).get('api_key', 'EMPTY'),
            base_url=self.config.get('llm', {}).get('base_url', 'http://localhost:11434/v1'),
            model=self.config.get('llm', {}).get('model', 'qwen2.5:latest'),
            temperature=self.config.get('llm', {}).get('temperature', 0.7),
            max_tokens=self.config.get('llm', {}).get('max_tokens', 2000),
            system_prompt=system_prompt_content,
            logger=self.logger
        )

        if system_prompt_enabled and system_prompt_content:
            self.logger.info("全局系统提示词已设置到LLM客户端")

        # 初始化RAG引擎（先初始化以获取嵌入模型）
        self.logger.info("初始化RAG引擎...")
        # 解析嵌入模型路径（相对于配置文件目录）
        embedding_config = self.config.get('embedding', {}).copy()
        if embedding_config.get('provider') == 'local' and embedding_config.get('model'):
            embedding_config['model'] = self.config_loader.resolve_path(embedding_config['model'])
            self.logger.info(f"解析嵌入模型路径: {embedding_config['model']}")

        # 创建配置副本并更新嵌入配置
        config_with_resolved_paths = self.config.copy()
        config_with_resolved_paths['embedding'] = embedding_config

        self.rag_engine = RAGEngine(config_with_resolved_paths, llm_client=self.llm_client, logger=self.logger)

        # 初始化记忆系统
        self.logger.info("初始化记忆系统...")

        # 获取短期记忆配置
        short_term_config = self.config.get('memory', {}).get('short_term', {})
        rolling_summary_config = short_term_config.get('rolling_summary', {})

        self.short_term_memory = ShortTermMemory(
            max_messages=short_term_config.get('max_messages', 10),
            logger=self.logger
        )

        # 配置滚动总结参数
        if rolling_summary_config.get('enabled', True):
            trigger_count = rolling_summary_config.get('trigger_count', 5)
            self.short_term_memory.summary_trigger_count = trigger_count

            # 保存滚动总结配置供后续使用
            self.rolling_summary_min_messages = rolling_summary_config.get('min_messages_for_summary', 2)
            self.rolling_summary_ratio = rolling_summary_config.get('summary_ratio', 0.5)

            self.logger.info(
                f"滚动总结已启用 - 触发阈值: {trigger_count}, 最小消息数: {self.rolling_summary_min_messages}, 总结比例: {self.rolling_summary_ratio}")
        else:
            # 禁用滚动总结（设置为一个很大的数，实际上不会触发）
            self.short_term_memory.summary_trigger_count = 999999
            self.rolling_summary_min_messages = 999999
            self.rolling_summary_ratio = 0.5
            self.logger.info("滚动总结已禁用")

        # 使用 RAG 引擎的嵌入模型初始化长期记忆
        self.long_term_memory = LongTermMemory(
            persist_directory=self.config.get('memory', {}).get('long_term', {}).get('persist_directory',
                                                                                     './data/long_term_memory'),
            embedding_function=self.rag_engine.embeddings,
            logger=self.logger
        )
        self.long_term_memory.create_session(self.session_id)

        # 如果是恢复的历史会话，加载历史消息到短期记忆
        if self._is_resumed_session:
            self._load_session_history()
        # 如果是新会话且有用户，在 session_manager 中创建记录
        elif self.user_id:
            self.session_manager.create_session(
                user_id=self.user_id, title='新会话'
            )
            # 用刚创建的 session_id 替换随机生成的
            sessions = self.session_manager.list_sessions(self.user_id, limit=1)
            if sessions:
                self.session_id = sessions[0].session_id
                self.logger.info(f"新会话已持久化: {self.session_id[:8]}...")

        # 初始化MCP管理器
        mcp_config_file = self.config.get('mcp', {}).get('config_file', './config/mcp_servers.yaml')
        self.logger.info(f"初始化MCP管理器: {mcp_config_file}")
        self.mcp_manager = MCPManager(mcp_config_file, logger=self.logger)

        # 初始化意图识别器
        self.logger.info("初始化意图识别器...")
        self.intent_recognizer = IntentRecognizer(
            self.config,
            mcp_manager=self.mcp_manager,
            llm_client=self.llm_client,
            logger=self.logger
        )

        # 初始化任务评估器
        self.logger.info("初始化任务评估器...")
        self.task_evaluator = TaskEvaluator(
            self.config,
            llm_client=self.llm_client,
            logger=self.logger
        )

        # 输入层：输入验证与过滤
        self.logger.info("初始化输入层...")
        self.input_validator = InputValidator(self.config, logger=self.logger)

        # 输出层：安全审查
        self.logger.info("初始化输出安全审查...")
        self.output_guard = OutputGuard(self.config, logger=self.logger)

        # 目标理解层：增强意图识别，支持澄清机制
        self.logger.info("初始化目标理解层...")
        self.goal_understanding = GoalUnderstanding(
            self.config,
            llm_client=self.llm_client,
            intent_recognizer=self.intent_recognizer,
            long_term_memory=self.long_term_memory,
            logger=self.logger,
        )

        # 频率限制器
        self.logger.info("初始化频率限制器...")
        self.rate_limiter = RateLimiter(config=self.config, logger=self.logger)

        # 工具管理器（统一管理本地工具 + MCP 工具）
        self.logger.info("初始化工具管理器...")
        self.tool_manager = ToolManager(
            config=self.config,
            mcp_manager=self.mcp_manager,
            rate_limiter=self.rate_limiter,
            logger=self.logger,
        )

        # 注册本地工具
        self._register_local_tools()

        # 上下文窗口管理器
        self.logger.info("初始化上下文窗口管理器...")
        self.context_manager = ContextWindowManager(
            config=self.config,
            tool_registry=self.tool_manager,
            llm_client=self.llm_client,
            logger=self.logger
        )

        # 任务规划器
        self.logger.info("初始化任务规划器...")
        self.task_planner = TaskPlanner(
            self.config,
            llm_client=self.llm_client,
            mcp_manager=self.mcp_manager,
            rag_engine=self.rag_engine,
            evaluator=self.task_evaluator,
            context_manager=self.context_manager,
            tool_manager=self.tool_manager,
            logger=self.logger
        )

        # 子Agent路由器（售前客服 + 功能处理）
        self.logger.info("初始化子Agent路由器...")
        sub_agent_config = self.config.get('sub_agents', {})
        self.agent_router = AgentRouter(
            llm_client=self.llm_client,
            rag_engine=self.rag_engine,
            mcp_manager=self.mcp_manager,
            config=sub_agent_config,
            logger=self.logger,
        )

        # 商品知识库（售前子Agent依赖）
        if sub_agent_config.get('presale', {}).get('init_product_kb', True):
            try:
                self.logger.info("初始化商品FAQ知识库...")
                faq_count = build_product_knowledge_base(self.rag_engine, self.logger)
                self.logger.info(f"商品FAQ知识库初始化完成，共 {faq_count} 条")
            except Exception as e:
                self.logger.warning(f"商品FAQ知识库初始化失败（不影响主功能）: {e}")

        # Agent评估器
        self.logger.info("初始化评估层...")
        self.agent_evaluator = AgentEvaluator(self.config, logger=self.logger)

        # 对话计数器（用于长期记忆总结）
        self.conversation_count = 0
        self.summary_threshold = self.config.get('memory', {}).get('long_term', {}).get('summary_threshold', 5)

        # 响应摘要配置
        response_summary_config = self.config.get('memory', {}).get('short_term', {}).get('response_summary', {})
        self.response_summary_enabled = response_summary_config.get('enabled', True)
        self.response_max_length = response_summary_config.get('max_length', 300)
        self.logger.info(
            f"响应摘要功能: {'启用' if self.response_summary_enabled else '禁用'}, 最大长度阈值: {self.response_max_length}")

        self.logger.info("SmartAgent 初始化完成！")
        self.logger.info("=" * 60)

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        与用户进行对话

        完整流程：
        输入验证 → 记忆上下文 → 目标理解（含澄清） → 任务规划与执行 → 输出审查 → 记忆更新 → Agent评估

        Args:
            user_input: 用户输入
            verbose: 是否显示详细处理过程

        Returns:
            Agent的回复
        """
        tracer = get_tracer()
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        with tracer.start_span("agent.chat", {
            "agent.session_id": self.session_id,
            "agent.user_input": user_input,
            "agent.conversation_count": self.conversation_count,
            "agent.task_id": task_id,
        }):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"用户输入: {user_input}")
            self.logger.info(f"{'=' * 60}")

            # ========== 0. 输入层：验证与过滤 ==========
            validation = self.input_validator.validate(user_input)
            if not validation.is_valid:
                self.logger.warning(f"输入被拦截: {validation.risk_level} - {validation.block_reason}")
                if verbose:
                    print(f"\n⚠️ 输入安全检查: {validation.block_reason}\n")
                return f"抱歉，您的输入未通过安全检查：{validation.block_reason}"

            if verbose and validation.risk_level not in ("low",):
                print(f"\n🛡️ 输入安全检查: 风险等级 {validation.risk_level}\n")

            # 添加到短期记忆
            self.short_term_memory.add_message("user", user_input)
            self.short_term_memory.increment_message_count()

            # 持久化用户消息到会话存储
            if self.user_id:
                self.session_manager.add_message(self.session_id, 'user', user_input)

            # 检查是否需要触发滚动总结
            if self.short_term_memory.should_trigger_summary():
                self._generate_short_term_summary(verbose)

            # 检查是否需要压缩历史摘要
            if self.short_term_memory.needs_compression():
                self._compress_historical_summary()

            # 获取上下文（包含历史摘要）- 增加上下文窗口
            with tracer.start_span("memory.retrieve_context"):
                context = self.short_term_memory.get_context_string(last_n=8)

                # 注入当前登录用户信息，让LLM在调用工具时使用正确的用户标识
                if self.user_id:
                    user_ctx = f"[当前登录用户] user_id={self.user_id}"
                    if self.username:
                        user_ctx += f", username={self.username}"
                    context = user_ctx + "\n" + context

                # Orchestrator enriche：解析用户对子Agent结果的引用（如"方案一"→具体商品）
                sub_agent_enabled = self.config.get('sub_agents', {}).get('enabled', True)
                if sub_agent_enabled and hasattr(self, 'agent_router'):
                    context = self.agent_router.enrich_context(user_input, context)

                self.logger.info(f"从短期记忆中获取到的上下文：{context}")

                # 检索相关的长期记忆（语义搜索历史对话）
                relevant_long_term = self._retrieve_relevant_long_term_context(user_input, top_k=3)
                if relevant_long_term:
                    self.logger.info(f"检索到相关历史对话")

                tracer.set_span_attributes({
                    "memory.context_length": len(context),
                    "memory.has_long_term": bool(relevant_long_term),
                })

            # ========== 1. 目标理解层（增强意图识别 + 澄清机制） ==========
            # 获取orchestrator结构化记忆，供参数完整性检查使用
            orchestrator_ctx = ""
            if sub_agent_enabled and hasattr(self, 'agent_router'):
                orchestrator_ctx = self.agent_router.memory.get_context_for_sub_agent()
            self.logger.info(
                f"[主Agent] 传入目标理解层 | user_input: {user_input[:80]} | "
                f"context_len: {len(context)} | orch_ctx_len: {len(orchestrator_ctx)}"
            )
            self.logger.debug(
                f"[主Agent] context(last400): {context[-400:] if context else '(空)'}"
            )
            if orchestrator_ctx:
                self.logger.debug(
                    f"[主Agent] orchestrator_ctx: {orchestrator_ctx[:300]}"
                )
            goal_result = self.goal_understanding.understand(
                user_input, context, orchestrator_context=orchestrator_ctx
            )
            intent_result = goal_result.intent_result
            self.logger.info(
                f"目标理解结果: intent={intent_result.intent_type.value} | "
                f"tool={intent_result.tool_name or '(无)'} | "
                f"complexity={intent_result.complexity.value} | "
                f"confidence={goal_result.confidence:.2f} | "
                f"需要澄清: {goal_result.needs_clarification}"
            )

            if verbose:
                print(f"\n🔍 意图识别: {intent_result.intent_type.value} | 复杂度: {intent_result.complexity.value} | 置信度: {goal_result.confidence:.2f}")
                print(f"📝 {self.intent_recognizer.explain(intent_result)}")
                if goal_result.user_goal and goal_result.user_goal != user_input:
                    print(f"🎯 用户目标: {goal_result.user_goal}")
                if goal_result.constraints:
                    print(f"📌 约束条件: {', '.join(goal_result.constraints)}")
                print()

            # 如果需要澄清，直接返回澄清问题
            if goal_result.needs_clarification:
                clarification = goal_result.clarification_question
                self.short_term_memory.add_message("assistant", clarification)
                self.short_term_memory.increment_message_count()
                if verbose:
                    print(f"❓ 需要澄清\n")
                return clarification

            # 记录任务开始（Agent评估）
            self.agent_evaluator.task_started(
                task_id=task_id,
                query=user_input,
                intent_type=intent_result.intent_type.value,
                complexity=intent_result.complexity.value,
            )

            # ========== 2. 子Agent路由 + 任务规划与执行 ==========
            sub_agent_enabled = self.config.get('sub_agents', {}).get('enabled', True)
            route_target = None
            if sub_agent_enabled:
                route_target = self.agent_router.route(user_input, intent_result, context)

            # 路由决策：
            # - 简单任务 + 匹配子Agent → 子Agent直接处理（单步执行）
            # - 中等/复杂任务 → 主Agent规划执行（保留复杂性识别、任务规划、依赖管理）
            # - 无匹配子Agent → 主Agent原有逻辑处理
            use_sub_agent_direct = (
                route_target
                and intent_result.complexity == TaskComplexity.SIMPLE
            )

            if use_sub_agent_direct:
                # 简单任务：子Agent直接处理
                if verbose:
                    route_label = "🛒 售前客服" if route_target == AgentRouter.ROUTE_PRESALE else "⚙️ 功能处理"
                    print(f"{route_label} 子Agent正在处理...\n")
                self.logger.info(f"简单任务路由到子Agent: {route_target}")
                response = self.agent_router.dispatch(
                    route=route_target,
                    user_query=user_input,
                    context=context,
                    user_id=self.user_id or "",
                    username=self.username or "",
                    tool_name=getattr(intent_result, 'tool_name', '') or '',
                    long_term_context=relevant_long_term,
                )
            else:
                # 中等/复杂任务 或 无匹配子Agent：主Agent规划执行
                planning_context = context
                # 统一注入orchestrator结构化记忆（已选方案、实体、决策等）
                if orchestrator_ctx:
                    planning_context += f"\n[Orchestrator已知信息]\n{orchestrator_ctx}"
                if route_target:
                    if verbose:
                        route_label = "🛒 售前" if route_target == AgentRouter.ROUTE_PRESALE else "⚙️ 功能"
                        print(f"📊 {route_label}域任务 | 复杂度: {intent_result.complexity.value} → 主Agent规划执行\n")
                    self.logger.info(
                        f"中等/复杂任务由主Agent规划执行 | route={route_target} | "
                        f"complexity={intent_result.complexity.value}"
                    )

                response = self.task_planner.execute(
                    user_query=user_input,
                    intent=intent_result,
                    context=planning_context,
                    long_term_context=relevant_long_term,
                    verbose=verbose,
                    on_step_complete=self.agent_router.memory.extract_entities_from_step_result,
                )

                # 子Agent域内的任务完成后，同步更新orchestrator记忆
                if route_target and sub_agent_enabled:
                    self.agent_router.memory.update_from_sub_agent_result(
                        route=route_target, query=user_input, response=response,
                    )

            # ========== 3. 输出层：安全审查 ==========
            review_result = self.output_guard.review(response)
            if not review_result.is_safe:
                self.logger.warning(f"输出审查警告: {review_result.warnings}")
                response = review_result.cleaned_output or "抱歉，回复内容未通过安全审查。"
            elif review_result.cleaned_output != response:
                self.logger.info("输出已进行敏感信息脱敏")
                response = review_result.cleaned_output

            # 添加到短期记忆（智能摘要长响应，仅用于上下文窗口管理）
            with tracer.start_span("memory.update"):
                response_for_memory = self._prepare_response_for_memory(response, user_input, context)
                self.short_term_memory.add_message("assistant", response_for_memory)
                self.short_term_memory.increment_message_count()

                # 持久化 Agent 响应到会话存储（存原始完整回复，与用户看到的一致）
                if self.user_id:
                    self.session_manager.add_message(self.session_id, 'assistant', response)
                    # 首条用户消息时自动生成会话标题
                    if self.conversation_count == 0:
                        self.session_manager.auto_generate_title(self.session_id, user_input)

                # 更新对话计数
                self.conversation_count += 1

                # 检查是否需要生成长期记忆总结
                if self.conversation_count % self.summary_threshold == 0:
                    self._generate_long_term_summary()

            # ========== 4. Agent评估：记录任务完成 ==========
            self.agent_evaluator.task_completed(
                task_id=task_id,
                success=True,
                error="",
            )

            self.logger.info(f"Agent回复: {response[:100]}...")

            tracer.set_span_attributes({
                "agent.intent_type": intent_result.intent_type.value,
                "agent.complexity": intent_result.complexity.value,
                "agent.response_length": len(response),
                "agent.goal_confidence": goal_result.confidence,
            })
            tracer.set_span_ok()

            return response

    def _retrieve_relevant_long_term_context(self, query: str, top_k: int = 3) -> str:
        """检索相关的长期记忆上下文"""
        try:
            # 搜索当前会话的相关历史对话
            similar_convs = self.long_term_memory.search_similar_conversations(
                query=query,
                session_id=self.session_id,
                n_results=top_k
            )

            if not similar_convs:
                return ""

            # 格式化为上下文字符串
            context_parts = ["[相关历史对话记忆]"]
            for i, conv in enumerate(similar_convs, 1):
                context_parts.append(f"{i}. {conv['summary']}")

            return "\n".join(context_parts)
        except Exception as e:
            self.logger.error(f"检索长期记忆失败: {e}")
            return ""

    def _prepare_response_for_memory(self, response: str, user_input: str, context: str) -> str:
        """
        准备响应以存入短期记忆（长响应自动摘要）
        
        Args:
            response: 完整的助手响应
            user_input: 用户输入
            context: 对话上下文
            
        Returns:
            要存入短期记忆的响应（原文或摘要）
        """
        # 如果禁用响应摘要功能，直接返回原响应
        if not self.response_summary_enabled:
            return response

        # 如果响应长度在阈值内，直接返回
        if len(response) <= self.response_max_length:
            self.logger.debug(f"响应长度({len(response)})在阈值内，直接存储")
            return response

        # 响应过长，需要提取关键信息
        self.logger.info(f"响应过长({len(response)}字符)，准备提取关键信息后存储")

        try:
            summary = self._summarize_response(response, user_input, context)
            self.logger.info(f"响应摘要完成，原长度: {len(response)}, 摘要长度: {len(summary)}，原响应：{response}，摘要信息：{summary}")
            return summary
        except Exception as e:
            self.logger.error(f"响应摘要失败: {e}，使用截断方式")
            # 如果摘要失败，使用简单截断策略
            return response[:self.response_max_length] + "...(已截断)"

    def _summarize_response(self, response: str, user_input: str, context: str) -> str:
        """
        使用LLM提取响应中的关键信息
        
        Args:
            response: 完整响应
            user_input: 用户输入
            context: 对话上下文
            
        Returns:
            提取的关键信息摘要
        """
        prompt = f"""你是一个信息提取助手。用户刚刚提出了一个问题，AI助手给出了一个详细的回答。现在需要从这个回答中提取核心信息，用于后续对话的上下文记忆。

用户问题：
{user_input}

助手的完整回答：
{response}

请提取回答中的关键信息，要求：
1. **【最重要】必须原样保留所有ID标识符**（如用户ID: UID-XXXX、商品ID: PXXX、订单号: ORD-XXXX、地址ID: ADDR-XXX、卡号ID: CARD-XXX等），这些是后续操作的关键依赖
2. 保留直接回答用户问题的核心内容
3. 保留重要的事实、数据、结论
4. 保留用户可能在后续对话中引用的关键点（价格、名称、状态等）
5. 去除示例代码、详细步骤、空行等冗长内容（只保留要点）
6. 去除客套话和重复内容
7. 控制在{self.response_max_length}字以内
8. 保持信息的准确性和可理解性

提取的关键信息："""

        summary = self.llm_client.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=int(self.response_max_length * 1.5)  # 留一些余量
        ).strip()

        return summary

    def _compress_historical_summary(self):
        """
        压缩过长的历史摘要，保留核心信息
        """
        self.logger.info("触发历史摘要压缩")

        try:
            current_summary = self.short_term_memory.get_historical_summary()

            if not current_summary:
                return

            # 使用LLM压缩摘要
            prompt = f"""以下是一段历史对话摘要，内容较长。请将其压缩为更简洁的版本，保留所有关键信息和重要上下文：

{current_summary}

压缩要求：
1. **【最重要】必须原样保留所有ID标识符**（如UID-XXXX、PXXX、ORD-XXXX、ADDR-XXX、CARD-XXX等），绝不能丢弃或修改这些标识符
2. 保留所有重要的用户需求、决定和实体信息
3. 去除冗余和重复内容
4. 控制在200字以内
5. 保持信息的连贯性和可理解性

压缩后的摘要："""

            compressed = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=300
            ).strip()

            # 更新压缩后的摘要
            self.short_term_memory.compress_summary(compressed)

            self.logger.info(f"历史摘要压缩完成，原长度: {len(current_summary)}, 新长度: {len(compressed)}")

        except Exception as e:
            self.logger.error(f"压缩历史摘要失败: {e}")

    def _generate_short_term_summary(self, verbose: bool = False):
        """
        生成短期记忆的滚动总结
        
        当短期记忆队列满后，对最旧的消息进行总结，避免上下文丢失
        """
        self.logger.info("触发短期记忆滚动总结")

        if verbose:
            print("\n📝 正在生成对话历史摘要...\n")

        try:
            # 获取当前所有消息（即将被移除的旧消息也在其中）
            all_messages = self.short_term_memory.get_messages()

            # 检查消息数量是否满足最小要求（从配置读取）
            if len(all_messages) < self.rolling_summary_min_messages:
                self.logger.info(f"消息数量不足（需要至少{self.rolling_summary_min_messages}条），跳过总结")
                return

            # 根据配置的比例决定总结多少条消息
            num_to_summarize = max(1, int(len(all_messages) * self.rolling_summary_ratio))
            messages_to_summarize = all_messages[:num_to_summarize]

            self.logger.info(
                f"准备总结 {len(messages_to_summarize)} 条历史消息（总消息数: {len(all_messages)}, 总结比例: {self.rolling_summary_ratio}）")

            # 构建总结prompt
            conversation_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in messages_to_summarize
            ])

            prompt = f"""请简要总结以下对话片段的关键信息，保留重要的上下文和用户意图：

{conversation_text}

总结要求：
1. **【重要】必须原样保留所有ID标识符和关键数值**，例如：用户ID(UID-XXXX)、商品ID(PXXX)、订单号(ORD-XXXX)、地址ID(ADDR-XXX)、卡ID(CARD-XXX)、价格等。这些在后续对话中会被引用
2. 提取关键信息和讨论主题
3. 保留重要的用户需求、决定和操作结果
4. 简洁明了，控制在150字以内
5. 使用第三人称描述

总结："""

            summary = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=250
            ).strip()

            # 更新短期记忆的历史摘要
            self.short_term_memory.update_historical_summary(summary)

            self.logger.info(f"短期记忆摘要已生成: {summary[:50]}...")

            if verbose:
                print(f"✅ 历史摘要已更新\n")

        except Exception as e:
            self.logger.error(f"生成短期记忆摘要失败: {e}")

    def _generate_long_term_summary(self):
        """生成长期记忆总结"""
        self.logger.info(f"生成长期记忆总结（对话轮次: {self.conversation_count}）")

        try:
            # 获取最近的对话
            recent_messages = self.short_term_memory.get_messages()

            if len(recent_messages) < 2:
                return

            # 构建总结prompt
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages
            ])

            prompt = f"""请总结以下对话的主要内容：

{conversation_text}

请提供：
1. 对话的主要主题
2. 关键讨论点
3. **【必须保留】所有出现的ID标识符**（用户ID、商品ID、订单号、地址ID等），这些在后续对话中会被引用

总结："""

            summary = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            ).strip()

            # 提取主题
            topics = []  # 简化版，实际可以用NLP提取

            # 保存到长期记忆
            self.long_term_memory.add_conversation_summary(
                session_id=self.session_id,
                summary=summary,
                topics=topics
            )

            self.logger.info("长期记忆总结已保存")
        except Exception as e:
            self.logger.error(f"生成长期记忆总结失败: {e}")

    def add_documents_to_knowledge_base(
            self,
            documents: List[str],
            metadatas: List[Dict[str, Any]] = None
    ):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表
        """
        self.logger.info(f"添加 {len(documents)} 个文档到知识库")

        try:
            self.rag_engine.add_documents(documents, metadatas)
            self.logger.info("文档添加成功")

            # 显示知识库信息
            info = self.rag_engine.get_collection_info()
            self.logger.info(f"知识库文档总数: {info['document_count']}")
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        return self.rag_engine.get_collection_info()

    def list_mcp_services(self) -> List[Dict[str, Any]]:
        """列出可用的MCP服务"""
        return self.mcp_manager.get_enabled_servers()

    def call_mcp_tool(
            self,
            server_name: str,
            tool_name: str,
            parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        调用MCP工具
        
        Args:
            server_name: 服务名称
            tool_name: 工具名称
            parameters: 参数
            
        Returns:
            执行结果
        """
        self.logger.info(f"调用MCP工具: {server_name}.{tool_name}")
        return self.mcp_manager.call_tool(server_name, tool_name, parameters)

    def get_conversation_history(self, last_n: int = None) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.short_term_memory.get_messages(last_n)

    def get_long_term_summaries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取长期记忆总结"""
        return self.long_term_memory.get_session_summaries(self.session_id, limit)

    def clear_short_term_memory(self):
        """清空短期记忆"""
        self.short_term_memory.clear()
        self.logger.info("短期记忆已清空")

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        删除指定的多个会话

        如果当前会话在删除列表中，会自动重置到新会话。

        Args:
            session_ids: 要删除的会话ID列表

        Returns:
            实际删除的会话数
        """
        if not session_ids:
            return 0

        deleted = self.session_manager.delete_sessions(session_ids)

        # 如果当前会话被删除，重置到新会话
        if self.session_id in session_ids:
            self.logger.info("当前会话已被删除，自动重置")
            self.reset_session()

        self.logger.info(f"已删除 {deleted} 个会话")
        return deleted

    def delete_all_sessions(self) -> int:
        """
        删除当前用户的所有会话，并自动重置到新会话

        Returns:
            删除的会话数
        """
        if not self.user_id:
            self.logger.warning("未登录用户，无法删除会话")
            return 0

        deleted = self.session_manager.delete_all_sessions(self.user_id)

        # 删除全部后重置到新会话
        if deleted > 0:
            self.reset_session()
            self.logger.info(f"已删除全部 {deleted} 个会话，已重置到新会话")

        return deleted

    def reset_session(self):
        """重置会话（创建新会话）"""
        self.short_term_memory.clear()
        self.conversation_count = 0

        if self.user_id:
            session_info = self.session_manager.create_session(self.user_id)
            self.session_id = session_info.session_id
        else:
            self.session_id = str(uuid.uuid4())

        self.long_term_memory.create_session(self.session_id)
        self._is_resumed_session = False
        self.logger.info(f"会话已重置，新会话ID: {self.session_id}")

    def resume_session(self, session_id: str) -> bool:
        """
        恢复历史会话

        Args:
            session_id: 要恢复的会话ID

        Returns:
            是否成功恢复
        """
        session_info = self.session_manager.get_session(session_id)
        if not session_info:
            self.logger.warning(f"会话 {session_id} 不存在")
            return False

        self.session_id = session_id
        self._is_resumed_session = True
        self.short_term_memory.clear()
        self.long_term_memory.create_session(self.session_id)
        self._load_session_history()
        self.logger.info(f"已恢复会话: {session_id[:8]}... | 标题: {session_info.title}")
        return True

    def _register_local_tools(self):
        """注册本地工具到工具管理器"""
        from tool_manager.local_tools import LOCAL_TOOLS

        for tool_def in LOCAL_TOOLS:
            self.tool_manager.register_local_tool(
                name=tool_def["name"],
                description=tool_def["description"],
                handler=tool_def["handler"],
                input_schema=tool_def.get("input_schema"),
            )
        self.logger.info(f"注册了 {len(LOCAL_TOOLS)} 个本地工具")

    def _load_session_history(self):
        """从会话存储加载历史消息到短期记忆"""
        messages = self.session_manager.get_messages(self.session_id)
        if not messages:
            self.logger.info("历史会话无消息")
            return

        # 只加载最近 N 条消息到短期记忆（受 max_messages 限制）
        max_load = self.short_term_memory.max_messages
        recent = messages[-max_load:] if len(messages) > max_load else messages

        for msg in recent:
            self.short_term_memory.add_message(msg['role'], msg['content'])

        self.conversation_count = len(messages) // 2  # 估算对话轮数
        self.logger.info(
            f"从会话存储加载 {len(recent)}/{len(messages)} 条消息到短期记忆"
        )
