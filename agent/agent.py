import threading
import uuid
from typing import Dict, Any, List  # noqa: F401 - used in type hints

from context import ContextWindowManager
from evaluation import AgentEvaluator, RAGEvaluator, EvalStore
from input_gate import InputValidator
from mcp_manager_module import MCPManager
from memory import ShortTermMemory, LongTermMemory
from observability import AgentTracer, get_tracer
from rag.rag_engine import RAGEngine
from security import OutputGuard, RateLimiter
from session import SessionManager
from tool_manager import ToolManager
from utils import setup_logger, ConfigLoader, LLMClient
from utils.logger import set_trace_id
from .goal_understanding import GoalUnderstanding
from .intent_recognizer import IntentRecognizer
from .orchestrator import Orchestrator
from .product_knowledge import build_product_knowledge_base
from .task_evaluator import TaskEvaluator
from .task_planner import TaskPlanner


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

        # 记录初始化失败的组件（容错模式）
        self._degraded_components: list = []

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
        llm_routing = self.config.get('llm', {}).get('routing', {})
        # 过滤掉空值（未配置的 task_type）
        llm_routing = {k: v for k, v in llm_routing.items() if v}
        self.llm_client = LLMClient(
            api_key=self.config.get('llm', {}).get('api_key', 'EMPTY'),
            base_url=self.config.get('llm', {}).get('base_url', 'http://localhost:11434/v1'),
            model=self.config.get('llm', {}).get('model', 'qwen2.5:latest'),
            temperature=self.config.get('llm', {}).get('temperature', 0.7),
            max_tokens=self.config.get('llm', {}).get('max_tokens', 2000),
            system_prompt=system_prompt_content,
            routing=llm_routing,
            logger=self.logger
        )

        if system_prompt_enabled and system_prompt_content:
            self.logger.info("全局系统提示词已设置到LLM客户端")

        # 初始化RAG引擎（先初始化以获取嵌入模型）
        self.logger.info("初始化RAG引擎...")
        self.rag_engine = None
        try:
            embedding_config = self.config.get('embedding', {}).copy()
            if embedding_config.get('provider') == 'local' and embedding_config.get('model'):
                embedding_config['model'] = self.config_loader.resolve_path(embedding_config['model'])
                self.logger.info(f"解析嵌入模型路径: {embedding_config['model']}")
            config_with_resolved_paths = self.config.copy()
            config_with_resolved_paths['embedding'] = embedding_config
            self.rag_engine = RAGEngine(config_with_resolved_paths, llm_client=self.llm_client, logger=self.logger)
        except Exception as e:
            self._degraded_components.append('rag_engine')
            self.logger.warning(f"RAG引擎初始化失败（降级运行，知识库功能不可用）: {e}")

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
        self.long_term_memory = None
        try:
            if self.rag_engine:
                self.long_term_memory = LongTermMemory(
                    persist_directory=self.config.get('memory', {}).get('long_term', {}).get('persist_directory',
                                                                                             './data/long_term_memory'),
                    embedding_function=self.rag_engine.embeddings,
                    logger=self.logger
                )
                self.long_term_memory.create_session(self.session_id)
            else:
                self._degraded_components.append('long_term_memory')
                self.logger.warning("RAG引擎不可用，跳过长期记忆初始化")
        except Exception as e:
            self._degraded_components.append('long_term_memory')
            self.logger.warning(f"长期记忆初始化失败（降级运行）: {e}")

        # 如果是恢复的历史会话，加载历史消息到短期记忆
        if self._is_resumed_session:
            self._load_session_history()
        # 如果是新会话且有用户
        elif self.user_id:
            # 优先复用已有的空会话（message_count=0），避免每次登录都创建新会话
            existing = self.session_manager.list_sessions(self.user_id, limit=3)
            empty_session = next(
                (s for s in existing if s.message_count == 0), None
            )
            if empty_session:
                self.session_id = empty_session.session_id
                self.logger.info(f"复用已有空会话: {self.session_id[:8]}...")
            else:
                self.session_manager.create_session(
                    user_id=self.user_id, title='新会话'
                )
                sessions = self.session_manager.list_sessions(self.user_id, limit=1)
                if sessions:
                    self.session_id = sessions[0].session_id
                    self.logger.info(f"新会话已持久化: {self.session_id[:8]}...")

        # 初始化MCP管理器
        mcp_config_file = self.config.get('mcp', {}).get('config_file', './config/mcp_servers.yaml')
        self.logger.info(f"初始化MCP管理器: {mcp_config_file}")
        self.mcp_manager = None
        try:
            self.mcp_manager = MCPManager(mcp_config_file, logger=self.logger)
        except Exception as e:
            self._degraded_components.append('mcp_manager')
            self.logger.warning(f"MCP管理器初始化失败（降级运行，工具调用不可用）: {e}")

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

        # 编排器：5阶段生命周期，动态创建子Agent，失败处理
        self.logger.info("初始化Orchestrator编排器...")
        self.orchestrator = Orchestrator(
            llm_client=self.llm_client,
            mcp_manager=self.mcp_manager,
            rag_engine=self.rag_engine,
            task_planner=self.task_planner,
            tool_manager=self.tool_manager,
            context_manager=self.context_manager,
            config=self.config,
            logger=self.logger,
        )
        # 注入 factory：TaskPlanner 的分层规划并行执行依赖 SubAgentFactory，
        # 但两者存在循环依赖（factory→task_planner→factory），所以在这里做延迟注入。
        if hasattr(self.orchestrator, 'factory'):
            self.task_planner.factory = self.orchestrator.factory

        # 商品知识库（RAG依赖）
        sub_agent_config = self.config.get('sub_agents', {})
        if sub_agent_config.get('init_product_kb', True) and self.rag_engine:
            try:
                self.logger.info("初始化商品FAQ知识库...")
                faq_count = build_product_knowledge_base(self.rag_engine, self.logger)
                self.logger.info(f"商品FAQ知识库初始化完成，共 {faq_count} 条")
            except Exception as e:
                self.logger.warning(f"商品FAQ知识库初始化失败（不影响主功能）: {e}")

        # Agent评估器
        self.logger.info("初始化评估层...")
        self.agent_evaluator = AgentEvaluator(self.config, logger=self.logger)

        # RAG 质量评估器
        self.rag_evaluator = RAGEvaluator(self.config, llm_client=self.llm_client, logger=self.logger)

        # 评估结果持久化存储
        eval_db_path = self.config.get('evaluation', {}).get('db_path', './data/eval_store.db')
        self.eval_store = EvalStore(db_path=eval_db_path, logger=self.logger)

        # 对话计数器（用于长期记忆总结）—— 保留供外部读取和 reset_session 使用
        self.conversation_count = 0
        self.summary_threshold = self.config.get('memory', {}).get('long_term', {}).get('summary_threshold', 5)

        # D1 fix: 响应摘要配置由 PostResponsePipeline 从 config 直接读取，此处无需重复存储

        # 上下文装配流水线
        from .context_pipeline import ContextPipeline
        self.context_pipeline = ContextPipeline(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            orchestrator=self.orchestrator,
            config=self.config,
            user_id=self.user_id,
            username=self.username,
            session_id=self.session_id,
            degraded_components=self._degraded_components,
            logger=self.logger,
        )

        # 响应后处理流水线
        from .post_response_pipeline import PostResponsePipeline
        self.post_pipeline = PostResponsePipeline(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            session_manager=self.session_manager,
            llm_client=self.llm_client,
            agent_evaluator=self.agent_evaluator,
            config=self.config,
            degraded_components=self._degraded_components,
            logger=self.logger,
        )

        if self._degraded_components:
            self.logger.warning(
                f"SmartAgent 初始化完成（降级模式）| 不可用组件: {self._degraded_components}"
            )
        else:
            self.logger.info("SmartAgent 初始化完成！（全功能模式）")
        self.logger.info("=" * 60)

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        与用户进行对话

        完整流程：
        输入验证 → 记忆上下文 → 目标理解（含澄清） → 任务规划与执行 → 输出审查 → 后处理流水线 → Agent评估

        Args:
            user_input: 用户输入
            verbose: 是否显示详细处理过程

        Returns:
            Agent的回复
        """

        tracer = get_tracer()
        task_id = str(uuid.uuid4())[:8]
        set_trace_id(task_id)  # 绑定请求级 trace ID，同一请求内所有日志携带该 ID

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

            # 检查是否需要触发滚动总结（在 ContextPipeline 之前执行）
            if self.short_term_memory.should_trigger_summary():
                self._generate_short_term_summary(verbose)

            # 检查是否需要压缩历史摘要（在 ContextPipeline 之前执行）
            if self.short_term_memory.needs_compression():
                self._compress_historical_summary()

            # ========== 1. 上下文装配（仅一次） ==========
            with tracer.start_span("memory.retrieve_context"):
                bundle = self.context_pipeline.build(user_input)
                tracer.set_span_attributes({
                    "memory.context_length": len(bundle.context),
                    "memory.has_long_term": bool(bundle.long_term),
                })

            # ========== 2. 目标理解层（增强意图识别 + 澄清机制） ==========
            # 获取orchestrator结构化记忆，供参数完整性检查使用
            orchestrator_ctx = ""
            sub_agent_enabled = self.config.get('sub_agents', {}).get('enabled', True)
            if sub_agent_enabled:
                orchestrator_ctx = self.orchestrator.memory.get_context_for_sub_agent()
            self.logger.info(
                f"[主Agent] 传入目标理解层 | user_input: {user_input} | "
                f"context_len: {len(bundle.context)} | orch_ctx_len: {len(orchestrator_ctx)}"
            )
            self.logger.debug(
                f"[主Agent] context: {bundle.context if bundle.context else '(空)'}"
            )
            if orchestrator_ctx:
                self.logger.debug(f"[主Agent] orchestrator_ctx: {orchestrator_ctx}")
            goal_result = self.goal_understanding.understand(
                user_input,
                bundle.context,
                orchestrator_context=orchestrator_ctx,
                long_term_context=bundle.long_term,
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
                # B4 fix: 澄清回复也需要持久化到会话存储，否则会话恢复后上下文断裂
                if self.user_id:
                    self.session_manager.add_message(self.session_id, 'assistant', clarification)
                if verbose:
                    print(f"❓ 需要澄清\n")
                return clarification

            # 取 token 快照（在任务开始前）
            usage_snapshot = self.llm_client.get_usage_snapshot()

            # 记录任务开始（Agent评估）
            self.agent_evaluator.task_started(
                task_id=task_id,
                query=user_input,
                intent_type=intent_result.intent_type.value,
                complexity=intent_result.complexity.value,
            )

            # ========== 3. Orchestrator统一执行 ==========
            # 主Agent的三重身份：
            # a) 执行者 —— 简单任务自己通过TaskPlanner完成
            # b) 指挥官 —— 中等/复杂任务分解、动态创建子Agent委派、汇总
            # c) 对话者 —— 唯一与用户直接交互
            #
            # Orchestrator根据复杂度自动决策：
            # - SIMPLE → 主Agent直接执行（不创建子Agent）
            # - MEDIUM/COMPLEX → 5阶段生命周期（理解→规划→执行→整合→交付）
            #   其中执行阶段会动态创建子Agent处理可并行/上下文溢出/离题的子任务
            response = self.orchestrator.handle_request(
                user_query=user_input,
                intent_result=intent_result,
                context=bundle.context,
                long_term_context=bundle.long_term,
                user_id=bundle.user_id,
                username=bundle.username,
                verbose=verbose,
            )

            # 执行中途需要用户补充信息（与前置澄清走相同路径）
            if isinstance(response, str) and response.startswith("__NEED_INPUT__:"):
                question = response[len("__NEED_INPUT__:"):]
                self.short_term_memory.add_message("assistant", question)
                self.short_term_memory.increment_message_count()
                if self.user_id:
                    self.session_manager.add_message(self.session_id, 'assistant', question)
                # B1 fix: task_started 已在前面调用，此处提前返回必须配套 task_completed，
                # 否则 AgentEvaluator._active_tasks 会持续累积，造成内存泄漏
                self.agent_evaluator.task_completed(
                    task_id=task_id, success=False, error="执行中需要用户输入"
                )
                # M2 fix: 记录 need_input 触发事件，便于监控"执行中途需要补充信息"的频率
                self.logger.info(
                    f"[监控] need_input 触发 | task_id={task_id} | "
                    f"intent={intent_result.intent_type.value} | "
                    f"question_len={len(question)}"
                )
                tracer.set_span_attributes({
                    "agent.need_input": True,
                    "agent.intent_type": intent_result.intent_type.value,
                })
                if verbose:
                    print(f"\u23f8 执行中需要用户补充信息\n")
                return question

            # 持久化任务状态（支持会话恢复时重建）
            task_state = self.orchestrator.get_current_task_state()
            if task_state and self.user_id:
                try:
                    self.session_manager.save_task_state(
                        self.session_id, task_state.to_json()
                    )
                except Exception as e:
                    self.logger.debug(f"任务状态持久化失败（不影响主流程）: {e}")

            # ========== 4. 输出层：安全审查 ==========
            review_result = self.output_guard.review(response)
            if not review_result.is_safe:
                self.logger.warning(f"输出审查警告: {review_result.warnings}")
                response = review_result.cleaned_output or "抱歉，回复内容未通过安全审查。"
            elif review_result.cleaned_output != response:
                self.logger.info("输出已进行敏感信息脱敏")
                response = review_result.cleaned_output

            # ========== 5. 后处理流水线 ==========
            with tracer.start_span("memory.update"):
                is_failed = getattr(self.orchestrator, '_last_request_failed', False)
                first_message = (self.post_pipeline.conversation_count == 0)
                token_usage = self.llm_client.compute_usage_delta(usage_snapshot)
                self.post_pipeline.run(
                    task_id=task_id,
                    user_input=user_input,
                    response=response,
                    bundle=bundle,
                    is_failed=is_failed,
                    first_message=first_message,
                    token_usage=token_usage,
                )
                # 同步 agent 级别的对话计数器（供外部读取和 reset_session 使用）
                self.conversation_count = self.post_pipeline.conversation_count

            response = self._strip_json_wrapper(response)
            self.logger.info(f"Agent回复: {response[:100]}{'...' if len(response) > 100 else ''}")

            # M1 fix: 记录 token 消耗，便于成本监控
            attrs = {
                "agent.intent_type": intent_result.intent_type.value,
                "agent.complexity": intent_result.complexity.value,
                "agent.response_length": len(response),
                "agent.goal_confidence": goal_result.confidence,
            }
            if token_usage:
                attrs["agent.input_tokens"] = token_usage.get("input_tokens", 0)
                attrs["agent.output_tokens"] = token_usage.get("output_tokens", 0)
                attrs["agent.llm_calls"] = token_usage.get("llm_calls", 0)
            tracer.set_span_attributes(attrs)
            tracer.set_span_ok()

            return response

    @staticmethod
    def _strip_json_wrapper(text: str) -> str:
        """剥离 LLM 误输出的 JSON 包裹壳，提取真实的用户可见内容。

        处理两种常见问题格式：
        1. Markdown 代码块包裹：```json\n{"reply": "实际内容"}\n```
        2. 顶层 JSON 对象含 reply/answer/content 键：{"reply": "实际内容"}
        """
        import json, re
        s = text.strip()

        # 1. 剥掉 ```json ... ``` 代码块
        code_block = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', s, re.DOTALL)
        if code_block:
            s = code_block.group(1).strip()

        # 2. 如果剩余内容是 JSON 对象，尝试提取回复键
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                for key in ("reply", "answer", "content", "response", "message"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            except (json.JSONDecodeError, Exception):
                pass

        return text  # 无需处理时原样返回

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

    def get_health_status(self) -> dict:
        """获取 Agent 各组件的健康状态

        Returns:
            {'status': 'ok'|'degraded', 'degraded_components': [...], 'available_features': {...}}
        """
        available = {
            'chat': True,           # LLMClient 始终可用（初始化失败会抛出异常）
            'rag': self.rag_engine is not None,
            'mcp_tools': self.mcp_manager is not None,
            'long_term_memory': self.long_term_memory is not None,
            'session_persistence': self.user_id is not None,
        }
        return {
            'status': 'degraded' if self._degraded_components else 'ok',
            'degraded_components': list(self._degraded_components),
            'available_features': available,
        }

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
        """获取长期记忆总结（用户维度，跨会话）"""
        if not self.long_term_memory:
            return []
        return self.long_term_memory.get_user_summaries(self.user_id or "", limit)

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

    def clear_long_term_memory(self) -> str:
        """清空当前用户的长期记忆（对话摘要 + 用户偏好）。

        由 /clear_memory 命令触发，不影响短期记忆和当前会话消息。

        Returns:
            面向用户的操作结果说明
        """
        if not self.long_term_memory:
            return "长期记忆不可用（RAG 引擎未初始化）。"

        cleared = []

        # 清除当前用户的全部对话摘要（跨会话）
        conv_count = self.long_term_memory.clear_conversation_summaries(self.user_id or "")
        if conv_count > 0:
            cleared.append(f"对话摘要 {conv_count} 条")

        # 清除当前用户的全部偏好
        if self.user_id:
            pref_count = self.long_term_memory.clear_all_user_preferences(self.user_id)
            if pref_count > 0:
                cleared.append(f"用户偏好 {pref_count} 条")

        if cleared:
            msg = f"已清空长期记忆：{', '.join(cleared)}。"
        else:
            msg = "长期记忆本就是空的，无需清理。"

        self.logger.info(f"[Command /clear_memory] {msg}")
        return msg

    def reset_session(self):
        """重置会话（创建新会话）"""
        self.short_term_memory.clear()
        self.conversation_count = 0

        if self.user_id:
            session_info = self.session_manager.create_session(self.user_id)
            self.session_id = session_info.session_id
        else:
            self.session_id = str(uuid.uuid4())

        # 同步 context_pipeline.session_id，避免 bundle.session_id 过期
        # 导致助手回复被写入旧 session（用户消息写新 session，数据割裂）
        self.context_pipeline.session_id = self.session_id
        # 重置 post_pipeline 计数器，确保新会话第一条消息能触发自动标题生成
        self.post_pipeline.conversation_count = 0

        if self.long_term_memory:
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
        # 同步 context_pipeline.session_id，确保 bundle.session_id 与当前 session 一致
        self.context_pipeline.session_id = self.session_id
        self._is_resumed_session = True
        self.short_term_memory.clear()
        if self.long_term_memory:
            self.long_term_memory.create_session(self.session_id)
        self._load_session_history()
        # 恢复的会话已有历史消息，post_pipeline 计数器需设为正数，
        # 避免 first_message=True 触发 auto_generate_title 覆盖原有标题
        self.post_pipeline.conversation_count = max(
            self.post_pipeline.conversation_count,
            session_info.message_count // 2 if session_info.message_count else 1,
        )
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
        """从会话存储加载历史消息到短期记忆，并恢复上一次的任务状态"""
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

        # 通过专用方法恢复上一次的任务状态到Orchestrator
        if hasattr(self, 'orchestrator'):
            try:
                import json
                task_state_json = self.session_manager.get_last_task_state(self.session_id)
                if task_state_json:
                    state_data = json.loads(task_state_json)
                    self.orchestrator.restore_task_state(state_data)
                    self.logger.info(
                        f"已恢复上一次任务状态: phase={state_data.get('phase', '?')}"
                    )
            except Exception as e:
                self.logger.warning(f"恢复任务状态失败（不影响主功能）: {e}")

        self.logger.info(
            f"从会话存储加载 {len(recent)}/{len(messages)} 条消息到短期记忆"
        )
