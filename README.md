# SmartAgent - 智能AI助手系统

一个面向企业场景的AI Agent系统，基于OpenAI兼容API实现。具备用户认证与会话管理、意图识别、任务规划与评估、双重记忆系统、上下文工程、高级RAG检索、MCP工具集成、安全防护体系、效果评估以及OpenTelemetry全链路追踪能力。

## 系统架构

```
┌───────────────────────────────────────────────────────────────────┐
│                     interactive_chat.py (CLI)                     │
│              登录 → 会话选择 → 多轮对话 → 命令系统                     │
├─────────┬─────────────────────────────────────────────┬───────────┤
│  Auth   │              SmartAgent 核心                │  Session   │
│  认证   │  意图识别 → 子Agent路由/规划 → 执行 → 响应  │  会话管理  │
├─────────┴─────────────────────────────────────────────┴───────────┤
│                                                                   │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐   │
│  │ Input   │ │ Context  │ │ Tool     │ │  Rate    │ │ Output  │   │
│  │ Gate    │ │ Engine   │ │ Manager  │ │ Limiter  │ │ Guard   │   │
│  │ 输入验证 │ │ 上下文    │ │ 工具管理   │ │ 频率限制  │ │ 输出审查 │   │
│  └─────────┘ └──────────┘ └──────────┘ └──────────┘ └─────────┘  │
│                                                                  │
│  ┌─────────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Memory          │ │ RAG      │ │ MCP      │ │ Sub-Agents   │  │
│  │ 短期+长期记忆     │ │ 混合检索  │ │ 工具协议   │ │ 售前+功能     │  │
│  └─────────────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
│  │ Evaluation   │ │ Observability│ │ Utils                    │  │
│  │ Agent+RAG评估 │ │  OTel追踪    │ │ LLM客户端/配置/日志/连接池   │  │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## 核心功能

### 1. 用户认证与会话管理

- **认证模块** (`auth/`)：用户名密码登录（SQLite 数据库校验）、本地 Token 缓存（JSON）用于免登录、Token 有效期校验（默认48小时）、自动复用未过期 Token 时从数据库获取最新用户数据
- **会话管理** (`session/`)：每用户独立会话列表、SQLite 持久化对话历史、支持创建/恢复/切换/删除（单个/批量/全部）会话、会话标题自动生成
- **连接池** (`utils/db_pool.py`)：线程安全的 SQLite 连接池，基于 Queue 实现连接复用，上下文管理器自动借还，WAL 模式提升并发性能

### 2. 输入安全防护

**输入层** (`input_gate/`)：
- Prompt 注入攻击检测（忽略指令、角色扮演、越狱尝试等）
- 输入长度与格式校验
- 敏感词过滤与脱敏
- 输入审计日志，可追踪风险请求

### 3. 意图识别与目标理解

基于LLM的意图识别器（`agent/intent_recognizer.py`）+ 目标理解层（`agent/goal_understanding.py`），自动判断用户意图类型、任务复杂度，并在信息不充分时主动澄清：

**意图类型：**
- `greeting` - 问候，直接回复
- `simple_chat` - 简单对话，基于已有知识回答
- `rag_simple` - 简单RAG检索（公司制度、人事、考勤等知识库问题）
- `rag_advanced` - 高级RAG检索（多文档综合、对比分析）
- `mcp_execute` - MCP工具执行（创建/修改/删除等操作）
- `mcp_ask_info` - 询问MCP工具参数信息

**任务复杂度：**
- `simple` - 单步任务，直接ReAct执行
- `medium` - 2-3步任务，Plan and Execute模式
- `complex` - 4步以上任务，精细化拆分 + ReAct执行

识别策略：LLM识别优先，失败时降级为启发式规则。

**目标理解与澄清机制**（`agent/goal_understanding.py`）：
- 结合长期记忆增强意图识别（低置信度时检索历史对话辅助推理）
- 置信度校准：用户输入缺少关键参数时自动降低置信度
- MCP 执行类意图额外进行参数完整性检查（基于工具 inputSchema）
- 信息不完整时生成针对性澄清问题（列出需补充的关键信息），避免盲目执行
- 提取用户目标、约束条件和成功标准

### 4. 上下文工程

企业级 LLM 上下文窗口管理（`context/context_manager.py`），三大核心能力：

- **分区预算管理**：将 LLM 窗口划分为 system_prompt、planning_steps、tools、short_term_memory、rag_results、long_term_memory、tool_results 等分区，各分区独立预算与优先级
- **工具懒加载**：选择阶段仅加载工具名称+描述，参数填充阶段按需加载完整 Input Schema，减少约 60-80% 的工具描述 token 占用
- **预算溢出分级处理**：裁剪(Trim) → 微压缩(Micro-compress) → 投影(Project) → 自动压缩(Auto-compress)，按优先级依次执行

### 5. 工具管理器

类似服务注册中心的设计（`tool_manager/tool_manager.py`），统一管理本地工具与 MCP 远程工具：

- **注册与生命周期**：本地 Python 工具通过 `register_local_tool()` 注册，MCP 工具通过 `sync_mcp_tools()` 自动导入；支持卸载、启用/禁用
- **自动分类与关键词提取**：注册时自动归类（order/user/product/address/card/query/manage/general）并提取中英文关键词，建立倒排索引
- **多层过滤** `get_tools_for_query(query)`：
  1. 关键词/分类粗过滤 — 基于用户查询匹配工具关键词与分类
  2. 频率过滤 — 集成 RateLimiter，排除已达限额的工具
  3. top_k 截断 — 返回精简候选列表
- **懒加载 Schema**：选择阶段仅暴露 name + description，参数填充阶段才返回完整 inputSchema
- **本地工具**（`tool_manager/local_tools.py`）：预置工具（如 `delete_user_orders` 需人工确认），可通过 `LOCAL_TOOLS` 列表扩展

### 6. 频率限制

工具调用频率限制器（`security/rate_limiter.py`），防止预算花超和服务过载：

- 滑动窗口算法（精确到秒级），三层限制：全局频率、单工具频率、会话总量
- 支持特定高危工具的自定义限额（如 `delete_order: 3次/分钟`）
- 工具执行前自动检查频率，超限时拒绝并返回原因

### 7. 任务规划与执行

根据意图识别的复杂度，采用不同的执行策略（`agent/task_planner.py`）：

- **简单任务**：根据意图类型直接执行（对话/RAG查询/ReAct工具调用）
- **中等/复杂任务**：Plan and Execute流程
  1. LLM生成执行计划
  2. 执行前评估（检查计划合理性）
  3. 逐步执行（每步支持知识检索、MCP工具调用、答案生成）
  4. 执行中评估（每步完成后检查结果）
  5. 生成最终答案
  6. 执行后评估（检查是否满足用户需求）

评估不通过时自动重新规划（最多可配置重试次数）。

### 8. 安全执行体系

三层安全防护：

- **权限控制** (`security/permission_manager.py`)：基于角色的访问控制（RBAC），admin/user/viewer 三级角色，工具级细粒度权限映射，操作频率限制
- **人类确认** (`security/human_confirmation.py`)：敏感操作分级确认（NONE → INFORM → CONFIRM → DOUBLE_CONFIRM），高危操作（删除、修改重要数据等）需要用户显式确认
- **安全执行器** (`execution/safe_executor.py`)：受控环境执行工具操作，超时控制 + 异常隔离，集成权限检查与人类确认，执行审计日志

### 9. 输出审查

**输出层** (`security/output_guard.py`)：
- 输出内容安全审查（过滤不当内容）
- 敏感数据自动脱敏（手机号、身份证、银行卡、API Key、IP地址等）
- 禁止输出系统内部信息（system prompt、内部密钥等）

### 10. 任务评估

**三阶段评估机制**（`agent/task_evaluator.py`），评估结果分三个级别：

- **MUST_FIX** - 阻塞性问题，必须修复，触发重新规划
- **ACCEPTABLE** - 有瑕疵但不影响主流程，可继续执行
- **REMINDER** - 轻微建议，附加到最终回答

### 11. 效果评估

- **Agent 评估** (`evaluation/agent_evaluator.py`)：任务完成率、鲁棒性、执行效率、用户满意度估计，多维度量化 Agent 性能
- **RAG 评估** (`evaluation/rag_evaluator.py`)：检索相关性、答案忠实度、答案相关性、上下文利用率四维评分

### 12. 双重记忆系统

- **短期记忆** (`memory/short_term_memory.py`)：保存最近N条对话消息，滚动总结机制，长响应自动提取关键信息，历史摘要过长时自动压缩
- **长期记忆** (`memory/long_term_memory.py`)：ChromaDB向量库存储对话总结，每N轮自动生成总结并持久化，支持语义搜索历史对话

### 13. 高级RAG能力

实现了多种RAG技术（`rag/`），可在配置文件中独立开关：

- **查询重写**（Query Rewriting）：根据对话上下文优化查询
- **多查询生成**（Multi-Query）：从不同角度生成多个查询
- **HyDE**（Hypothetical Document Embeddings）：生成假设性文档提升检索效果
- **混合检索**：向量检索 + BM25关键词检索加权合并
- **父子分块**：大块存储保留上下文，小块检索提升精度
- **重排序**（Reranker）：对检索结果进行二次排序
- **Self-Fix**：自动验证和修正答案质量

支持多种文档格式加载：PDF、Word、PPT、Excel、CSV、Markdown、HTML、XML、TXT。

### 14. MCP工具集成

基于Model Context Protocol（`mcp_manager_module/`），支持动态连接外部工具服务：

- 支持SSE和HTTP JSON-RPC两种通信协议
- 从MCP服务器动态获取工具列表和参数Schema
- 工具调用时自动提取参数（结合用户输入和前置步骤结果）
- 支持动态添加/删除/启用/禁用服务

### 15. 子Agent路由

主Agent作为 Orchestrator，根据用户意图和任务复杂度路由到子Agent（`agent/agent_router.py`）：

- **售前客服子Agent** (`agent/presale_agent.py`)：商品推荐、FAQ问答，基于商品知识库（`agent/product_knowledge.py`）提供专业导购
- **功能处理子Agent** (`agent/functional_agent.py`)：订单、地址、银行卡等功能操作，调用 MCP 工具完成业务流程
- **子Agent基类** (`agent/sub_agent_base.py`)：统一的子Agent接口和生命周期管理
- **子Agent上下文** (`agent/sub_agent_context.py`)：为子Agent构建精简的领域上下文
- **Orchestrator记忆** (`agent/orchestrator_memory.py`)：维护子Agent间结构化信息流转（推荐方案缓存、实体缓存、用户决策记录）
- **消息总线** (`agent/message_bus.py`)：子Agent间信息请求与应答机制

路由策略：
- 简单任务 + 匹配子Agent → 子Agent直接处理（单步执行）
- 中等/复杂任务 → 主Agent规划执行（保留任务规划、依赖管理能力）
- 无匹配子Agent → 主Agent原有逻辑处理

### 16. 可观测性（OpenTelemetry）

基于OpenTelemetry的全链路追踪（`observability/`）：

- 支持Console和OTLP两种Exporter（可同时启用）
- 追踪Agent对话、意图识别、任务规划/执行/评估、RAG检索、MCP调用、LLM调用全流程
- 记录GenAI语义约定属性（模型、token用量、延迟等）
- 可配置是否记录prompt/response内容（生产环境可关闭）
- 提供 `@trace_span()` 装饰器和 NoOpSpan 空操作支持

### 17. 统一LLM客户端

基于OpenAI API的统一客户端（`utils/llm_client.py`），兼容多种LLM服务：

- 支持Ollama、OpenAI、Azure OpenAI、vLLM等兼容服务
- 提供 `chat`、`generate`、`chat_with_context`、`batch_generate` 等接口
- 支持流式输出
- 支持全局系统提示词（自动注入所有请求）

## 系统要求

- Python 3.8+
- Ollama（本地运行）或其他OpenAI API兼容服务

### 推荐模型
- 主模型：`qwen2.5`、`qwen3-coder`、`llama3.2` 等
- 嵌入模型：本地 `bge-small-zh-v1.5` 或 Ollama `nomic-embed-text`

## 快速开始

### 1. 安装Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**启动Ollama服务:**
```bash
ollama serve
```

**下载所需模型:**
```bash
ollama pull qwen2.5:latest
```

### 2. 安装Python依赖

```bash
cd test_project
pip install -r requirements.txt
```

### 3. 配置系统

编辑 `config/settings.yaml` 根据需要调整配置（LLM、嵌入模型、记忆、安全、认证、会话、RAG技术开关、可观测性等）。

### 4. 运行交互式对话

```bash
python interactive_chat.py
```

启动后会进入登录流程 → 会话选择 → 多轮对话交互。

### 5. 添加文档到知识库

```bash
# 添加单个文件
python add_documents.py --file document.txt --topic "主题"

# 添加目录中的文档
python add_documents.py --directory ./docs --topic "技术文档"

# 指定文件类型
python add_documents.py --directory ./docs --extensions .txt .md .pdf

# 交互式添加
python add_documents.py --interactive
```

## 使用指南

### 交互式对话命令

```
/help          - 显示帮助信息
/quit, /exit   - 退出程序
/sessions      - 查看历史会话列表
/switch <N>    - 切换到第N个历史会话
/new           - 创建新会话
/delete <N..>  - 删除指定会话（如 /delete 1 3 5）
/delete_all    - 删除所有会话
/clear         - 清空短期记忆
/reset         - 重置会话
/history [n]   - 显示最近n轮对话（默认10）
/summary       - 显示长期记忆总结
/info          - 显示知识库信息
/mcp           - 显示MCP服务信息
/metrics       - 显示Agent评估指标
/tools         - 显示已注册工具列表
/logout        - 注销并退出
/verbose on    - 开启详细日志
/verbose off   - 关闭详细日志
```

### 编程接口

```python
from test_project.agent.agent import SmartAgent
from test_project.auth import AuthManager
from test_project.session import SessionManager

# 初始化（带用户和会话）
agent = SmartAgent(user_id="UID-8888", session_id="some-session-id")

# 对话（自动进行意图识别 → 任务规划 → 执行 → 评估）
response = agent.chat("你好")
response = agent.chat("公司的考勤制度是什么？")  # 自动走RAG检索

# 添加知识
agent.add_documents_to_knowledge_base(
    documents=["文档内容..."],
    metadatas=[{"source": "文档1", "topic": "主题A"}]
)

# 获取对话历史
history = agent.get_conversation_history(last_n=5)

# 获取长期记忆
summaries = agent.get_long_term_summaries()

# 知识库信息
info = agent.get_knowledge_base_info()

# MCP服务
services = agent.list_mcp_services()
result = agent.call_mcp_tool("order", "create_order", {"item": "test"})

# 会话管理
agent.reset_session()                           # 重置当前会话
agent.delete_sessions(["sid-1", "sid-2"])        # 删除指定会话
agent.delete_all_sessions()                      # 删除当前用户所有会话

# 本地工具注册
agent.tool_manager.register_local_tool(
    name="my_tool", description="自定义工具",
    handler=lambda params: {"result": "ok"},
    input_schema={"type": "object", "properties": {"key": {"type": "string"}}}
)
```

## 项目结构

```
test_project/
├── config/                          # 配置文件
│   ├── settings.yaml               # 主配置（LLM、RAG、记忆、安全、认证、会话、可观测性等）
│   └── mcp_servers.yaml            # MCP服务配置
├── agent/                           # Agent核心
│   ├── agent.py                    # SmartAgent主类（对话入口、记忆管理、会话集成）
│   ├── intent_recognizer.py        # 意图识别器（LLM + 启发式规则 + 置信度校准）
│   ├── goal_understanding.py      # 目标理解层（澄清机制、参数完整性检查、长期记忆增强）
│   ├── task_planner.py             # 任务规划器（ReAct / Plan-and-Execute）
│   ├── task_evaluator.py           # 任务评估器（执行前/中/后三阶段）
│   ├── agent_router.py            # 子Agent路由器（售前/功能路由决策）
│   ├── presale_agent.py           # 售前客服子Agent（商品推荐、FAQ）
│   ├── functional_agent.py        # 功能处理子Agent（订单、地址、银行卡等）
│   ├── sub_agent_base.py          # 子Agent基类（统一接口与生命周期）
│   ├── sub_agent_context.py       # 子Agent上下文管理
│   ├── orchestrator_memory.py     # Orchestrator结构化任务记忆
│   ├── message_bus.py             # 子Agent消息总线
│   └── product_knowledge.py       # 商品知识库构建
├── auth/                            # 认证模块
│   └── auth_manager.py            # 认证管理器（SQLite用户存储、登录校验、本地Token缓存）
├── session/                         # 会话管理
│   └── session_manager.py         # 会话管理器（CRUD、对话历史持久化，SQLite连接池）
├── input_gate/                      # 输入安全层
│   └── input_validator.py         # 输入验证器（注入检测、敏感词过滤、格式校验）
├── context/                         # 上下文工程
│   └── context_manager.py         # 上下文窗口管理器（分区预算、懒加载、溢出处理）
├── tool_manager/                    # 工具管理器
│   ├── tool_manager.py            # 工具注册/分类/多层过滤/懒加载（本地+MCP统一管理）
│   └── local_tools.py             # 本地工具集合（delete_user_orders等）
├── execution/                       # 安全执行层
│   └── safe_executor.py           # 安全执行器（超时控制、异常隔离、权限+确认集成）
├── security/                        # 安全防护
│   ├── output_guard.py            # 输出审查（内容审查、数据脱敏）
│   └── rate_limiter.py            # 频率限制器（滑动窗口、per-tool/全局限额）
├── memory/                          # 记忆系统
│   ├── short_term_memory.py       # 短期记忆（滚动总结、响应摘要）
│   └── long_term_memory.py        # 长期记忆（ChromaDB语义搜索）
├── rag/                             # RAG组件
│   ├── rag_engine.py              # RAG引擎（整合所有RAG组件）
│   ├── vector_store.py            # 向量数据库（ChromaDB）
│   ├── embeddings.py              # 嵌入模型（本地/Ollama/OpenAI）
│   ├── document_loader.py         # 多格式文档加载器
│   ├── document_processor.py      # 文档处理（父子分块）
│   ├── query_rewriter.py          # 查询重写 + 多查询生成
│   ├── hyde.py                    # HyDE假设性文档生成
│   ├── hybrid_retriever.py        # 混合检索（向量+BM25）
│   ├── reranker.py                # 重排序
│   └── self_fix.py                # 答案自我修正
├── evaluation/                      # 效果评估
│   ├── agent_evaluator.py         # Agent多维评估（完成率、鲁棒性、效率、满意度）
│   └── rag_evaluator.py           # RAG质量评估（相关性、忠实度、利用率）
├── mcp_manager_module/              # MCP集成
│   └── mcp_manager.py            # MCP管理器（SSE/HTTP协议、工具发现与调用）
├── observability/                   # 可观测性
│   └── tracer.py                 # OpenTelemetry追踪器
├── utils/                           # 公共工具
│   ├── llm_client.py             # 统一LLM客户端（OpenAI API兼容）
│   ├── config_loader.py          # 配置加载器
│   ├── logger.py                 # 日志工具（彩色终端输出）
│   └── db_pool.py                # SQLite连接池（线程安全、上下文管理器）
├── docs/                            # 知识库文档
├── data/                            # 数据（向量库、长期记忆、会话DB、认证Token）
├── logs/                            # 日志文件
├── interactive_chat.py              # 交互式对话入口（登录→会话选择→多轮对话）
├── add_documents.py                 # 文档添加工具
├── requirements.txt                 # 依赖列表
└── README.md                        # 本文件
```

## 工作流程

### 完整对话处理流程

```
用户启动 interactive_chat.py
  │
  ├─ 登录流程（检查本地Token → 过期则要求输入用户名密码）
  │
  ├─ 会话选择（列出历史会话 / 创建新会话）
  │
  └─ 进入对话循环
       │
       ├─ 输入验证（Input Gate: 注入检测、敏感词过滤、长度校验）
       │     ├─ 高风险 → 拦截并提示
       │     └─ 通过 → 继续
       │
       ├─ 添加到短期记忆（检查是否触发滚动总结/摘要压缩）
       │
       ├─ 构建上下文（Context Engine: 分区预算分配、溢出处理）
       │     ├─ 短期记忆
       │     ├─ 长期记忆语义搜索
       │     └─ 工具列表（懒加载，仅名称+描述）
       │
       ├─ 目标理解（意图识别 + 澄清机制）
       │     │
       │     ├─ 置信度校准（信息不完整时降低置信度）
       │     ├─ MCP意图参数完整性检查（缺少关键参数 → 生成澄清问题）
       │     ├─ 信息不足 → 返回澄清问题，等待用户补充
       │     │
       │     ├─ greeting → 直接回复
       │     ├─ simple + simple_chat → 直接对话
       │     ├─ simple + rag_simple → 基础RAG检索
       │     ├─ simple + rag_advanced → 高级RAG全流程
       │     │     （查询重写 + 多查询 + HyDE + 混合检索 + 重排序 + Self-Fix）
       │     ├─ simple + mcp_execute → ReAct循环（思考-行动-观察）
       │     │     │
       │     │     └─ ToolManager 多层过滤 → 频率检查 → 执行
       │     │           ├─ 关键词/分类过滤（缩小候选工具集）
       │     │           ├─ RateLimiter 频率检查（超限则拒绝）
       │     │           ├─ 本地工具 → 直接调用 handler
       │     │           └─ MCP工具 → 通过 MCPManager 调用
       │     ├─ mcp_ask_info → 返回工具参数说明
       │     └─ medium/complex → Plan and Execute
       │                           │
       │                           ├─ LLM生成执行计划
       │                           ├─ 执行前评估
       │                           ├─ 逐步执行（知识检索/MCP调用/答案生成）
       │                           ├─ 执行中评估（每步后）
       │                           ├─ 生成最终答案
       │                           └─ 执行后评估（不通过则重新规划）
       │
       ├─ 输出审查（Output Guard: 内容审查、敏感数据脱敏）
       │
       ├─ 响应摘要（长响应自动提取关键信息）
       ├─ 添加到短期记忆
       ├─ 持久化到会话历史（SQLite）
       └─ 定期生成长期记忆总结
```

## 高级配置

### 主要配置项（config/settings.yaml）

| 配置项 | 说明 |
|--------|------|
| `llm` | LLM服务配置（provider、api_key、base_url、model、temperature等） |
| `embedding` | 嵌入模型配置（支持local/ollama/openai） |
| `memory.short_term` | 短期记忆（max_messages、滚动总结、响应摘要） |
| `memory.long_term` | 长期记忆（summary_threshold、persist_directory） |
| `intent.complexity` | 意图复杂度阈值 |
| `evaluator` | 任务评估开关（执行前/中/后独立控制） |
| `planner` | 任务规划（max_replan_attempts） |
| `react` | ReAct配置（max_iterations、temperature） |
| `rag.*` | RAG各技术开关和参数 |
| `context_window` | 上下文窗口预算（max_chars、分区预算比例、溢出策略） |
| `input_gate` | 输入验证（最大长度、注入检测模式、敏感词列表） |
| `security.rate_limit` | 频率限制（全局/单工具/会话限额、特定工具覆盖限额） |
| `security.output_guard` | 输出审查（脱敏规则、禁止输出模式） |
| `auth` | 认证配置（Token有效期、存储目录） |
| `session` | 会话管理（数据库目录、每用户最大会话数、连接池大小） |
| `mcp` | MCP服务配置文件路径和超时 |
| `system_prompt` | 全局系统提示词 |
| `observability` | OpenTelemetry配置（exporter、endpoint等） |
| `logging` | 日志配置 |

### MCP服务配置（config/mcp_servers.yaml）

```yaml
servers:
  - name: "order"
    enabled: true
    endpoint: "http://localhost:8000/sse"
    description: "订单和用户管理服务"
```

工具列表从MCP服务器动态获取，无需手动配置。

## 技术栈

- **LLM接口**：OpenAI API（兼容Ollama/OpenAI/vLLM等）
- **向量数据库**：ChromaDB
- **嵌入模型**：Sentence Transformers（本地）/ Ollama / OpenAI
- **检索算法**：向量相似度 + BM25（rank-bm25）
- **文档加载**：LangChain Document Loaders（PDF、Word、PPT、Excel、CSV、Markdown、HTML、XML）
- **数据持久化**：SQLite（用户数据 + 会话/对话历史）+ JSON（本地Token缓存）+ ChromaDB（记忆/知识库）
- **连接池**：自研线程安全 SQLite ConnectionPool（Queue + WAL模式）
- **MCP通信**：SSE（mcp SDK）+ HTTP JSON-RPC
- **可观测性**：OpenTelemetry（Console / OTLP Exporter）
- **日志**：Python logging + colorlog

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 连接LLM失败 | 确保Ollama正在运行（`ollama serve`），检查 `settings.yaml` 中的 `base_url` 和端口 |
| 模型未找到 | 下载模型（`ollama pull <model>`），检查 `settings.yaml` 中的 `model` 名称 |
| 嵌入模型加载失败 | 检查本地模型路径是否正确，或切换为ollama嵌入模型 |
| 知识库为空 | 使用 `add_documents.py` 添加文档 |
| MCP工具调用失败 | 检查MCP服务是否启动，确认 `mcp_servers.yaml` 中的 endpoint |
| 检索结果不准确 | 调整混合检索权重、分块大小，启用更多RAG技术 |
| 登录失败 | 检查数据库 `data/db/app.db` 中 users 表是否正常；首次启动会自动创建默认用户（admin/admin123、test/test123） |
| 会话数据丢失 | 检查 `data/sessions/sessions.db` 文件是否存在，确认 `settings.yaml` 中 `session.data_dir` 配置正确 |
| 工具调用被限流 | 检查 `security.rate_limit` 配置，调整 `global_max_per_minute` 或在 `tool_overrides` 中为特定工具放宽限额 |

## License

MIT License
