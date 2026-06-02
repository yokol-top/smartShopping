# SmartShopping — 智能购物助手

## 这是什么

SmartShopping 是一个 **AI 驱动的智能购物平台**。用户通过自然语言对话，即可完成从"我想买个手机"到"订单已创建"的完整购物流程——无需翻页、搜索、加购物车。

它不是一个简单的聊天机器人，而是一个具备**理解、规划、执行、整合**能力的多 Agent 系统。

### 能做什么

```
用户: 预算2万，帮我配一套大学生数码装备
助手: 根据您的预算，推荐以下方案：
      💻 [P006] 华为MateBook X Pro ¥11,999
      📱 [P003] 小米14 Ultra ¥5,999
      🎧 [P008] AirPods Pro 2 ¥1,899
      合计 ¥19,897，在预算内。需要下单吗？

用户: 就这个方案，帮我下单
助手: 好的，正在为您创建订单...（确认收货地址和支付方式后下单）
```

### 核心能力

| 能力 | 说明 |
|------|------|
| 商品推荐 | 根据需求、预算、使用场景推荐，支持跨类目组合方案 |
| 商品咨询 | 回答参数、评测、对比等售前问题（基于 RAG 知识库） |
| 智能下单 | 推荐 → 选择 → 确认 → 创建订单的完整购物流程 |
| 订单管理 | 查询订单状态、历史订单 |
| 用户管理 | 个人信息、收货地址、银行卡的查询与修改 |

---

## 业务设计

### 商品目录

平台经营 **7 个类目、15 个 SKU**，覆盖数码、家电、运动三大板块：

| 板块 | 类目 | 商品 | 价格区间 |
|------|------|------|----------|
| 数码 | 手机 | iPhone 15 Pro Max、华为Mate 60 Pro、小米14 Ultra | ¥5,999 - ¥9,999 |
| 数码 | 笔记本电脑 | MacBook Pro 16寸、ThinkPad X1 Carbon、华为MateBook X Pro | ¥11,999 - ¥19,999 |
| 数码 | 耳机 | 索尼WH-1000XM5、AirPods Pro 2 | ¥1,899 - ¥2,499 |
| 数码 | 平板电脑 | iPad Air 5、华为MatePad Pro 13.2 | ¥4,799 - ¥5,699 |
| 数码 | 智能手表 | Apple Watch Ultra 2 | ¥6,499 |
| 家电 | 家电 | 戴森V15吸尘器、海尔冰箱BCD-500 | ¥3,299 - ¥4,990 |
| 运动 | 运动鞋 | Nike Air Max 270、Adidas Ultraboost | ¥899 - ¥1,299 |

### 用户购物流程

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  表达需求 │────▶│ 接收推荐 │────▶│ 选择商品 │────▶│ 确认下单 │────▶│ 订单完成 │
│          │     │          │     │          │     │          │     │          │
│"想买手机" │     │ 3个方案  │     │"选方案一"│     │ 确认地址 │     │ 订单号   │
│"预算5000"│     │ 带理由   │     │          │     │ 确认支付 │     │ 金额汇总 │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
       ▲                                │
       │         需求模糊时主动澄清       │
       └─────────────────────────────────┘
```

### 知识库内容

系统内置 **35 条**商品知识文档（`init/product_descriptions.json`），包括：

- **详细评测**：每款商品的核心参数、技术解析
- **使用场景**：适合什么人群、什么用途，以及不推荐的场景
- **常见 FAQ**：用户高频问题的标准回答
- **选购指南**：学生装备方案、新居家电方案、运动装备方案等跨类目组合

---

## 系统设计

### 系统分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         用户接入层                                 │
│      CLI (interactive_chat.py)    Web (FastAPI + 静态前端)         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                       安全与会话层                                  │
│  ┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Auth 认证  │  │ InputValidator│  │RateLimiter│  │SessionManager│ │
│  │ 登录/Token │  │ 注入/敏感词  │  │ 频率限制  │  │ SQLite 持久  │  │
│  └───────────┘  └──────────────┘  └──────────┘  └─────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      SmartAgent 核心层                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ 记忆系统                                                   │    │
│  │  ShortTermMemory     LongTermMemory     OrchestratorMemory│    │
│  │  滚动摘要窗口         语义历史检索         方案/实体/决策    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────┐    │
│  │ 目标理解层                                                 │    │
│  │   IntentRecognizer (LLM) ── GoalUnderstanding            │    │
│  │   意图/复杂度/工具/目标      参数完整性检查 + 澄清机制      │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────┐    │
│  │ Orchestrator 编排层                                        │    │
│  │                                                           │    │
│  │  SIMPLE ──▶ TaskPlanner                                  │    │
│  │               ├─ direct LLM     ├─ RAGEngine             │    │
│  │               ├─ UnifiedReActExecutor                    │    │
│  │               │    └─ ToolCaller (MCP/本地工具执行)       │    │
│  │               └─ Plan-and-Execute + TaskEvaluator replan │    │
│  │                                                           │    │
│  │  MEDIUM/COMPLEX ──▶ 5阶段生命周期                         │    │
│  │               理解→规划→执行→整合→交付                     │    │
│  │                        │                                 │    │
│  │               SubAgentFactory                            │    │
│  │                ├─ CircuitBreaker (按工具类别分组)          │    │
│  │                └─ DynamicSubAgent × N (并行)             │    │
│  │                      └─ UnifiedReActExecutor             │    │
│  │                           └─ ToolCaller                  │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────┐    │
│  │ 输出安全层                                                 │    │
│  │   OutputGuard  敏感信息脱敏 + 禁止内容拦截                  │    │
│  └──────────────────────────────────────────────────────────┘    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌─────────────┬──────────────▼──────────────┬────────────────────┐
│  知识检索层  │       工具调用层              │    可观测性         │
│             │                             │                    │
│  RAGEngine  │  ToolManager               │  OpenTelemetry     │
│  混合检索   │  └─ MCPManager              │  AgentEvaluator    │
│  HyDE/重排  │       └─ MCP Server         │  RAGEvaluator      │
│  查询重写   │          购物平台后端         │                    │
│             │          (商品/订单/用户)    │                    │
└─────────────┴──────────────┬──────────────┴────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                          数据存储层                                 │
│                                                                   │
│   SQLite                                  ChromaDB               │
│   ├─ users / products / orders            ├─ 商品知识向量库          │
│   └─ sessions / messages / task_states   │  （评测/FAQ/选购指南）    │
│                                           └─ 长期记忆               │
│                                              （历史摘要+用户偏好）    │
└──────────────────────────────────────────────────────────────────┘
```

### 执行流程

```
用户输入
  │
  ▼
SmartAgent.chat()                ← 流量入口 + 记忆调度 + 安全过滤（容错降级）
  │
  ├─ InputValidator              ← 注入攻击检测、敏感词过滤
  ├─ ShortTermMemory             ← 滚动上下文窗口（含 LLM 滚动总结）
  ├─ LongTermMemory (ChromaDB)   ← 语义检索历史对话摘要 + 注入用户偏好
  │
  ▼
GoalUnderstanding.understand()   ← 意图识别 + 参数完整性检查 + 澄清机制
  │  └─ IntentRecognizer         ← LLM 单次调用（意图/复杂度/工具/目标/约束 一次返回）
  │
  ▼
Orchestrator.handle_request()
  │
  ├─ SIMPLE  ──▶  TaskPlanner.execute()
  │                 ├─ SIMPLE_CHAT  → 直接 LLM
  │                 ├─ RAG_*        → RAGEngine（混合检索）
  │                 ├─ MCP_EXECUTE  → ReAct → UnifiedReActExecutor
  │                 ├─ MEDIUM       → Plan-and-Execute（带 evaluator 反馈）
  │                 └─ COMPLEX      → 分层规划 → Plan-and-Execute × N阶段
  │
  └─ MEDIUM/COMPLEX  ──▶  5阶段生命周期
       Phase1: 理解          LLM 提取目标与约束
       Phase2: 规划          LLM 分解子任务，决定主Agent做/委派子Agent
       Phase3: 执行
         ├─ 主Agent任务  → TaskPlanner.execute()（复杂度自动推断）
         └─ 委派任务     → SubAgentFactory → DynamicSubAgent → UnifiedReActExecutor
       Phase4: 整合          拼接结果 + 关键操作校验（如订单号检查）
       Phase5: 交付          LLM 生成用户友好回复
  │
  ▼
OutputGuard → Memory update → AgentEvaluator（记录真实成功/失败状态）
```

### 多 Agent 协作架构

#### 主 Agent 三重身份

主 Agent（Orchestrator）在一次对话中同时扮演三个角色：

| 角色 | 职责 |
|------|------|
| **执行者** | 简单/中等任务自己通过 TaskPlanner 完成 |
| **指挥官** | 复杂任务分解子任务、动态创建子 Agent 并行委派、汇总结果 |
| **对话者** | 唯一与用户直接交互的角色，负责最终回复 |

#### 多 Agent 协作全景图

```
                          用户
                           │
                           ▼
              ┌────────────────────────┐
              │      主 Agent          │
              │   (Orchestrator)       │◀──── OrchestratorMemory
              │                        │       方案/实体/决策缓存
              │  Phase1: 理解目标       │
              │  Phase2: 分解子任务 ────┼──────────────────────────┐
              └────────────┬───────────┘                          │
                           │                                      │
              ┌────────────▼───────────┐              ┌──────────▼──────────┐
              │  主 Agent 自执行任务    │              │    委派子任务         │
              │  (assigned_to=main)    │              │ (assigned_to=sub_agent)│
              │                        │              └──────────┬──────────┘
              │  TaskPlanner.execute() │                         │
              │  ├─ RAG 检索           │              ┌──────────▼──────────┐
              │  ├─ ReAct 工具调用     │              │   SubAgentFactory    │
              │  └─ Plan-and-Execute  │              │                      │
              └────────────┬───────────┘              │  ┌─────────────────┐│
                           │                          │  │  CircuitBreaker  ││
                           │                          │  │  按工具类别分组  ││
                           │                          │  │  order_ops       ││
                           │                          │  │  product_search  ││
                           │                          │  │  user_ops        ││
                           │                          │  └─────────────────┘│
                           │                          │                      │
                           │        asyncio.gather    │  并行创建 N 个子Agent │
                           │         ┌────────────────┼──────┬───────┐      │
                           │         │                │      │       │      │
                           │  ┌──────▼──────┐  ┌─────▼────┐ │ ┌────▼────┐  │
                           │  │DynamicSubAgent│  │DynamicSub│ │ │DynamicSub│ │
                           │  │             │  │  Agent   │ │ │  Agent  │  │
                           │  │ 角色: 搜索   │  │ 角色: 推荐│ │ │角色: 下单│  │
                           │  │ 工具白名单   │  │ 工具白名单│ │ │工具白名单│  │
                           │  │ 独立上下文   │  │ 独立上下文│ │ │独立上下文│  │
                           │  │      │      │  │    │     │ │ │   │     │  │
                           │  │ UnifiedReAct│  │UnifiedReAct│ │ │UnifiedReAct│
                           │  │  Executor   │  │ Executor │ │ │ Executor│  │
                           │  │   +ToolCaller│  │+ToolCaller│ │ │+ToolCaller│ │
                           │  └──────┬──────┘  └─────┬────┘ │ └────┬────┘  │
                           │         │                │      │      │       │
                           │         └────────────────┘      └──────┘       │
                           │                    │                           │
                           │              SubAgentResult                    │
                           │            (只返回 summary 摘要)                │
                           │            子Agent内部过程对主Agent不可见        │
                           │                    │                           │
                           └──────────────────▶─┘◀──────────────────────────┘
                                               │
                              Phase4: 整合所有结果 + 质量校验
                              Phase5: 生成最终回复
                                               │
                                               ▼
                                             用户
```

#### 委派决策规则

满足以下任一条件，子任务交给子 Agent 处理，否则主 Agent 自己执行：

| 条件 | 说明 |
|------|------|
| **可并行加速** | 多个子任务相互独立，并行比串行快 |
| **上下文隔离** | 子任务会产生大量中间结果，隔离避免污染主上下文 |
| **离题子任务** | 与当前主线无关，不影响主 Agent 继续推进 |

主 Agent 执行的子任务（`assigned_to=main`）其复杂度会被自动推断——描述中含"然后/接着/再"等多步连接词时升级为 MEDIUM，走 Plan-and-Execute 路径，而非强制 SIMPLE。

#### 子 Agent 隔离性

每个动态子 Agent 拥有独立的：
- **角色身份**（`role`）：由主 Agent 按子任务需要注入的 system prompt
- **工具白名单**（`allowed_tools`）：只能调用被授权的工具
- **上下文窗口**：由 ContextWindowManager 独立管理，防止溢出
- **生命周期**：执行完毕即销毁，不跨任务保留状态

主 Agent **只看子 Agent 的 `summary` 摘要**——子 Agent 内部经历了多少次工具重试、走了多少推理轮次，主 Agent 完全不感知，如同只收到一封"结论邮件"。

#### 失败处理链

```
子Agent执行失败
      │
      ├─ 还有重试次数？─── 是 ──▶ SubAgentFactory 重试
      │                              │
      │                         成功 ──▶ 继续
      │                         仍失败 ──▶ ↓
      │
      └─ 降级到主Agent直接执行（assigned_to 改为 main）
                 │
            仍然失败 ──▶ 标记该子任务失败，取消所有依赖此任务的后续子任务

熔断器（CircuitBreaker，按 order_ops / product_search / user_ops / general 分组）：
  连续失败 N 次 ──▶ OPEN（拒绝该类委派）
  冷却期结束    ──▶ HALF_OPEN（允许一次试探）
  试探成功      ──▶ CLOSED（恢复正常）
```

SmartAgent 初始化采用**容错降级**策略——RAGEngine、MCPManager、LongTermMemory 任一初始化失败，均不影响其他组件运行，Agent 以降级模式提供服务，可通过 `agent.get_health_status()` 查询各组件状态。

### RAG 知识检索

商品知识库基于 ChromaDB 向量数据库，支持多种高级检索策略：

| 技术 | 作用 |
|------|------|
| 混合检索 | 向量相似度（70%）+ BM25 关键词（30%）双路召回 |
| 查询重写路由 | 4 级路由替代原无条件重写（见下表），按需选择策略 |
| HyDE | 先生成假设性回答再检索，提高语义匹配精度 |
| 重排序 | 对初筛结果用 LLM 重排序，取 Top-K |
| 父子分块 | 小块检索、大块返回，兼顾精度和上下文完整性 |

#### 查询重写 4 级路由（`rag/query_rewrite_router.py`）

路由决策互斥，命中即退出：

| 级别 | 触发条件 | 策略 |
|------|---------|------|
| **DEEP**（Level 3） | `rag_advanced` 意图 | 完整多查询展开 + HyDE，最大召回 |
| **SKIP**（Level 0） | 无上下文 / 纯关键词短查询 / 长查询且无触发词 | 不重写，直接检索 |
| **COREF**（Level 1） | 含指代词（"它/这款/哪款"等）且有上下文 | 轻量指代消解（temperature=0.1） |
| **SEMANTIC**（Level 2） | 含模糊词（"好不好/值不值"等）且查询较短 | 完整语义扩写 |
| **SKIP**（默认） | 不匹配以上任何条件 | 不重写 |

### 记忆系统

| 层级 | 存储 | 生命周期 | 作用 |
|------|------|----------|------|
| 短期记忆 | 内存 | 单次会话 | 保留最近对话，滚动总结防溢出 |
| 长期记忆 | ChromaDB | 跨会话 | 历史对话摘要语义检索 + 用户偏好（独立 collection） |
| 任务状态 | SQLite (`task_states` 表) | 可恢复 | 序列化 TaskState，独立存储，会话恢复时重建 |
| Orchestrator 记忆 | 内存 | 单次会话 | 子 Agent 结果缓存（推荐方案/实体/用户决策） |

#### 用户偏好（长期记忆子集）

偏好数据存储在 `LongTermMemory.preferences_collection`（独立 ChromaDB 集合），与对话摘要集合并列，均属长期记忆模块（`memory/long_term_memory.py`）。

- **提取**：每轮对话结束后，后台线程异步调用 LLM 提取偏好（budget / interests / brands / usage_scenario），**不阻塞响应**
- **存储策略**：append-only，每次写入生成新文档（`user:{uid}:{pref_key}:{uuid8}`），历史版本完整保留
- **读取**：按 `pref_key` 分组，返回每个字段最新值；支持 `get_preference_history()` 获取完整变更记录
- **注入时机**：主线程检索长期记忆后，将用户偏好 prepend 到上下文（同步，快速路径）

### 意图驱动的工具过滤

MCPManager 的 `get_tools_for_context(intent_type, query)` 方法按意图阶段过滤工具列表，减少传入 LLM 的无关上下文：

| 意图类型 | 可见工具 | 说明 |
|---------|---------|------|
| `rag_simple` / `rag_advanced` | 仅 search 类 | 知识检索不需要下单工具 |
| `greeting` / `simple_chat` | 无工具 | 纯对话无需任何工具 |
| `mcp_execute` | 按 query 关键词动态推断 | 含下单词 → order+user；含搜索词 → search |
| 未知意图 | 全量返回 | 安全兜底，不过滤 |

工具按关键词自动分类为 `search / order / user / other` 四类，分类逻辑在 `mcp_manager.py` 的 `_classify_tool()` 方法中。

### 上下文工程

ContextWindowManager 在所有执行路径（TaskPlanner、UnifiedReActExecutor、DynamicSubAgent）中统一管理上下文预算：按 `short_term_memory / tools / tool_results / rag_results / long_term_memory` 分区动态分配，防止上下文溢出。

### 安全治理

- **输入验证**：注入攻击检测、敏感词过滤、长度限制
- **输出审查**：敏感信息脱敏、禁止内容拦截
- **操作确认**：下单需 confirm、退款/删除需 double confirm
- **频率限制**：全局 30次/分钟、单工具 10次/分钟、单会话 200次上限
- **权限控制**：基于角色的工具访问控制

---

## 快速开始

### 环境要求

- Python 3.10+
- 一个 OpenAI 兼容的 LLM 服务（Ollama 本地模型 / OpenAI API / 其他兼容服务）

### 第 1 步：安装依赖

```bash
git clone <repo-url> && cd smartShopping
pip install -r requirements.txt
```

### 第 2 步：配置 LLM

编辑 `config/settings.yaml`，设置你的 LLM 服务：

```yaml
llm:
  # 使用本地 Ollama
  base_url: "http://localhost:11434/v1"
  model: "qwen2.5:7b"
  api_key: "EMPTY"

  # 或使用 OpenAI
  # base_url: "https://api.openai.com/v1"
  # model: "gpt-4"
  # api_key: "sk-xxx"
```

### 第 3 步：初始化数据

```bash
# 初始化数据库（业务库 + 会话库，含种子用户和商品数据）
python -m init.init_db

# 导入商品描述到 RAG 向量知识库（35 条评测/FAQ/选购指南）
python -m init.import_product_kb --force
```

### 第 4 步：启动服务

```bash
# 终端 1：启动购物平台后端（MCP 服务，提供商品搜索/用户管理/订单管理）
python mcp_manager_module/test_server_sse.py

# 终端 2：启动交互对话（CLI 模式）
python interactive_chat.py

# 或启动 Web 界面
uvicorn web.app:app --reload --port 8080
```

### 第 5 步：登录并开始对话

使用默认测试账号登录：

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | admin123 | 管理员 |
| test | test123 | 普通用户 |

登录后即可开始对话：

```
你: 推荐一款适合拍照的手机
你: 小米14 Ultra和华为Mate 60 Pro哪个好
你: 帮我买一台小米14 Ultra
你: 查询我的订单
```

### 常用 CLI 命令

| 命令 | 作用 |
|------|------|
| `/help` | 查看所有命令 |
| `/sessions` | 列出历史会话 |
| `/new` | 创建新会话 |
| `/switch <编号>` | 切换到历史会话 |
| `/verbose on/off` | 开关详细执行过程 |
| `/quit` | 退出 |

---

## 项目结构

```
smartShopping/
│
├── init/                            # 🗄️ 数据初始化（唯一的建表和种子数据入口）
│   ├── seed_schema.sql              #    业务表 DDL（users/products/orders/...）
│   ├── seed_schema_sessions.sql     #    会话表 DDL（sessions/messages/task_states）
│   ├── seed_data.sql                #    商品种子数据（15 个 SKU）
│   ├── product_descriptions.json    #    商品知识素材（35 条评测/FAQ/指南）
│   ├── init_db.py                   #    数据库初始化脚本
│   └── import_product_kb.py         #    向量库导入脚本
│
├── agent/                           # 🧠 Agent 核心
│   ├── agent.py                     #    SmartAgent 主入口（对话循环，容错降级）
│   ├── orchestrator.py              #    Orchestrator 编排器（5 阶段生命周期）
│   ├── orchestrator_memory.py       #    Orchestrator 结构化记忆（方案/实体/决策）
│   ├── task_state.py                #    任务状态模型（可序列化，支持会话恢复）
│   ├── task_planner.py              #    任务规划（ReAct / Plan-and-Execute / 分层规划）
│   ├── task_evaluator.py            #    任务评估（前/中/后三阶段，支持 replan）
│   ├── unified_react_executor.py    #    统一 ReAct 执行引擎（工具白名单/RAG/上下文管理）
│   ├── tool_caller.py               #    MCP/本地工具调用器（参数提取/Schema压缩/执行）
│   ├── dynamic_sub_agent.py         #    动态子 Agent（按需创建，直接 handle_task 调用）
│   ├── sub_agent_factory.py         #    子 Agent 工厂（并行执行 + 分类熔断器）
│   ├── sub_agent_base.py            #    子 Agent 基类（只定义 handle_task 接口）
│   ├── intent_recognizer.py         #    意图识别（LLM 一次返回意图/复杂度/工具/目标）
│   ├── goal_understanding.py        #    目标理解（参数完整性检查 + 澄清机制）
│   └── product_knowledge.py         #    商品 FAQ 生成器
│
├── config/                          # ⚙️ 配置
│   ├── settings.yaml                #    主配置（LLM/RAG/安全/编排器...）
│   └── mcp_servers.yaml             #    MCP 服务端点配置
│
├── rag/                             # 🔍 RAG 检索引擎（混合检索/HyDE/重排序/查询重写路由）
│   ├── query_rewrite_router.py      #    查询重写 4 级路由（SKIP/COREF/SEMANTIC/DEEP）
│   └── ...                          #    rag_engine/hybrid_retriever/reranker/...
├── memory/                          # 💾 短期（滚动总结）+ 长期（向量）记忆
├── auth/                            # 🔐 用户认证（登录/Token）
├── session/                         # 📋 会话管理（SQLite，含 task_states 独立表）
├── mcp_manager_module/              # 🔧 MCP 协议管理 + 购物平台后端
├── context/                         # 📐 上下文窗口工程（分区预算管理）
├── input_gate/                      # 🛡️ 输入安全验证
├── security/                        # 🔒 输出审查 + 频率限制
├── evaluation/                      # 📊 Agent + RAG 效果评估
├── observability/                   # 📡 OpenTelemetry 追踪
├── tool_manager/                    # 🧰 工具管理（本地 + MCP，多层过滤）
├── utils/                           # 🔧 工具类（LLM 客户端/配置/日志/连接池）
├── web/                             # 🌐 Web 前端（FastAPI + 静态页面）
│
├── interactive_chat.py              # CLI 交互入口
└── requirements.txt                 # Python 依赖
```

---

## 核心配置说明

配置文件：`config/settings.yaml`

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `llm.base_url` | LLM 服务地址 | `http://localhost:11434/v1` |
| `llm.model` | 模型名称 | `qwen2.5:7b` |
| `embedding.model` | 嵌入模型路径 | `../data/model/bge-small-zh-v1.5` |
| `system_prompt.content` | 购物助手系统提示词 | 见配置文件 |
| `orchestrator.max_subtask_retries` | 子 Agent 失败重试次数 | `2` |
| `orchestrator.circuit_breaker` | 熔断器（失败阈值 + 冷却期） | `3次 / 60秒` |
| `security.human_confirmation` | 敏感操作确认级别 | 下单 confirm / 退款 double |

---

## 技术栈

| 层面 | 技术选型 |
|------|---------|
| LLM | OpenAI 兼容 API（Ollama / OpenAI / 其他） |
| 向量数据库 | ChromaDB（本地持久化） |
| 嵌入模型 | BGE-small-zh-v1.5（本地）/ Ollama / OpenAI |
| 工具协议 | MCP (Model Context Protocol) over SSE |
| 数据库 | SQLite（用户/商品/订单/会话/任务状态） |
| 可观测性 | OpenTelemetry → Jaeger / Grafana Tempo |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | 原生 HTML/CSS/JS |

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| 数据库表不存在 | 运行 `python -m init.init_db --reset` |
| MCP 工具调用失败 | 确认 MCP 服务已启动：`python mcp_manager_module/test_server_sse.py` |
| 商品搜索无结果 | 运行 `python -m init.import_product_kb --force` 导入知识库 |
| LLM 连接失败 | 检查 `settings.yaml` 中的 `base_url` 和 `model`，确认 LLM 服务在线 |
| 登录失败 | 先运行 `python -m init.init_db` 初始化用户数据 |
| Agent 部分功能不可用 | 调用 `agent.get_health_status()` 查看各组件状态，确认降级组件 |
