# SmartShopping — 智能购物助手

## 这是什么

SmartShopping 是一个 **AI 驱动的智能购物平台**。用户通过自然语言对话，即可完成从"我想买个手机"到"订单已创建"的完整购物流程——无需翻页、搜索、加购物车。

它不是一个简单的聊天机器人，而是一个具备**理解、规划、并行执行、整合**能力的多 Agent 系统。

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

系统内置 **35 条**商品知识文档（`init/product_descriptions.json` 自动生成），包括：

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
│  │  滚动摘要窗口         语义历史+用户偏好    方案/实体/决策    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────┐    │
│  │ 目标理解层                                                 │    │
│  │   IntentRecognizer (LLM) ── GoalUnderstanding            │    │
│  │   意图/复杂度/工具/目标      参数完整性检查 + 澄清机制      │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────┐    │
│  │ Orchestrator 编排层（intent_type × complexity 路由矩阵）   │    │
│  │                                                           │    │
│  │  RAG_SIMPLE/CHAT  →  直接执行（RAG 检索 / LLM 回答）      │    │
│  │                                                           │    │
│  │  RAG_ADVANCED                                             │    │
│  │    medium → Plan-and-Execute（步骤 DAG 并行）             │    │
│  │    complex → 分层规划（阶段 DAG 并行 + SubAgent）          │    │
│  │                                                           │    │
│  │  MCP_EXECUTE                                              │    │
│  │    simple  → ReAct（单步探索）                             │    │
│  │    medium  → Plan-and-Execute（步骤 DAG 并行）             │    │
│  │    complex → 5 阶段生命周期（SubAgent 并行 + 熔断保护）     │    │
│  │                                                           │    │
│  │  ※ is_plannable=False → 强制走 ReAct（覆盖矩阵）           │    │
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
│  HyDE/重排  │       └─ MCP Server         │                    │
│  查询重写   │          购物平台后端         │                    │
│  4级路由    │          (商品/订单/用户)    │                    │
└─────────────┴──────────────┬──────────────┴────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                          数据存储层                                 │
│                                                                   │
│   SQLite                                  ChromaDB               │
│   ├─ data/db/app.db                       ├─ 商品知识向量库          │
│   │  users / products / orders / ...     │  （评测/FAQ/选购指南）    │
│   ├─ data/sessions/sessions.db           └─ 长期记忆               │
│   │  sessions / messages / task_states      （历史摘要+用户偏好）    │
│   └─ data/eval_store.db（评估结果）                                │
│      task_records / regression_reports / rag_scores              │
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
  ├─ LongTermMemory (ChromaDB)   ← 语义检索历史摘要 + 注入用户偏好（后台异步提取）
  │
  ▼
GoalUnderstanding.understand()   ← 意图识别 + 参数完整性检查 + 澄清机制
  │  └─ IntentRecognizer (LLM)   ← 一次调用返回：意图类型/复杂度/工具/目标/约束/is_plannable
  │
  ▼
Orchestrator.handle_request()    ← intent_type × complexity 路由矩阵
  │
  ├─[is_plannable=False]──────────▶ ReAct（UnifiedReActExecutor，边做边看）
  │
  ├─[RAG_SIMPLE / SIMPLE_CHAT]───▶ 直接执行（RAGEngine / LLM）
  │
  ├─[RAG_ADVANCED / medium]──────▶ Plan-and-Execute
  │                                  ├─ LLM 生成步骤计划（含 depends_on 依赖声明）
  │                                  ├─ 拓扑排序 → 执行波次
  │                                  └─ 同波次独立步骤 ThreadPoolExecutor 并行
  │
  ├─[RAG_ADVANCED / complex]─────▶ 分层规划
  │                                  ├─ LLM 生成阶段计划（含 depends_on 依赖声明）
  │                                  ├─ 拓扑排序 → 执行波次
  │                                  └─ 同波次独立阶段 SubAgentFactory 并行
  │                                       └─ 每个阶段内走 Plan-and-Execute
  │
  ├─[MCP_EXECUTE / simple]───────▶ ReAct（UnifiedReActExecutor）
  │
  ├─[MCP_EXECUTE / medium]───────▶ Plan-and-Execute（同上，步骤 DAG 并行）
  │
  └─[MCP_EXECUTE / complex]──────▶ 5 阶段生命周期
       Phase1: 理解          LLM 提取目标与约束
       Phase2: 规划          LLM 分解子任务，决定主Agent做/委派子Agent
       Phase3: 执行
         ├─ 主Agent任务  → TaskPlanner（Plan-and-Execute / ReAct）
         └─ 委派任务     → SubAgentFactory → DynamicSubAgent（asyncio.gather 并行）
                               └─ UnifiedReActExecutor + ToolCaller
       Phase4: 整合          拼接结果 + 关键操作校验（如订单号检查）
       Phase5: 交付          LLM 生成用户友好回复
  │
  ▼
OutputGuard → PostResponsePipeline（记忆更新 + 偏好提取 + 评估）
```

### 路由矩阵详解

Orchestrator 的路由完全基于两个高置信度信号：**意图类型**（LLM 识别）和**复杂度**（LLM 识别），不依赖任何不稳定的中间信号。

```
                   simple              medium                  complex
──────────────────────────────────────────────────────────────────────────
RAG_SIMPLE      直接 RAG 查询       直接 RAG 查询           Plan-and-Execute
                （单问题，最快）     （单问题，最快）         （多步分析）

RAG_ADVANCED    直接 RAG 查询       Plan-and-Execute        分层规划
                （单品咨询）         步骤 DAG 并行           阶段 DAG 并行
                                    （多步对比分析）         （跨类目方案）

MCP_EXECUTE     ReAct               Plan-and-Execute        5 阶段生命周期
                （单步操作）         步骤 DAG 并行           SubAgent 并行
                （查订单/查用户）     （推荐+下单流程）        （复杂采购任务）

CHAT/其他       直接 LLM 回答       直接 LLM 回答           直接 LLM 回答
──────────────────────────────────────────────────────────────────────────
覆盖规则：is_plannable=False → 强制走 ReAct（步骤未知，需边做边看）
```

**各路径典型示例：**

| 用户输入 | 意图/复杂度 | 路由路径 |
|---------|-----------|---------|
| "iPhone 15 Pro Max 怎么样" | rag_simple / simple | 直接 RAG |
| "帮我对比小米和华为两款手机" | rag_advanced / medium | P&E（并行搜索两款） |
| "预算2万配一套数码装备" | rag_advanced / complex | 分层规划（手机/电脑/耳机并行搜索，再综合推荐） |
| "查一下我的订单" | mcp_execute / simple | ReAct |
| "推荐台笔记本然后帮我下单" | mcp_execute / medium | P&E |
| "配齐全套装备全部下单" | mcp_execute / complex | 5 阶段生命周期 |
| "帮我探索一个最合适的方案" | - / is_plannable=false | ReAct（覆盖） |

### DAG 并行执行

Plan-and-Execute 和分层规划均支持基于 DAG 的并行执行，加速有独立子任务的复杂请求。

#### Plan-and-Execute 步骤级并行

LLM 在生成执行计划时，通过 `depends_on` 字段声明步骤间的依赖关系。执行时对步骤做拓扑排序，同一波次内的独立步骤通过 `ThreadPoolExecutor` 并发执行：

```
计划：[步骤1: 搜手机] [步骤2: 搜耳机] [步骤3: 综合推荐(依赖1,2)]
                                               ↓ 拓扑排序
波次1（并行）: 步骤1 ──┐
                      ├──▶ 波次2（串行）: 步骤3
波次1（并行）: 步骤2 ──┘

执行时间：max(步骤1, 步骤2) + 步骤3  <  步骤1 + 步骤2 + 步骤3
```

- **并行机制**：`ThreadPoolExecutor`，同波次共享波次开始时的历史快照（只读）
- **结果合并**：波次结束后按 step_id 升序批量追加到执行历史
- **评估时机**：单步完成后立刻评估；并行波次全部完成后统一评估一次

#### 分层规划阶段级并行

分层规划（`rag_advanced/complex`）在生成高层阶段计划时，通过 `depends_on` 声明阶段间依赖，同一波次内的独立阶段通过 `SubAgentFactory.execute_subtasks_parallel()` 并发执行：

```
阶段计划：[阶段1: 搜手机] [阶段2: 搜耳机] [阶段3: 综合推荐(依赖1,2)]
                                               ↓ 拓扑排序
波次1（并行）: 阶段1 ──┐  ← SubAgent 独立执行（内部走 Plan-and-Execute）
                      ├──▶ 波次2（串行）: 阶段3
波次1（并行）: 阶段2 ──┘  ← SubAgent 独立执行
```

- **并行机制**：`asyncio.gather` via `SubAgentFactory`，每个阶段由独立 SubAgent 执行
- **每个 SubAgent** 内部走完整的 Plan-and-Execute（含步骤级 DAG 并行）
- **串行兜底**：单阶段波次或无 factory 时直接串行执行，行为与旧版一致

### 多 Agent 协作架构

#### 主 Agent 三重身份

主 Agent（Orchestrator）在一次对话中同时扮演三个角色：

| 角色 | 职责 |
|------|------|
| **执行者** | 简单/中等任务自己通过 TaskPlanner 完成 |
| **指挥官** | 复杂 MCP 任务分解子任务、动态创建子 Agent 并行委派、汇总结果 |
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
              │      └─ 步骤DAG并行   │              │  ┌─────────────────┐ │
              └────────────┬───────────┘              │  │  CircuitBreaker  │ │
                           │                          │  │  按工具类别分组  │ │
                           │                          │  │  order_ops       │ │
                           │                          │  │  product_search  │ │
                           │                          │  │  user_ops        │ │
                           │                          │  └─────────────────┘ │
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
                           │         └────────────────┘      └──────┘       │
                           │                    │                            │
                           │              SubAgentResult                     │
                           │            (只返回 summary 摘要)                 │
                           │            子Agent内部过程对主Agent不可见         │
                           │                    │                            │
                           └──────────────────▶─┘◀───────────────────────────┘
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

> 注意：下单类工具（`create_complex_order` 等）在代码层强制只能由主 Agent 执行，Prompt 约束无法保证 100% 可靠，代码层兜底确保安全。

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
| 查询重写路由 | 4 级路由按需选择策略，替代原无条件重写（见下表） |
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
| 任务状态 | SQLite (`task_states` 表) | 可恢复 | 序列化 TaskState，会话恢复时重建 |
| Orchestrator 记忆 | 内存 | 单次会话 | 子 Agent 结果缓存（推荐方案/实体/用户决策） |

#### 用户偏好（长期记忆子集）

偏好数据存储在 `LongTermMemory.preferences_collection`（独立 ChromaDB 集合）：

- **提取**：每轮对话结束后，后台 daemon 线程异步调用 LLM 提取偏好（budget / interests / brands / usage_scenario），**不阻塞响应**
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
- **操作确认**：下单需 confirm、高危删除需 double confirm
- **频率限制**：全局 30次/分钟、单工具 10次/分钟、单会话 200次上限
- **权限控制**：基于角色的工具访问控制（管理员/普通用户）
- **代码层兜底**：下单工具只能由主 Agent 执行，不可委派子 Agent，Prompt 约束不可完全信任

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

## 评估体系

### 评估框架概览

系统内置完整的 Agent 评估框架，覆盖从单元测试到端到端回归的全链路质量保障。

```
┌─────────────────────────────────────────────────────────────────┐
│  黄金测试集 (GoldenTestSuite)                                    │
│  TC-INTENT-xxx  意图识别准确性                                   │
│  TC-E2E-xxx     端到端任务完成                                   │
│  TC-ROB-xxx     鲁棒性（注入攻击/异常输入）                       │
│  TC-MULTI-xxx   多轮对话连贯性                                   │
│  TC-RAG-xxx     RAG 检索质量                                     │
├─────────────────────────────────────────────────────────────────┤
│  回归运行器 (RegressionRunner)                                   │
│  支持按 category / priority / tags 筛选运行                      │
│  生成 JSON 报告，统计通过率 / 失败 case / 耗时                   │
├─────────────────────────────────────────────────────────────────┤
│  RAG 评估器 (RAGEvaluator，LLM-as-Judge)                        │
│  检索相关性 / 忠实性 / 回答相关性 / 上下文利用率                  │
├─────────────────────────────────────────────────────────────────┤
│  评估存储 (EvalStore → data/eval_store.db)                      │
│  task_records / regression_reports / rag_scores                 │
└─────────────────────────────────────────────────────────────────┘
```

### 运行评估

```bash
# 运行全部黄金测试
python -m evaluation.regression_runner

# 仅运行意图识别测试（快速，不启动完整 Agent）
python -m evaluation.regression_runner --category intent

# 仅运行 P0 核心用例
python -m evaluation.regression_runner --priority P0

# 按标签筛选
python -m evaluation.regression_runner --tags rag,mcp

# 生成 JSON 报告
python -m evaluation.regression_runner --report

# 通过 pytest 运行（可接入 CI）
pytest tests/test_golden_suite.py -v
pytest tests/test_golden_suite.py -m p0  # 仅 P0，用于 CI 门禁
```

### RAG 质量评估

```bash
# 对指定问题评估 RAG 链路质量
python -m evaluation.rag_evaluator
```

| 评估维度 | 说明 | 分数范围 |
|---------|------|---------|
| 检索相关性 | 召回的文档与问题是否相关 | 0 - 1 |
| 忠实性 | 回答是否忠实于检索内容，不编造信息 | 0 - 1 |
| 回答相关性 | 回答是否切题，真正解答了问题 | 0 - 1 |
| 上下文利用率 | 检索到的文档被有效利用了多少 | 0 - 1 |

### 评估用例优先级

| 优先级 | 含义 | CI 策略 |
|-------|------|---------|
| **P0** | 核心路径，不能有任何失败 | 失败阻断 PR 合并 |
| **P1** | 重要场景 | 失败发 Warning，不阻断 |
| **P2** | 边缘 case | 仅记录，用于趋势分析 |

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
│   ├── agent.py                     #    SmartAgent 主入口（对话循环 + 容错降级）
│   ├── orchestrator.py              #    Orchestrator（intent×complexity 路由 + 5阶段生命周期）
│   ├── orchestrator_memory.py       #    Orchestrator 结构化记忆（方案/实体/决策）
│   ├── task_state.py                #    任务状态模型（可序列化，支持会话恢复）
│   ├── task_planner.py              #    任务规划（ReAct/P&E/分层规划，均支持 DAG 并行）
│   ├── task_evaluator.py            #    任务评估（前/中/后三阶段，支持 replan）
│   ├── unified_react_executor.py    #    统一 ReAct 执行引擎（工具白名单/RAG/上下文管理）
│   ├── tool_caller.py               #    MCP/本地工具调用器（参数提取/Schema压缩/执行）
│   ├── dynamic_sub_agent.py         #    动态子 Agent（按需创建，执行完即销毁）
│   ├── sub_agent_factory.py         #    子 Agent 工厂（asyncio 并行 + 分类熔断器）
│   ├── sub_agent_base.py            #    子 Agent 基类（只定义 handle_task 接口）
│   ├── circuit_breaker.py           #    熔断器（三态：CLOSED/OPEN/HALF_OPEN）
│   ├── intent_recognizer.py         #    意图识别（LLM 一次返回意图/复杂度/工具/目标）
│   ├── intent_utils.py              #    轻量意图推断（关键词规则，无 LLM 调用）
│   ├── goal_understanding.py        #    目标理解（参数完整性检查 + 澄清机制）
│   ├── context_pipeline.py          #    上下文组装流水线（记忆 + 引用解析）
│   ├── post_response_pipeline.py    #    响应后处理（记忆更新 + 偏好提取 + 评估）
│   ├── exceptions.py                #    Agent 自定义异常（NeedUserInputException 等）
│   └── product_knowledge.py         #    商品 FAQ 自动生成器（按需生成）
│
├── config/                          # ⚙️ 配置
│   ├── settings.yaml                #    主配置（LLM/RAG/安全/编排器...）
│   ├── mcp_servers.yaml             #    MCP 服务端点配置
│   └── agent_templates.yaml         #    子 Agent 角色模板（预设工具白名单/角色/迭代数）
│
├── rag/                             # 🔍 RAG 检索引擎
│   ├── rag_engine.py                #    RAG 主引擎（检索 + 重排 + 生成）
│   ├── query_rewrite_router.py      #    查询重写 4 级路由（SKIP/COREF/SEMANTIC/DEEP）
│   ├── query_rewriter.py            #    查询重写器 + CoRefResolver（指代消解）
│   ├── hybrid_retriever.py          #    混合检索（向量70% + BM25 30%）
│   ├── hyde.py                      #    HyDE（假设文档生成）
│   ├── reranker.py                  #    LLM 重排序
│   ├── self_fix.py                  #    检索自修复（质量不足时重试）
│   ├── embeddings.py                #    嵌入模型（本地 BGE / Ollama / OpenAI）
│   ├── vector_store.py              #    ChromaDB 封装
│   ├── document_processor.py        #    父子分块处理器
│   └── document_loader.py           #    多格式文档加载器
│
├── memory/                          # 💾 记忆系统
│   ├── short_term_memory.py         #    短期记忆（滚动窗口 + LLM 总结）
│   └── long_term_memory.py          #    长期记忆（ChromaDB：历史摘要 + 用户偏好）
│
├── auth/                            # 🔐 用户认证（登录/Token）
├── session/                         # 📋 会话管理（SQLite，含 task_states 独立表）
├── mcp_manager_module/              # 🔧 MCP 协议管理 + 购物平台后端服务
├── context/                         # 📐 上下文窗口工程（分区预算管理）
├── input_gate/                      # 🛡️ 输入安全验证
├── security/                        # 🔒 输出审查（output_guard）+ 频率限制（rate_limiter）
├── evaluation/                      # 📊 Agent 评估体系
│   ├── golden_test_suite.py         #    黄金测试集（5类用例：意图/端到端/鲁棒性/多轮/RAG）
│   ├── regression_runner.py         #    自动化回归测试运行器（支持分类/优先级/标签筛选）
│   ├── rag_evaluator.py             #    RAG 质量评估（检索相关性/忠实性/回答相关性/上下文利用率）
│   ├── eval_store.py                #    评估结果持久化（SQLite，支持历史趋势查询）
│   ├── agent_evaluator.py           #    任务执行多维度评估（完成率/鲁棒性/效率/满意度/RAG质量）
│   └── __init__.py
│
├── tests/
│   └── test_golden_suite.py         #    pytest 黄金测试集入口（可接入 CI）
├── observability/                   # 📡 OpenTelemetry 追踪
├── tool_manager/                    # 🧰 统一工具管理（本地工具 + MCP 工具，多层过滤）
├── utils/                           # 🔧 工具类（LLM 客户端/配置加载/日志/连接池）
├── web/                             # 🌐 Web 界面（FastAPI + 静态页面）
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
| `security.human_confirmation` | 敏感操作确认级别 | 下单 confirm / 删除 double |
| `react.max_iterations` | ReAct 最大迭代轮次 | `5` |
| `planner.max_replan_attempts` | 评估失败最大重规划次数 | `2` |

子 Agent 角色模板：`config/agent_templates.yaml`

预置了 `order_creation / order_query / product_search / user_info` 四个模板，每个模板定义了工具白名单、角色描述和最大迭代次数，匹配规则为子任务工具集是模板工具集的子集时优先使用预置模板。

---

## 技术栈

| 层面 | 技术选型 |
|------|---------|
| LLM | OpenAI 兼容 API（Ollama / OpenAI / 其他） |
| 向量数据库 | ChromaDB（本地持久化） |
| 嵌入模型 | BGE-small-zh-v1.5（本地）/ Ollama / OpenAI |
| 工具协议 | MCP (Model Context Protocol) over SSE |
| 数据库 | SQLite（用户/商品/订单/会话/任务状态） |
| 并行执行 | `asyncio.gather`（SubAgent 阶段级）+ `ThreadPoolExecutor`（步骤级） |
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
| 并行阶段执行超时 | 调整 `config/settings.yaml` 中子 Agent 超时配置（默认 90s/阶段） |
| 评估测试全部失败 | 确认 Agent 已正常启动，或加 `--category intent` 单独跑意图识别测试 |
| RAG 评估分数低 | 运行 `python -m init.import_product_kb --force` 确认知识库已导入 |
