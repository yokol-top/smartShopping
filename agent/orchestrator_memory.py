"""
Orchestrator 任务记忆

主Agent作为orchestrator，需要维护子Agent之间的结构化信息流转：
- 从子Agent结果中提取结构化事实（商品、方案、订单等）
- 解析用户引用（"方案一"→具体商品列表）
- 向子Agent构建任务时注入所需的结构化信息
- 仅当自身也无法解析时，才向用户发起澄清

生命周期：
- 会话级存储，跨轮次保留
- 每轮子Agent返回后更新
- 用户切换话题时逐步淘汰旧记忆
"""

import logging
from typing import Dict, Any, List, Optional


class OrchestratorMemory:
    """Orchestrator 的结构化任务记忆

    存储子Agent产出的结构化信息，供后续轮次的：
    1. 目标理解层：解析用户引用（"方案一"→具体商品）
    2. 子Agent路由：enrich context
    3. 子Agent信息请求：orchestrator 从记忆中应答
    """

    def __init__(self, llm_client=None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.llm_client = llm_client

        # 推荐方案缓存：{"方案一": {products: [...], total: 15397}, ...}
        self._recommendations: Dict[str, Dict[str, Any]] = {}

        # 关键实体缓存：最近交互中产生的实体（商品、订单、地址、银行卡等）
        self._entities: Dict[str, Dict[str, Any]] = {}

        # 用户决策记录：用户做出的选择（"选了方案一"、"确认下单"等）
        self._decisions: List[Dict[str, str]] = []

        # 用户已选择的方案商品（结构化：选定后仅保留该方案的商品）
        self._selected_plan_name: str = ""
        self._selected_products: List[Dict[str, Any]] = []


    # ================================================================
    # 写入：子Agent完成后由 orchestrator 调用
    # ================================================================

    def update_from_sub_agent_result(
            self, route: str, query: str, response: str,
    ):
        """从子Agent结果中提取结构化信息并存储

        orchestrator 在每次 dispatch 完成后调用此方法。
        """
        # 不依赖 route 字符串，基于响应内容自动判断提取策略
        # 总是提取实体信息（订单号、商品ID等）
        self._extract_entities_from_text(response)

        # 如果响应包含推荐方案特征，额外提取方案结构
        recommendation_hints = ['方案', '推荐', '¥', '套餐', '组合', '商品ID', 'P0']
        if any(hint in response for hint in recommendation_hints):
            self._extract_recommendations(response)

        self.logger.info(
            f"[OrchestratorMemory] 更新记忆 | route={route} | "
            f"方案数={len(self._recommendations)} | "
            f"实体数={len(self._entities)} | "
            f"决策数={len(self._decisions)}"
        )

    def record_user_decision(self, decision: str, resolved_info: Dict[str, Any] = None):
        """记录用户的决策"""
        entry = {"decision": decision}
        if resolved_info:
            entry["resolved"] = resolved_info
        self._decisions.append(entry)
        self.logger.info(f"[OrchestratorMemory] 记录用户决策: {decision}")

    def select_plan(self, plan_name: str):
        """用户选择某个方案后，提取该方案的商品并结构化存储

        选定后：
        1. 将该方案的商品列表提升为 _selected_products
        2. 记录用户决策
        3. 后续 get_context_for_sub_agent 仅输出已选商品，不再输出所有方案
        """
        plan = self._recommendations.get(plan_name)
        if not plan:
            self.logger.warning(f"[OrchestratorMemory] 未找到方案: {plan_name}")
            return

        self._selected_plan_name = plan_name
        self._selected_products = plan.get("products", [])

        # 将选定商品的ID存入实体缓存
        for p in self._selected_products:
            pid = p.get("product_id", "")
            if pid:
                self._entities[pid] = p

        # 记录决策
        product_ids = [p.get("product_id", "") for p in self._selected_products if p.get("product_id")]
        self.record_user_decision(
            f"选择{plan_name}",
            {"plan": plan_name, "product_ids": product_ids, "products": self._selected_products}
        )
        self.logger.info(
            f"[OrchestratorMemory] 用户选择{plan_name} | "
            f"商品IDs: {product_ids} | 商品数: {len(self._selected_products)}"
        )

    # ================================================================
    # 读取：为目标理解和子Agent提供结构化信息
    # ================================================================

    def resolve_reference(self, user_query: str, context: str = "") -> Optional[str]:
        """尝试解析用户查询中的引用

        如 "方案一" → 返回方案一的具体商品列表和价格，并在用户选择时记录决策
        如 "那个耳机" → 返回上下文中讨论的耳机信息

        Returns:
            解析后的补充信息，或 None（表示无法解析，需要澄清）
        """
        # 判断用户是否在做选择动作
        is_selecting = any(kw in user_query for kw in [
            "选", "下单", "购买", "买", "就这个", "确认",
        ])

        # 1. 尝试匹配方案引用
        for key, plan in self._recommendations.items():
            if key in user_query:
                if is_selecting:
                    self.select_plan(key)
                return self._format_plan(key, plan)

        # 2. 用 LLM 做模糊匹配（"方案一"、"第一个"、"推荐的那个"等）
        if self._recommendations and self.llm_client:
            resolved_name = self._llm_resolve_plan_name(user_query, context)
            if resolved_name:
                if is_selecting:
                    self.select_plan(resolved_name)
                return self._format_plan(resolved_name, self._recommendations[resolved_name])

        # 3. 从实体缓存中查找
        for entity_id, entity in self._entities.items():
            if entity_id in user_query or entity.get("name", "") in user_query:
                return f"[已知信息] {entity}"

        return None

    def get_context_for_sub_agent(self) -> str:
        """生成供子Agent/TaskPlanner使用的结构化上下文

        当用户已选择方案时，仅输出已选方案的商品（精简上下文）。
        未选择时输出所有推荐方案。
        """
        parts = []

        if self._selected_products:
            # 用户已选择方案，仅输出选定商品（结构化）
            parts.append(f"[用户已选方案: {self._selected_plan_name}]")
            product_ids = []
            for p in self._selected_products:
                pid = p.get('product_id', '')
                name = p.get('name', '')
                price = p.get('price', '')
                parts.append(f"- [{pid}] {name} ¥{price}")
                if pid:
                    product_ids.append(pid)
            if product_ids:
                parts.append(f"商品ID列表: {','.join(product_ids)}")
        elif self._recommendations:
            # 未选择，输出所有方案（精简格式）
            parts.append("[已推荐方案]")
            for key, plan in self._recommendations.items():
                products = plan.get("products", [])
                if products:
                    items = ", ".join(
                        f"[{p.get('product_id', '')}]{p.get('name', '')}" for p in products
                    )
                    total = plan.get("total_price", "")
                    parts.append(f"{key}: {items}{f' 合计¥{total}' if total else ''}")
                else:
                    text = plan.get("text", "")[:150]
                    parts.append(f"{key}: {text}")

        if self._entities:
            parts.append("\n[已知实体]")
            for eid, info in list(self._entities.items())[-5:]:
                parts.append(f"- {eid}: {info}")

        if self._decisions:
            parts.append("\n[用户决策]")
            for d in self._decisions[-3:]:
                parts.append(f"- {d['decision']}")
                if 'resolved' in d:
                    resolved = d['resolved']
                    if isinstance(resolved, dict) and 'product_ids' in resolved:
                        parts.append(f"  商品IDs: {resolved['product_ids']}")
                    else:
                        parts.append(f"  详情: {str(resolved)[:150]}")

        return "\n".join(parts) if parts else ""

    def answer_sub_agent_question(self, question: str) -> Optional[str]:
        """尝试从记忆中回答子Agent的问题

        Returns:
            回答文本，或 None（表示记忆中无此信息）
        """
        # 检查是否在问方案相关内容
        for key, plan in self._recommendations.items():
            if key in question or "方案" in question:
                return self._format_plan(key, plan)

        # 检查实体信息
        for eid, info in self._entities.items():
            if eid in question:
                return str(info)

        return None

    # ================================================================
    # 内部：信息提取
    # ================================================================

    def _extract_recommendations(self, response: str):
        """从售前子Agent的推荐响应中提取方案结构"""
        if not self.llm_client:
            # 无 LLM 时用简单规则
            self._simple_extract_recommendations(response)
            return

        prompt = f"""从以下商品推荐回复中提取方案信息。

回复内容：
{response[:1500]}

请以JSON格式返回，结构如下：
{{
  "方案一": {{
    "name": "方案简称",
    "products": [
      {{"product_id": "P006", "name": "华为MateBook X Pro", "price": 11999, "quantity": 1}},
      ...
    ],
    "total_price": 15397
  }},
  "方案二": {{ ... }}
}}

要求：
1. 保留所有商品ID（如P006）和价格
2. 如果回复中没有方案结构，返回空对象 {{}}
3. 只返回JSON"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.1)
            resp = resp.strip()
            if resp.startswith("```"):
                lines = resp.split("\n")
                resp = "\n".join(l for l in lines if not l.startswith("```"))

            import json
            plans = json.loads(resp)
            if plans:
                self._recommendations = plans
                # 同时将商品信息存入实体缓存
                for plan_name, plan_data in plans.items():
                    for product in plan_data.get("products", []):
                        pid = product.get("product_id", "")
                        if pid:
                            self._entities[pid] = product
                self.logger.info(
                    f"[OrchestratorMemory] 提取到 {len(plans)} 个推荐方案"
                )
        except Exception as e:
            self.logger.warning(f"[OrchestratorMemory] 方案提取失败: {e}")
            self._simple_extract_recommendations(response)

    def _simple_extract_recommendations(self, response: str):
        """简单规则提取（LLM不可用时的降级方案）"""
        import re
        # 匹配 "方案一"、"方案二" 等模式，保存对应的文本段
        sections = re.split(r'(##\s*🥇|##\s*🥈|##\s*🥉|##\s*方案)', response)
        plan_idx = 0
        plan_names = ["方案一", "方案二", "方案三", "方案四"]
        for i, section in enumerate(sections):
            if "方案" in section or "🥇" in section or "🥈" in section or "🥉" in section:
                if i + 1 < len(sections) and plan_idx < len(plan_names):
                    self._recommendations[plan_names[plan_idx]] = {
                        "text": sections[i + 1][:400],
                    }
                    plan_idx += 1

    def _extract_operation_results(self, response: str):
        """从功能子Agent的操作结果中提取实体信息"""
        self._extract_entities_from_text(response)

    def extract_entities_from_step_result(self, step_result: str):
        """从TaskPlanner单步执行结果中提取实体信息

        在每个步骤完成后由TaskPlanner回调调用，将结果中的ID和关联详情
        存入orchestrator记忆，避免后续步骤重复查询。

        Args:
            step_result: 步骤执行结果文本（MCP工具返回值，可信来源）
        """
        if not step_result:
            return
        old_count = len(self._entities)
        # trusted_source=True：工具返回值是可信地址来源
        self._extract_entities_from_text(step_result, trusted_source=True)
        new_count = len(self._entities)
        if new_count > old_count:
            new_ids = list(self._entities.keys())[old_count:]
            self.logger.info(
                f"[OrchestratorMemory] 从步骤结果中提取到 {new_count - old_count} 个实体: {new_ids}"
            )

    def _extract_entities_from_text(self, text: str, trusted_source: bool = False):
        """从文本中提取结构化实体信息（ID + 关联详情）

        Args:
            text: 待提取文本
            trusted_source: 是否来自可信的工具执行结果。
                True  → 允许写入 _address（真实 MCP 返回值）
                False → 跳过 _address 提取（AI 生成文本，避免将提问内容误存为地址）
        """
        import re

        # 提取各类ID
        id_patterns = {
            "order_id": r'(ORD-\d+)',
            "user_id": r'(UID-\d+)',
            "card_id": r'(CARD-\d+)',
            "addr_id": r'(ADDR-\d+)',
            "product_id": r'(P\d{3,})',
        }
        for id_type, pattern in id_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in self._entities:
                    self._entities[match] = {"type": id_type, "id": match}

        # 从结构化数据中提取ID与详情的关联（如 'id': 'ADDR-427', 'location': '北京...'）
        # 匹配 {'id': 'XXX-nnn', 'key': 'value'} 格式的dict片段
        dict_pattern = r"\{[^{}]*'id'\s*:\s*'([A-Z]+-\d+)'[^{}]*\}"
        for m in re.finditer(dict_pattern, text):
            entity_id = m.group(1)
            try:
                import ast
                entity_dict = ast.literal_eval(m.group(0))
                if isinstance(entity_dict, dict) and entity_id:
                    self._entities[entity_id] = entity_dict
            except (ValueError, SyntaxError):
                pass

        # 提取地址详情：仅信任工具执行结果，不从 AI 生成文本中提取
        # 避免把 AI 的提问内容（如"是否两件一起下单或先下某一件"）误存为地址
        if trusted_source:
            addr_patterns = [
                r'(?:地址|收货地址|address|location)[：:\s]*([^\n,，]+)',
            ]
            for pattern in addr_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    addr_val = match.group(1).strip().rstrip('。）)')
                    if len(addr_val) > 5:
                        self._entities.setdefault("_address", {})["location"] = addr_val

        # 提取手机号
        phone_matches = re.findall(r'(?:电话|手机|phone|联系)[：:\s]*(1\d{10})', text)
        for phone in phone_matches:
            self._entities.setdefault("_phone", {})["phone"] = phone

        # 提取姓名（收货人）
        name_matches = re.findall(r'(?:收货人|收件人|姓名|customer)[：:\s]*(\S{2,8})', text)
        for name in name_matches:
            if name not in ('请提供', '是什么', '的真实'):
                self._entities.setdefault("_customer", {})["name"] = name

    def extract_entities_from_context(self, context: str):
        """从主Agent对话上下文中扫描并提取实体信息

        解决问题：用户通过主Agent的TaskPlanner添加的地址/银行卡信息
        不经过子Agent路由，orchestrator memory不知道这些实体。
        在每次enrich_context时调用此方法补充。
        """
        if not context:
            return
        old_count = len(self._entities)
        self._extract_entities_from_text(context)
        new_count = len(self._entities)
        if new_count > old_count:
            self.logger.info(
                f"[OrchestratorMemory] 从上下文中补充了 {new_count - old_count} 个实体"
            )

    def _format_plan(self, plan_name: str, plan_data: Dict[str, Any]) -> str:
        """格式化方案为可读文本"""
        if "text" in plan_data:
            return f"{plan_name}: {plan_data['text'][:300]}"

        products = plan_data.get("products", [])
        total = plan_data.get("total_price", "")
        lines = [f"{plan_name}:"]
        for p in products:
            name = p.get("name", "")
            price = p.get("price", "")
            pid = p.get("product_id", "")
            qty = p.get("quantity", 1)
            lines.append(f"  - [{pid}] {name} ¥{price} x{qty}")
        if total:
            lines.append(f"  合计: ¥{total}")
        return "\n".join(lines)

    def _llm_resolve_plan_name(self, query: str, context: str) -> Optional[str]:
        """用 LLM 解析模糊引用，返回匹配的方案名"""
        import json
        plans_str = json.dumps(self._recommendations, ensure_ascii=False)

        prompt = f"""用户说了一句话，判断是否引用了之前推荐的方案。

用户说：{query}
对话上下文（最近）：{context[-300:]}

已有方案：
{plans_str[:800]}

如果用户引用了某个方案，返回方案名（如"方案一"）。
如果没有引用任何方案，返回"无"。
只返回方案名或"无"。"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.1).strip()
            if resp != "无" and resp in self._recommendations:
                return resp
        except Exception:
            pass
        return None

    # ================================================================
    # 淘汰策略
    # ================================================================

    def clear_on_topic_change(self):
        """用户明显切换话题时清理推荐方案缓存"""
        self._recommendations.clear()
        self._decisions.clear()
        self.logger.info("[OrchestratorMemory] 话题切换，清理方案和决策缓存")

    def clear_all(self):
        """完全清理（会话结束时）"""
        self._recommendations.clear()
        self._entities.clear()
        self._decisions.clear()
        self._selected_plan_name = ""
        self._selected_products.clear()
