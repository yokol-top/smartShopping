"""
响应后处理流水线 (Post-Response Pipeline)

职责：响应生成后的所有后处理——
  - 更新短期记忆（智能摘要长响应）
  - 持久化会话消息
  - 自动生成会话标题（首条消息）
  - 触发长期记忆总结
  - 后台提取并存储用户偏好（含重试队列）
  - 通知 AgentEvaluator 任务完成

在 OutputGuard 通过后执行，偏好提取在后台线程运行，不影响响应延迟。
"""
import json as _json
import logging
import re as _re
import threading
import time
from collections import deque
from typing import Optional

from .context_pipeline import ContextBundle


class PostResponsePipeline:
    """
    响应后处理流水线。
    在 OutputGuard 通过后执行，不影响响应延迟（偏好提取在后台线程执行）。
    """

    def __init__(
        self,
        short_term_memory,
        long_term_memory,        # 可为 None
        session_manager,
        llm_client,
        agent_evaluator,
        config: dict,
        degraded_components: list,
        logger: logging.Logger = None,
    ):
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.session_manager = session_manager
        self.llm_client = llm_client
        self.agent_evaluator = agent_evaluator
        self.config = config
        self.degraded_components = degraded_components
        self.logger = logger or logging.getLogger(__name__)

        # 长期记忆总结阈值
        self.summary_threshold = config.get('memory', {}).get('long_term', {}).get('summary_threshold', 5)

        # 响应摘要配置
        response_summary_cfg = config.get('memory', {}).get('short_term', {}).get('response_summary', {})
        self.response_summary_enabled = response_summary_cfg.get('enabled', True)
        self.response_max_length = response_summary_cfg.get('max_length', 300)

        # 对话计数器（用于长期记忆总结触发）
        self.conversation_count = 0

        # 偏好提取重试队列：元素为 (user_id, user_input, response, attempt_count)
        self._pref_retry_queue: deque = deque(maxlen=100)
        self._start_retry_daemon()

    def run(
        self,
        task_id: str,
        user_input: str,
        response: str,
        bundle: ContextBundle,
        is_failed: bool = False,
        first_message: bool = False,
        token_usage: Optional[dict] = None,
    ):
        """
        执行全部后处理步骤：
        1. 将响应（智能摘要后）加入短期记忆
        2. 持久化响应到会话存储
        3. 如果是第一条消息，自动生成会话标题
        4. 更新 conversation_count，达到 summary_threshold 时触发长期记忆总结
        5. 后台线程提取用户偏好（含重试队列）
        6. agent_evaluator.task_completed(task_id, success=not is_failed)
        """
        # Step 1: 将响应加入短期记忆（长响应先摘要）
        response_for_memory = self._prepare_for_memory(response, user_input, bundle.context)
        self.short_term_memory.add_message("assistant", response_for_memory)
        self.short_term_memory.increment_message_count()

        # Step 2 & 3: 持久化 + 可选的自动标题
        if bundle.user_id:
            self.session_manager.add_message(bundle.session_id, 'assistant', response)
            if first_message:
                self.session_manager.auto_generate_title(bundle.session_id, user_input)

        # Step 4: 更新对话计数，按阈值触发长期记忆总结
        self.conversation_count += 1
        if self.conversation_count % self.summary_threshold == 0:
            self._generate_long_term_summary(bundle.session_id, bundle.user_id)

        # Step 5: 后台提取用户偏好
        if bundle.user_id and self.long_term_memory and 'long_term_memory' not in self.degraded_components:
            self._schedule_preference_extraction(bundle.user_id, user_input, response)

        # Step 6: Agent评估
        self.agent_evaluator.task_completed(
            task_id=task_id,
            success=not is_failed,
            error="执行降级" if is_failed else "",
            token_usage=token_usage,
        )

    def _prepare_for_memory(self, response: str, user_input: str, context: str) -> str:
        """长响应自动摘要后存入短期记忆（逻辑同原 _prepare_response_for_memory）"""
        if not self.response_summary_enabled:
            return response

        if len(response) <= self.response_max_length:
            self.logger.debug(f"响应长度({len(response)})在阈值内，直接存储")
            return response

        self.logger.info(f"响应过长({len(response)}字符)，准备提取关键信息后存储")

        try:
            summary = self._summarize_response(response, user_input, context)
            # P2 fix: 不将完整 response/summary 写入 INFO，避免 PII 泄露；
            # 详细内容仅在 DEBUG 级别输出（生产环境通常关闭 DEBUG）
            self.logger.info(
                f"响应摘要完成，原长度: {len(response)}, 摘要长度: {len(summary)}"
            )
            self.logger.debug(f"原响应（前200字）: {response[:200]}")
            self.logger.debug(f"摘要信息（前200字）: {summary[:200]}")
            return summary
        except Exception as e:
            self.logger.error(f"响应摘要失败: {e}，使用截断方式")
            return response[:self.response_max_length] + "...(已截断)"

    def _summarize_response(self, response: str, user_input: str, context: str) -> str:
        """使用 LLM 提取响应中的关键信息（逻辑同原 _summarize_response）"""
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

        return self.llm_client.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=int(self.response_max_length * 1.5),
        ).strip()

    def _generate_long_term_summary(self, session_id: str, user_id: str):
        """触发长期记忆总结（逻辑同原 _generate_long_term_summary）"""
        if not self.long_term_memory or 'long_term_memory' in self.degraded_components:
            return

        self.logger.info(f"生成长期记忆总结（对话轮次: {self.conversation_count}）")

        try:
            recent_messages = self.short_term_memory.get_messages()

            if len(recent_messages) < 2:
                return

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
                max_tokens=200,
            ).strip()

            topics = []  # 简化版，实际可以用NLP提取

            self.long_term_memory.add_conversation_summary(
                session_id=session_id,
                user_id=user_id or "",
                summary=summary,
                topics=topics,
            )

            self.logger.info("长期记忆总结已保存")
        except Exception as e:
            self.logger.error(f"生成长期记忆总结失败: {e}")

    def _schedule_preference_extraction(self, user_id: str, user_input: str, response: str):
        """后台线程提取偏好，失败时加入 _pref_retry_queue。"""
        def _task():
            try:
                prefs = self._extract_user_preferences(user_input, response)
                if prefs:
                    self.long_term_memory.update_user_preferences(user_id, prefs)
                    self.logger.info(f"[后台] 用户偏好已更新: {list(prefs.keys())}")
            except Exception as e:
                self.logger.debug(f"[后台] 偏好提取失败，加入重试队列: {e}")
                self._pref_retry_queue.append((user_id, user_input, response, 0))

        threading.Thread(target=_task, daemon=True, name="pref-update").start()

    def _start_retry_daemon(self):
        """启动重试队列消费线程（每 5 分钟），最多重试 3 次"""
        def _retry_loop():
            while True:
                time.sleep(300)
                self._flush_retry_queue()

        threading.Thread(target=_retry_loop, daemon=True, name="pref-retry").start()

    def _flush_retry_queue(self):
        """消费重试队列（线程安全）

        B2 fix: 原先使用 list(queue) + queue.clear() 组合不是原子操作，
        在快照和清空之间，pref-update 线程可能追加新条目，会被 clear() 误删。
        改用 popleft() 逐个弹出，CPython GIL 保证单次 popleft 是原子的。
        新追加的条目要么在本次 flush 中被取出，要么留在队列等待下次 flush，
        不会丢失。
        """
        if not self.long_term_memory:
            return

        snapshot = []
        while True:
            try:
                snapshot.append(self._pref_retry_queue.popleft())
            except IndexError:
                break

        for user_id, user_input, response, attempt in snapshot:
            if attempt >= 3:
                self.logger.warning(
                    f"[PostPipeline] 偏好提取超过3次失败，放弃: user={user_id}"
                )
                continue
            try:
                prefs = self._extract_user_preferences(user_input, response)
                if prefs:
                    self.long_term_memory.update_user_preferences(user_id, prefs)
            except Exception:
                self._pref_retry_queue.append((user_id, user_input, response, attempt + 1))

    def _extract_user_preferences(self, user_input: str, response: str) -> dict:
        """使用 LLM 提取用户偏好（逻辑同原 _extract_user_preferences）"""
        prompt = f"""从以下对话中提取用户的偏好信息（仅提取本轮明确提到的内容）。

用户: {user_input}
助手: {response[:400]}

请以JSON格式返回，只包含本轮对话中**明确出现**的偏好字段：
{{
  "budget": "预算限制（如'2万以内'，未提到则不填此字段）",
  "interests": "兴趣/偏好品类（如'电子产品'，未提到则不填此字段）",
  "preferred_brands": "偏好品牌（未提到则不填此字段）",
  "usage_scenario": "使用场景（如'办公'、'游戏'，未提到则不填此字段）"
}}

规则：
- 没有明确提到的字段**不要**包含在JSON中
- 如果本轮没有任何偏好信息，返回 {{}}
- 只返回JSON，不要其他内容"""

        try:
            resp = self.llm_client.generate(prompt=prompt, temperature=0.1).strip()
            code_block = _re.search(r'```(?:json)?\s*\n?(.*?)\n?```', resp, _re.DOTALL)
            if code_block:
                resp = code_block.group(1).strip()
            brace_start = resp.find('{')
            brace_end = resp.rfind('}')
            if brace_start >= 0 and brace_end > brace_start:
                resp = resp[brace_start:brace_end + 1]
            prefs = _json.loads(resp)
            # 过滤掉空值
            return {k: v for k, v in prefs.items() if v and str(v).strip()}
        except Exception as e:
            self.logger.debug(f"偏好提取失败: {e}")
            return {}
