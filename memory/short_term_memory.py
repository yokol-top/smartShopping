from typing import List, Dict, Any
from collections import deque
import logging


class ShortTermMemory:
    """短期记忆：保存最近的对话历史"""
    
    def __init__(self, max_messages: int = 10, logger: logging.Logger = None):
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)
        self.logger = logger or logging.getLogger(__name__)
        
        # 历史对话摘要（保存被移除消息的总结）
        self.historical_summary = ""
        # 待总结的消息计数
        self.messages_since_last_summary = 0
        # 总结触发阈值（当消息队列满后，每N条新消息触发一次总结）
        self.summary_trigger_count = 5
        # 历史摘要最大字符数（防止无限增长）
        self.max_summary_length = 500
        # 摘要累积次数（用于决定何时压缩）
        self.summary_accumulation_count = 0
        self.max_accumulation_before_compress = 3
        
        self.logger.info(f"初始化短期记忆，最大消息数: {max_messages}")
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        添加消息到短期记忆
        
        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 元数据
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.logger.debug(f"添加消息到短期记忆 - 角色: {role}, 内容长度: {len(content)}")
    
    def get_messages(self, last_n: int = None) -> List[Dict[str, Any]]:
        """
        获取消息历史
        
        Args:
            last_n: 获取最后n条消息，None表示全部
            
        Returns:
            消息列表
        """
        if last_n is None:
            result = list(self.messages)
        else:
            result = list(self.messages)[-last_n:]
        
        self.logger.debug(f"获取短期记忆消息，数量: {len(result)}")
        return result
    
    def get_context_string(self, last_n: int = None, include_summary: bool = True) -> str:
        """
        获取格式化的上下文字符串
        
        Args:
            last_n: 获取最后n条消息
            include_summary: 是否包含历史摘要
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 如果存在历史摘要，添加到上下文开头
        if include_summary and self.historical_summary:
            context_parts.append(f"[历史对话摘要]\n{self.historical_summary}\n")
            context_parts.append("[最近对话]")
        
        messages = self.get_messages(last_n)
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            context_parts.append(f"{role.upper()}: {content}")
        
        context = "\n".join(context_parts)
        self.logger.debug(f"生成上下文字符串，长度: {len(context)}, 包含历史摘要: {include_summary and bool(self.historical_summary)}")
        return context
    
    def get_historical_summary(self) -> str:
        """获取历史摘要"""
        return self.historical_summary
    
    def update_historical_summary(self, new_summary: str):
        """
        更新历史对话摘要（智能管理摘要长度）
        
        Args:
            new_summary: 新的摘要内容（通常是LLM生成的）
        """
        if self.historical_summary:
            # 如果已有摘要，追加新摘要
            self.historical_summary = f"{self.historical_summary}\n\n{new_summary}"
            self.summary_accumulation_count += 1
            
            # 如果摘要过长，需要压缩
            if len(self.historical_summary) > self.max_summary_length:
                self.logger.info(f"历史摘要超长({len(self.historical_summary)}字符)，标记为需要压缩")
                # 标记需要压缩（由agent调用compress_summary）
                self._needs_compression = True
        else:
            self.historical_summary = new_summary
            self.summary_accumulation_count = 1
        
        self.messages_since_last_summary = 0
        self.logger.info(f"更新历史摘要，摘要长度: {len(self.historical_summary)}, 累积次数: {self.summary_accumulation_count}")
    
    def needs_compression(self) -> bool:
        """检查是否需要压缩历史摘要"""
        return getattr(self, '_needs_compression', False)
    
    def compress_summary(self, compressed_summary: str):
        """
        用压缩后的摘要替换当前历史摘要
        
        Args:
            compressed_summary: 压缩后的摘要
        """
        self.historical_summary = compressed_summary
        self.summary_accumulation_count = 1
        self._needs_compression = False
        self.logger.info(f"历史摘要已压缩，新长度: {len(self.historical_summary)}")
    
    def should_trigger_summary(self) -> bool:
        """
        判断是否应该触发总结
        
        Returns:
            是否需要总结
        """
        # 当消息队列已满，且新增消息数达到阈值时触发
        is_full = len(self.messages) >= self.max_messages
        has_enough_new = self.messages_since_last_summary >= self.summary_trigger_count
        
        return is_full and has_enough_new
    
    def increment_message_count(self):
        """增加待总结消息计数"""
        self.messages_since_last_summary += 1
    
    def clear(self):
        """清空短期记忆"""
        self.messages.clear()
        self.historical_summary = ""
        self.messages_since_last_summary = 0
        self.summary_accumulation_count = 0
        self._needs_compression = False
        self.logger.info("清空短期记忆")
    
    def __len__(self):
        return len(self.messages)
