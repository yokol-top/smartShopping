import logging
import threading
import time
from contextlib import contextmanager
from typing import List, Dict, Optional

from openai import OpenAI

from observability import get_tracer


@contextmanager
def _noop_context():
    """未启用追踪时的空上下文管理器"""
    yield


class LLMClient:
    """
    统一的LLM客户端，使用OpenAI API接口
    支持OpenAI、Azure OpenAI、以及兼容OpenAI API的服务（如Ollama、vLLM等）
    """
    
    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 120,
        system_prompt: Optional[str] = None,
        routing: dict = None,
        logger: logging.Logger = None
    ):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥（Ollama使用"EMPTY"或任意字符串）
            base_url: API基础URL
                - OpenAI: https://api.openai.com/v1
                - Ollama: http://localhost:11434/v1
                - Azure OpenAI: https://your-resource.openai.azure.com
                - vLLM: http://localhost:8000/v1
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间（秒）
            system_prompt: 全局系统提示词（自动添加到所有请求）
            logger: 日志记录器
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.global_system_prompt = system_prompt
        self.logger = logger or logging.getLogger(__name__)
        self._routing: dict = routing or {}

        # Token 使用量累计（线程安全）
        self._usage_lock = threading.Lock()
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_calls: int = 0
        
        self.logger.info(f"初始化LLM客户端")
        self.logger.info(f"  基础URL: {base_url}")
        self.logger.info(f"  模型: {model}")
        self.logger.info(f"  温度: {temperature}")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        self.logger.info("LLM客户端初始化完成")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        对话接口
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大token数（覆盖默认值）
            stream: 是否使用流式输出
            **kwargs: 其他参数
            
        Returns:
            模型响应文本
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # 按 task_type 路由模型（不修改 self.model，线程安全）
        effective_model = self._routing.get(task_type) if task_type else None
        effective_model = effective_model or self.model

        self.logger.debug(f"调用LLM - 消息数: {len(messages)}, 温度: {temp}，输入：{messages[1].get('content', '')}")

        tracer = get_tracer()
        with (tracer.start_span("gen_ai.chat", {
            "gen_ai.system": "openai",
            "gen_ai.request.model": effective_model,
            "gen_ai.request.temperature": temp,
            "gen_ai.request.max_tokens": max_tok,
        }) if tracer and tracer.enabled else _noop_context()):
            start_time = time.time()
            try:
                if stream:
                    result = self._chat_stream(messages, temp, max_tok, **kwargs)
                else:
                    response = self.client.chat.completions.create(
                        model=effective_model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        **kwargs
                    )

                    result = response.choices[0].message.content
                    self.logger.debug(f"LLM响应长度: {len(result)}，{result}")

                    # 记录 GenAI 标准属性
                    usage = getattr(response, 'usage', None)
                    if tracer and tracer.enabled:
                        prompt_text = messages[-1].get('content', '') if messages else ''
                        finish_reason = response.choices[0].finish_reason if response.choices else None
                        tracer.record_llm_call(
                            model=effective_model,
                            prompt=prompt_text,
                            response=result,
                            temperature=temp,
                            max_tokens=max_tok,
                            input_tokens=getattr(usage, 'prompt_tokens', None) if usage else None,
                            output_tokens=getattr(usage, 'completion_tokens', None) if usage else None,
                            finish_reason=finish_reason,
                            duration_ms=(time.time() - start_time) * 1000,
                        )
                        tracer.set_span_ok()

                    # 累计 token 使用量
                    with self._usage_lock:
                        if usage:
                            self._total_input_tokens += getattr(usage, 'prompt_tokens', 0) or 0
                            self._total_output_tokens += getattr(usage, 'completion_tokens', 0) or 0
                        self._total_calls += 1

                return result
            except Exception as e:
                if tracer and tracer.enabled:
                    tracer.record_exception(e)
                self.logger.error(f"LLM调用失败: {e}")
                raise
    
    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """流式对话（内部使用）"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end='', flush=True)
            
            print()  # 换行
            return full_response
        except Exception as e:
            self.logger.error(f"流式LLM调用失败: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        简单的生成接口（兼容Ollama风格）
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            system_prompt: 系统提示词（会覆盖全局系统提示词）
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        messages = []
        
        # 优先使用传入的system_prompt，否则使用全局系统提示词
        effective_system_prompt = system_prompt if system_prompt is not None else self.global_system_prompt
        
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens, task_type=task_type, **kwargs)

    def chat_with_context(
        self,
        user_message: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        带上下文的对话
        
        Args:
            user_message: 用户消息
            context: 上下文信息
            system_prompt: 系统提示词（会覆盖全局系统提示词）
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            模型响应
        """
        messages = []
        
        # 优先使用传入的system_prompt，否则使用全局系统提示词
        effective_system_prompt = system_prompt if system_prompt is not None else self.global_system_prompt
        
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        
        # 将上下文和用户消息组合
        combined_message = f"{context}\n\n{user_message}"
        messages.append({"role": "user", "content": combined_message})
        
        return self.chat(messages, temperature, **kwargs)
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 提示词列表
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            生成结果列表
        """
        self.logger.info(f"批量生成 - 数量: {len(prompts)}")
        
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt, temperature, max_tokens, **kwargs)
                results.append(result)
                self.logger.debug(f"批量生成进度: {i+1}/{len(prompts)}")
            except Exception as e:
                self.logger.error(f"批量生成失败 (项 {i}): {e}")
                results.append("")
        
        return results
    
    def get_usage_snapshot(self) -> dict:
        """获取当前 token 使用量快照（用于计算任务级增量）"""
        with self._usage_lock:
            return {
                'input_tokens': self._total_input_tokens,
                'output_tokens': self._total_output_tokens,
                'calls': self._total_calls,
            }

    def compute_usage_delta(self, snapshot: dict) -> dict:
        """计算自快照以来的 token 增量"""
        current = self.get_usage_snapshot()
        return {
            'input_tokens': current['input_tokens'] - snapshot.get('input_tokens', 0),
            'output_tokens': current['output_tokens'] - snapshot.get('output_tokens', 0),
            'calls': current['calls'] - snapshot.get('calls', 0),
        }

    def get_total_usage(self) -> dict:
        """获取全会话累计 token 使用量"""
        return self.get_usage_snapshot()

    def update_config(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        更新配置
        
        Args:
            model: 新的模型名称
            temperature: 新的温度参数
            max_tokens: 新的最大token数
        """
        if model is not None:
            self.model = model
            self.logger.info(f"更新模型: {model}")
        
        if temperature is not None:
            self.temperature = temperature
            self.logger.info(f"更新温度: {temperature}")
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.logger.info(f"更新最大token数: {max_tokens}")
