"""
Agent 可观测性模块 - 基于 OpenTelemetry + GenAI Semantic Conventions

提供:
1. Tracer 初始化与管理
2. 装饰器 @trace_span() 用于自动创建 Span
3. 工具函数用于记录 LLM 调用、Agent 步骤等
"""
import functools
import time
from typing import Optional, Dict, Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode


# 全局单例
_agent_tracer: Optional['AgentTracer'] = None


def get_tracer() -> Optional['AgentTracer']:
    """获取全局 AgentTracer 实例"""
    return _agent_tracer


class AgentTracer:
    """
    Agent 可观测性追踪器

    封装 OpenTelemetry TracerProvider 的初始化和常用操作。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化追踪器

        Args:
            config: observability 配置节，来自 settings.yaml
        """
        global _agent_tracer

        self.enabled = config.get('enabled', False)
        self.trace_llm_content = config.get('trace_llm_content', True)
        self.max_attr_length = config.get('max_attribute_length', 500)

        if not self.enabled:
            self.tracer = None
            _agent_tracer = self
            return

        service_name = config.get('service_name', 'smart-agent')
        exporter_type = config.get('exporter', 'console')
        otlp_endpoint = config.get('otlp_endpoint', 'http://localhost:4317')

        # 创建 Resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
        })

        # 创建 TracerProvider
        provider = TracerProvider(resource=resource)

        # 根据配置添加 Exporter
        if exporter_type in ("console", "both"):
            provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )

        if exporter_type in ("otlp", "both"):
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
            except ImportError:
                print("[Observability] OTLP exporter not installed, falling back to console only")
                if exporter_type == "otlp":
                    provider.add_span_processor(
                        SimpleSpanProcessor(ConsoleSpanExporter())
                    )

        # 设置为全局 TracerProvider
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.tracer = trace.get_tracer("smart-agent", "1.0.0")

        _agent_tracer = self

    def start_span(self, name: str, attributes: Dict[str, Any] = None):
        """
        创建并返回一个 Span 上下文管理器

        Args:
            name: Span 名称
            attributes: 初始属性

        Returns:
            Span 上下文管理器，如果未启用则返回 NoOpSpan
        """
        if not self.enabled or not self.tracer:
            return _NoOpSpanContext()

        return self.tracer.start_as_current_span(
            name,
            attributes=self._sanitize_attributes(attributes) if attributes else None,
        )

    def set_span_attributes(self, attributes: Dict[str, Any]):
        """在当前活跃 Span 上设置属性"""
        if not self.enabled:
            return
        span = trace.get_current_span()
        if span and span.is_recording():
            for k, v in self._sanitize_attributes(attributes).items():
                span.set_attribute(k, v)

    def record_exception(self, exception: Exception):
        """在当前 Span 上记录异常"""
        if not self.enabled:
            return
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_status(StatusCode.ERROR, str(exception))
            span.record_exception(exception)

    def set_span_ok(self):
        """标记当前 Span 为成功"""
        if not self.enabled:
            return
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_status(StatusCode.OK)

    def record_llm_call(
        self,
        model: str,
        prompt: str = "",
        response: str = "",
        temperature: float = None,
        max_tokens: int = None,
        input_tokens: int = None,
        output_tokens: int = None,
        finish_reason: str = None,
        duration_ms: float = None,
    ):
        """
        在当前 Span 上记录 LLM 调用的 GenAI 标准属性

        遵循 OpenTelemetry GenAI Semantic Conventions
        """
        if not self.enabled:
            return

        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.request.model": model,
        }

        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        if input_tokens is not None:
            attrs["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            attrs["gen_ai.usage.output_tokens"] = output_tokens
        if finish_reason:
            attrs["gen_ai.response.finish_reason"] = finish_reason
        if duration_ms is not None:
            attrs["llm.duration_ms"] = duration_ms

        # 记录 prompt/response 内容（可通过配置关闭）
        if self.trace_llm_content:
            attrs["llm.prompt_preview"] = self._truncate(prompt)
            attrs["llm.response_preview"] = self._truncate(response)

        self.set_span_attributes(attrs)

    def shutdown(self):
        """关闭 TracerProvider，确保所有 Span 被导出"""
        if self.enabled and hasattr(self, 'provider'):
            self.provider.shutdown()

    # ---- 内部方法 ----

    def _truncate(self, text: str) -> str:
        """截断文本到最大属性长度"""
        if not text:
            return ""
        if len(text) <= self.max_attr_length:
            return text
        return text[:self.max_attr_length] + "...(truncated)"

    def _sanitize_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """清理属性值，确保符合 OpenTelemetry 要求"""
        sanitized = {}
        for k, v in attributes.items():
            if v is None:
                continue
            if isinstance(v, str):
                sanitized[k] = self._truncate(v)
            elif isinstance(v, (int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, (list, tuple)):
                # OpenTelemetry 支持同类型列表
                sanitized[k] = [str(item) for item in v]
            else:
                sanitized[k] = str(v)[:self.max_attr_length]
        return sanitized


class _NoOpSpanContext:
    """当 observability 未启用时，提供空操作的上下文管理器"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def is_recording(self):
        return False


def trace_span(name: str = None, attributes: Dict[str, Any] = None):
    """
    装饰器：自动为函数创建 Span

    用法:
        @trace_span("intent.recognize")
        def recognize(self, query, context):
            ...

        @trace_span()  # 自动使用函数名
        def some_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer is None or not tracer.enabled:
                return func(*args, **kwargs)

            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with tracer.start_span(span_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    if span.is_recording():
                        span.set_status(StatusCode.OK)
                    return result
                except Exception as e:
                    if span.is_recording():
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                    raise
        return wrapper
    return decorator
