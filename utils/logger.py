import contextvars
import logging
import os
from typing import Optional

import colorlog

# ── 请求级 trace ID（ContextVar，协程安全）────────────────────────────
_current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    'trace_id', default='-'
)


def set_trace_id(trace_id: str) -> None:
    """设置当前请求的 trace ID，同一请求内所有日志将携带该 ID。"""
    _current_trace_id.set(trace_id)


def get_trace_id() -> str:
    """获取当前请求的 trace ID，无活跃请求时返回 '-'。"""
    return _current_trace_id.get()


class _TraceIdFilter(logging.Filter):
    """将当前 trace ID 注入每条 LogRecord，供 format 字符串引用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _current_trace_id.get()
        return True


# ── 统一 format 字符串 ──────────────────────────────────────────────
_FMT = (
    '%(asctime)s - %(name)s - %(levelname)s'
    ' - [%(filename)s:%(lineno)d] [%(trace_id)s] %(message)s'
)
_DATE_FMT = '%Y-%m-%d %H:%M:%S'


def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        console: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # trace ID filter（挂在 logger 上，对所有 handler 生效）
    logger.addFilter(_TraceIdFilter())

    file_formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s' + _FMT,
        datefmt=_DATE_FMT,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
