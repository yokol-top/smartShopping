"""
输入层 (Input Gate)

主要职责：
1. 规则拦截恶意输入（注入攻击、越权指令等）
2. 输入长度与格式校验
3. 敏感词过滤
4. 输入日志记录
"""
import re
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from observability import get_tracer


@dataclass
class InputValidationResult:
    """输入验证结果"""
    is_valid: bool
    sanitized_input: str
    blocked: bool = False
    block_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low / medium / high


class InputValidator:
    """
    输入验证器

    职责：
    1. 恶意输入检测（prompt注入、越权指令）
    2. 输入长度与格式校验
    3. 敏感词过滤与脱敏
    4. 输入审计日志
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        input_gate_config = config.get('input_gate', {})
        self.enabled = input_gate_config.get('enabled', True)
        self.max_input_length = input_gate_config.get('max_input_length', 5000)
        self.min_input_length = input_gate_config.get('min_input_length', 1)

        # 恶意模式（prompt注入攻击）
        self.injection_patterns = input_gate_config.get('injection_patterns', []) or [
            r'忽略(之前|上面|以上)(的|所有)(指令|提示|规则|指示)',
            r'ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?)',
            r'you\s+are\s+now\s+(?:a|an)\s+\w+',
            r'system\s*:\s*',
            r'<\|.*?\|>',
            r'\[INST\].*?\[/INST\]',
            r'(jailbreak|越狱|DAN模式|开发者模式)',
            r'假装你(是|没有|不受|不需要)',
            r'pretend\s+(you|to)\s+(are|be|have|don)',
        ]

        # 敏感词列表
        self.sensitive_words = input_gate_config.get('sensitive_words', []) or [
            '身份证号', '银行卡号', '密码', '信用卡',
            'password', 'secret_key', 'api_key', 'token',
            'ssn', 'credit_card',
        ]

        # 敏感数据正则模式（用于脱敏）
        self.sensitive_data_patterns = {
            'phone': r'1[3-9]\d{9}',
            'id_card': r'\d{17}[\dXx]',
            'email': r'[\w.+-]+@[\w-]+\.[\w.-]+',
            'bank_card': r'\d{16,19}',
        }

        self.logger.info("InputValidator 初始化完成")

    def validate(self, user_input: str) -> InputValidationResult:
        """
        验证用户输入（主入口）

        Args:
            user_input: 原始用户输入

        Returns:
            InputValidationResult 验证结果
        """
        tracer = get_tracer()

        with tracer.start_span("input_gate.validate", {
            "input.length": len(user_input),
        }) if tracer else _noop():
            if not self.enabled:
                return InputValidationResult(
                    is_valid=True,
                    sanitized_input=user_input,
                )

            start_time = time.time()
            warnings = []
            risk_level = "low"

            self.logger.info(f"[输入层] 验证输入，长度: {len(user_input)}")

            # 1. 空输入检查
            stripped = user_input.strip()
            if len(stripped) < self.min_input_length:
                self.logger.warning("[输入层] 输入为空或过短")
                return InputValidationResult(
                    is_valid=False,
                    sanitized_input="",
                    blocked=True,
                    block_reason="输入不能为空",
                )

            # 2. 长度检查
            if len(stripped) > self.max_input_length:
                self.logger.warning(f"[输入层] 输入过长: {len(stripped)}")
                stripped = stripped[:self.max_input_length]
                warnings.append(f"输入已被截断至{self.max_input_length}字符")
                risk_level = "medium"

            # 3. Prompt注入检测
            injection_result = self._detect_injection(stripped)
            if injection_result:
                self.logger.warning(f"[输入层] 检测到潜在注入攻击: {injection_result}")
                return InputValidationResult(
                    is_valid=False,
                    sanitized_input=stripped,
                    blocked=True,
                    block_reason=f"检测到潜在的恶意输入: {injection_result}",
                    risk_level="high",
                )

            # 4. 敏感词检测（不阻断，但记录警告）
            found_sensitive = self._detect_sensitive_words(stripped)
            if found_sensitive:
                warnings.append(f"输入包含敏感信息类型: {', '.join(found_sensitive)}")
                risk_level = max(risk_level, "medium", key=lambda x: {"low": 0, "medium": 1, "high": 2}[x])
                self.logger.info(f"[输入层] 检测到敏感词: {found_sensitive}")

            # 5. 敏感数据脱敏（用于日志记录）
            sanitized_for_log = self._mask_sensitive_data(stripped)

            elapsed = (time.time() - start_time) * 1000
            self.logger.info(
                f"[输入层] 验证通过 | 风险: {risk_level} | 耗时: {elapsed:.1f}ms | "
                f"脱敏日志: {sanitized_for_log[:100]}..."
            )

            if tracer:
                tracer.set_span_attributes({
                    "input.risk_level": risk_level,
                    "input.warnings_count": len(warnings),
                    "input.validation_ms": elapsed,
                })

            return InputValidationResult(
                is_valid=True,
                sanitized_input=stripped,
                warnings=warnings,
                risk_level=risk_level,
            )

    def _detect_injection(self, text: str) -> Optional[str]:
        """检测prompt注入攻击"""
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            try:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return f"匹配到注入模式: {pattern[:30]}..."
            except re.error:
                continue
        return None

    def _detect_sensitive_words(self, text: str) -> List[str]:
        """检测敏感词"""
        found = []
        text_lower = text.lower()
        for word in self.sensitive_words:
            if word.lower() in text_lower:
                found.append(word)
        return found

    def _mask_sensitive_data(self, text: str) -> str:
        """脱敏敏感数据（用于日志安全记录）"""
        masked = text
        for data_type, pattern in self.sensitive_data_patterns.items():
            masked = re.sub(pattern, f'[{data_type.upper()}_MASKED]', masked)
        return masked


class _noop:
    """空上下文管理器"""
    def __enter__(self): return self
    def __exit__(self, *a): pass
