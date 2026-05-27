"""
输出审查模块 (Output Guard)

职责：
1. 输出内容审查（过滤不当内容）
2. 敏感数据脱敏（防止泄露内部信息）
3. 输出格式规范化
"""
import re
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class OutputReviewResult:
    """输出审查结果"""
    is_safe: bool
    cleaned_output: str
    masked_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OutputGuard:
    """
    输出审查器

    在Agent返回结果给用户前进行安全审查和数据脱敏。
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        guard_config = config.get('security', {}).get('output_guard', {})
        self.enabled = guard_config.get('enabled', True)

        # 需要脱敏的数据模式
        self.masking_patterns = {
            'phone': (r'1[3-9]\d{9}', lambda m: m.group()[:3] + '****' + m.group()[-4:]),
            'id_card': (r'\d{17}[\dXx]', lambda m: m.group()[:6] + '********' + m.group()[-4:]),
            'email': (r'([\w.+-]+)@([\w-]+\.[\w.-]+)', r'\1[at]\2'),
            'bank_card': (r'\d{16,19}', lambda m: m.group()[:4] + ' **** **** ' + m.group()[-4:]),
            'api_key': (r'(api[_-]?key|token|secret)[=:]\s*["\']?[\w-]{16,}', '[API_KEY_MASKED]'),
            'ip_address': (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_MASKED]'),
        }

        # 禁止输出的关键词
        self.blocked_output_patterns = guard_config.get('blocked_patterns', []) or [
            r'system\s*prompt',
            r'内部(系统|密钥|密码|凭证)',
            r'数据库(密码|连接串|链接)',
        ]

        self.logger.info("OutputGuard 初始化完成")

    def review(self, output: str) -> OutputReviewResult:
        """
        审查输出内容

        Args:
            output: Agent即将返回给用户的输出

        Returns:
            OutputReviewResult 审查结果
        """
        if not self.enabled:
            return OutputReviewResult(is_safe=True, cleaned_output=output)

        warnings = []
        masked_fields = []
        cleaned = output

        # 1. 敏感数据脱敏
        for field_name, pattern_info in self.masking_patterns.items():
            pattern, replacement = pattern_info
            if re.search(pattern, cleaned, re.IGNORECASE):
                if callable(replacement):
                    cleaned = re.sub(pattern, replacement, cleaned)
                else:
                    cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
                masked_fields.append(field_name)
                self.logger.info(f"[输出审查] 脱敏字段: {field_name}")

        # 2. 检查禁止输出的内容
        for blocked_pattern in self.blocked_output_patterns:
            if re.search(blocked_pattern, cleaned, re.IGNORECASE):
                warnings.append(f"输出包含敏感模式: {blocked_pattern[:30]}")
                cleaned = re.sub(blocked_pattern, '[内容已过滤]', cleaned, flags=re.IGNORECASE)

        is_safe = len(warnings) == 0

        if masked_fields:
            self.logger.info(f"[输出审查] 已脱敏 {len(masked_fields)} 类敏感数据")
        if warnings:
            self.logger.warning(f"[输出审查] 发现 {len(warnings)} 个安全警告")

        return OutputReviewResult(
            is_safe=is_safe,
            cleaned_output=cleaned,
            masked_fields=masked_fields,
            warnings=warnings,
        )
