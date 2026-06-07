"""
熔断器 (Circuit Breaker)

连续失败达到阈值后进入OPEN状态，暂停委派。
经过冷却期后进入HALF_OPEN状态，允许少量试探。
试探成功则恢复CLOSED状态。

参考企业级方案：Netflix Hystrix / resilience4j
"""

import logging
import time


class CircuitBreaker:
    """熔断器

    连续失败达到阈值后进入OPEN状态，暂停委派。
    经过冷却期后进入HALF_OPEN状态，允许少量试探。
    试探成功则恢复CLOSED状态。

    参考企业级方案：Netflix Hystrix / resilience4j
    """

    CLOSED = "closed"         # 正常工作
    OPEN = "open"             # 熔断中（拒绝委派）
    HALF_OPEN = "half_open"   # 试探中

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        logger: logging.Logger = None,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger or logging.getLogger(__name__)

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count_in_half_open = 0

    @property
    def state(self) -> str:
        # 自动从OPEN转HALF_OPEN（冷却期过后）
        if self._state == self.OPEN:
            if time.time() - self._last_failure_time > self.cooldown_seconds:
                self._state = self.HALF_OPEN
                self._success_count_in_half_open = 0
                self.logger.info("[CircuitBreaker] OPEN → HALF_OPEN（冷却期结束）")
        return self._state

    def allow_request(self) -> bool:
        """是否允许委派"""
        s = self.state
        if s == self.CLOSED:
            return True
        if s == self.HALF_OPEN:
            return True  # 允许试探
        return False  # OPEN状态拒绝

    def record_success(self):
        """记录成功"""
        if self._state == self.HALF_OPEN:
            self._success_count_in_half_open += 1
            if self._success_count_in_half_open >= 2:
                self._state = self.CLOSED
                self._failure_count = 0
                self.logger.info("[CircuitBreaker] HALF_OPEN → CLOSED（试探成功）")
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """记录失败"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            self.logger.warning("[CircuitBreaker] HALF_OPEN → OPEN（试探失败）")
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            self.logger.warning(
                f"[CircuitBreaker] CLOSED → OPEN（连续失败{self._failure_count}次）"
            )
