"""Retry policy (bounded exponential backoff, seeded jitter) and per-tool circuit breaker.

Sans-IO: time is passed in, randomness comes from an injected seeded ``random.Random``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

FailureKind = Literal["deadline", "rpc_error", "unparseable", "validation"]
BreakerState = Literal["closed", "open", "half_open"]

# JSON-RPC errors that will not change on retry: policy denial, invalid request, unknown tool.
PERMANENT_RPC_CODES = frozenset({-32001, -32600, -32602})


def is_retryable_rpc_code(code: object) -> bool:
    return not (isinstance(code, int) and code in PERMANENT_RPC_CODES)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    base_s: float
    max_s: float
    seed: int

    def rng(self) -> random.Random:
        return random.Random(self.seed)  # noqa: S311 - seeded, not crypto

    def backoff_s(self, attempt: int, rng: random.Random) -> float:
        """Delay before retry ``attempt`` (0-based): capped exponential, jitter in [50%, 100%]."""
        cap = min(self.max_s, self.base_s * (2**attempt))
        return cap * (0.5 + 0.5 * rng.random())

    def may_retry(self, retries_done: int) -> bool:
        return retries_done < self.max_retries


class CircuitBreaker:
    """Closed -> open after ``threshold`` consecutive failures; half-open after ``cooldown_s``.

    Half-open admits exactly one probe; its outcome closes or re-opens the breaker.
    """

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._fails = 0
        self._opened_at: float | None = None
        self._probing = False

    def state(self, now: float) -> BreakerState:
        if self._opened_at is None:
            return "closed"
        if now - self._opened_at < self.cooldown_s:
            return "open"
        return "half_open"

    def allow(self, now: float) -> bool:
        s = self.state(now)
        if s == "closed":
            return True
        if s == "open" or self._probing:
            return False
        self._probing = True
        return True

    def record_success(self) -> None:
        self._fails = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self, now: float) -> None:
        if self._probing:
            self._probing = False
            self._opened_at = now
            return
        self._fails += 1
        if self._fails >= self.threshold:
            self._opened_at = now


class BreakerBank:
    """One breaker per tool."""

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._by_tool: dict[str, CircuitBreaker] = {}

    def get(self, tool: str) -> CircuitBreaker:
        b = self._by_tool.get(tool)
        if b is None:
            b = self._by_tool[tool] = CircuitBreaker(self._threshold, self._cooldown_s)
        return b
