"""Fault protocol, context/outcome types, and the mode registry."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from mcp_chaos.schemas import FaultApplied, FaultMode, FaultSpec


@dataclass(frozen=True)
class FaultContext:
    spec: FaultSpec
    rng: random.Random
    tool: str
    call_index: int
    arguments: dict[str, Any] | None
    # earlier unfaulted responses to the same tool + arguments, oldest first (for stale_data)
    history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FaultOutcome:
    """``response_obj`` is the replacement JSON-RPC message (None = keep the original)."""

    response_obj: dict[str, Any] | None = None
    # raw replacement line (frame-level corruption); takes precedence over response_obj
    response_bytes: bytes | None = None
    delay_s: float = 0.0
    hang: bool = False
    crash: bool = False
    applied: FaultApplied | None = None


class Fault(Protocol):
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome: ...


REGISTRY: dict[FaultMode, Fault] = {}
