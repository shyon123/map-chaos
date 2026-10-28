"""Fault protocol, context/outcome types, and the mode registry."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from mcp_chaos.schemas import FaultApplied, FaultMode, FaultSpec, SchemaDriftParams


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


@dataclass(frozen=True)
class CatalogContext:
    spec: FaultSpec
    rng: random.Random
    tool_call_counts: Mapping[str, int]


@dataclass(frozen=True)
class CatalogOutcome:
    """``tools`` replaces the input array (a vanished tool is simply absent)."""

    tools: list[dict[str, Any]]
    notify_list_changed: bool = False
    vanished: frozenset[str] = frozenset()
    applied: list[FaultApplied] = field(default_factory=lambda: [])


class CatalogFault(Protocol):
    def apply_list(self, ctx: CatalogContext, tools: list[dict[str, Any]]) -> CatalogOutcome: ...


CATALOG_REGISTRY: dict[FaultMode, CatalogFault] = {}


def is_catalog_spec(spec: FaultSpec) -> bool:
    """tool_vanish and schema_drift(layer=catalog) act on tools/list; the rest are per-call."""
    if spec.mode == FaultMode.TOOL_VANISH:
        return True
    p = spec.params
    return isinstance(p, SchemaDriftParams) and p.layer == "catalog"
