"""Per-run and global spend tracking, and SSE/body usage parsing."""

from __future__ import annotations

import json
from typing import Any, cast

from mcp_chaos.schemas.results import Usage


class TokenCounts:
    __slots__ = ("cache_read", "cache_write", "input", "output")

    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.cache_write = 0
        self.cache_read = 0

    def absorb(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        u = cast("dict[str, Any]", usage)
        for attr, field in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("cache_write", "cache_creation_input_tokens"),
            ("cache_read", "cache_read_input_tokens"),
        ):
            v = u.get(field)
            if isinstance(v, int) and not isinstance(v, bool):
                setattr(self, attr, v)


def usage_from_body(body: bytes) -> TokenCounts:
    t = TokenCounts()
    try:
        obj: Any = json.loads(body)
    except ValueError:
        return t
    if isinstance(obj, dict):
        t.absorb(cast("dict[str, Any]", obj).get("usage"))
    return t


def usage_from_sse(chunks: list[bytes]) -> TokenCounts:
    """Fold ``message_start`` then ``message_delta`` usage (later values override)."""
    t = TokenCounts()
    text = b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
    for event in text.split("\n\n"):
        for line in event.split("\n"):
            if not line.startswith("data:"):
                continue
            try:
                d: Any = json.loads(line[5:].strip())
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            dd = cast("dict[str, Any]", d)
            if dd.get("type") == "message_start":
                msg = dd.get("message")
                if isinstance(msg, dict):
                    t.absorb(cast("dict[str, Any]", msg).get("usage"))
            elif dd.get("type") == "message_delta":
                t.absorb(dd.get("usage"))
    return t


class BudgetTracker:
    def __init__(self, global_cap_usd: float) -> None:
        self.global_cap = global_cap_usd
        self.global_spend = 0.0
        self._caps: dict[str, float] = {}
        self._spend: dict[str, float] = {}
        self._usage: dict[str, Usage] = {}
        self._aborted: set[str] = set()

    def register(self, run_id: str, cap_usd: float) -> None:
        self._caps[run_id] = cap_usd
        self._spend.setdefault(run_id, 0.0)
        self._usage.setdefault(run_id, Usage())

    def known(self, run_id: str) -> bool:
        return run_id in self._caps

    def exceeded(self, run_id: str) -> bool:
        return (
            self._spend.get(run_id, 0.0) >= self._caps.get(run_id, 0.0)
            or self.global_spend >= self.global_cap
        )

    def mark_aborted(self, run_id: str) -> None:
        self._aborted.add(run_id)

    def is_aborted(self, run_id: str) -> bool:
        return run_id in self._aborted

    def record(self, run_id: str, t: TokenCounts, cost: float) -> None:
        self._spend[run_id] = self._spend.get(run_id, 0.0) + cost
        self.global_spend += cost
        u = self._usage.get(run_id, Usage())
        self._usage[run_id] = u.model_copy(
            update={
                "input_tokens": u.input_tokens + t.input,
                "output_tokens": u.output_tokens + t.output,
                "cache_read_tokens": u.cache_read_tokens + t.cache_read,
                "cache_write_tokens": u.cache_write_tokens + t.cache_write,
                "cost_usd": u.cost_usd + cost,
            }
        )

    def record_cache_hit(self, run_id: str) -> None:
        u = self._usage.get(run_id, Usage())
        self._usage[run_id] = u.model_copy(update={"cache_hits": u.cache_hits + 1})

    def get_usage(self, run_id: str) -> Usage:
        return self._usage.get(run_id, Usage())
