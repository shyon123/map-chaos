"""Per-run and global spend tracking, and SSE/body usage parsing."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
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


@dataclass(eq=False)
class Reservation:
    """Worst-case spend held for one in-flight upstream request until ``settle``."""

    run_id: str
    worst_case_usd: float
    settled: bool = False


class BudgetTracker:
    """Spend tracking with reservations.

    Invariant: for every run and globally, ``spent + reserved <= cap`` holds after every
    successful ``reserve`` (checked and updated atomically under a lock), so concurrent requests
    cannot jointly exceed a cap on their *worst-case* cost. Realized spend can exceed a cap only
    by one request's (actual - worst-case) error, i.e. when the input-token estimate was too low.
    """

    def __init__(self, global_cap_usd: float) -> None:
        self._lock = threading.RLock()
        self._reserved: dict[str, float] = {}
        self.global_reserved = 0.0
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

    def reserve(self, run_id: str, worst_case_usd: float) -> Reservation | None:
        """Hold ``worst_case_usd``; None if it would breach the run cap or the global cap."""
        with self._lock:
            run_held = self._spend.get(run_id, 0.0) + self._reserved.get(run_id, 0.0)
            glob_held = self.global_spend + self.global_reserved
            if (
                run_held + worst_case_usd > self._caps.get(run_id, 0.0)
                or glob_held + worst_case_usd > self.global_cap
            ):
                return None
            self._reserved[run_id] = self._reserved.get(run_id, 0.0) + worst_case_usd
            self.global_reserved += worst_case_usd
            return Reservation(run_id, worst_case_usd)

    def settle(self, reservation: Reservation, t: TokenCounts, cost: float) -> None:
        """Release the reservation and record actual spend (idempotent per reservation)."""
        with self._lock:
            if reservation.settled:
                return
            reservation.settled = True
            rid = reservation.run_id
            self._reserved[rid] = max(
                self._reserved.get(rid, 0.0) - reservation.worst_case_usd, 0.0
            )
            self.global_reserved = max(self.global_reserved - reservation.worst_case_usd, 0.0)
            self.record(rid, t, cost)  # same critical section: no window between release/record

    def reserved(self, run_id: str) -> float:
        return self._reserved.get(run_id, 0.0)

    def record(self, run_id: str, t: TokenCounts, cost: float) -> None:
        with self._lock:
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
