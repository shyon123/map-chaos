"""Seeded, deterministic fault-selection engine."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from fnmatch import fnmatchcase

from mcp_chaos.proxy.faults.base import REGISTRY
from mcp_chaos.proxy.faults.discover import load_all
from mcp_chaos.schemas import FaultPlan, FaultSpec

load_all()


class UnsupportedFaultError(Exception):
    """The plan uses a fault mode with no registered implementation."""


def derive_rng(seed: int, server: str, tool: str, call_index: int) -> random.Random:
    h = hashlib.blake2b(digest_size=16)
    for part in (str(seed), server, tool, str(call_index)):
        data = part.encode("utf-8")
        h.update(len(data).to_bytes(4, "big"))
        h.update(data)
    return random.Random(int.from_bytes(h.digest(), "big"))  # noqa: S311


@dataclass(frozen=True)
class Decision:
    call_index: int
    rng: random.Random
    spec: FaultSpec | None


class FaultEngine:
    def __init__(self, plan: FaultPlan, server: str) -> None:
        missing = sorted({f.mode.value for f in plan.faults if f.mode not in REGISTRY})
        if missing:
            raise UnsupportedFaultError(
                f"no implementation for fault mode(s): {', '.join(missing)}"
            )
        self.plan = plan
        self.server = server
        self._counts: dict[str, int] = {}

    def reserve_index(self, tool: str) -> int:
        """Allocate the next per-tool call index (called at request time)."""
        idx = self._counts.get(tool, 0)
        self._counts[tool] = idx + 1
        return idx

    def decide(self, tool: str, call_index: int) -> Decision:
        idx = call_index
        rng = derive_rng(self.plan.seed, self.server, tool, idx)
        roll = rng.random()  # always drawn first so the stream is consistent
        for spec in self.plan.faults:
            t = spec.trigger
            if not (fnmatchcase(tool, t.tool) and fnmatchcase(self.server, t.server)):
                continue
            if t.calls is not None and idx not in t.calls:
                continue
            if roll < t.probability:
                return Decision(idx, rng, spec)
        return Decision(idx, rng, None)
