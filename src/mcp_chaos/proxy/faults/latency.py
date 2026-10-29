"""F02 latency: delay the response by ``delay_s`` plus seeded jitter, then deliver it intact."""

from __future__ import annotations

from typing import Any

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, LatencyParams


class Latency:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        if "error" in response:
            return FaultOutcome()
        params = ctx.spec.params
        if not isinstance(params, LatencyParams):
            return FaultOutcome()
        delay = params.delay_s
        if params.jitter_s > 0:
            delay += ctx.rng.uniform(0, params.jitter_s)
        delay = round(delay, 3)
        applied = FaultApplied(
            fault_id=ctx.spec.id, mode=FaultMode.LATENCY, detail={"delay_s": delay}
        )
        return FaultOutcome(delay_s=delay, applied=applied)


REGISTRY[FaultMode.LATENCY] = Latency()
