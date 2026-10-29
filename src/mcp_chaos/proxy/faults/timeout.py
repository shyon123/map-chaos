"""F01 timeout: hold the response for ``delay_s`` (then deliver intact), or never deliver it."""

from __future__ import annotations

from typing import Any

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, TimeoutParams


class Timeout:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        if "error" in response:
            return FaultOutcome()
        params = ctx.spec.params
        if not isinstance(params, TimeoutParams):
            return FaultOutcome()
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.TIMEOUT,
            detail={"delay_s": params.delay_s, "hang": params.hang},
        )
        if params.hang:
            return FaultOutcome(hang=True, applied=applied)
        return FaultOutcome(delay_s=params.delay_s, applied=applied)


REGISTRY[FaultMode.TIMEOUT] = Timeout()
