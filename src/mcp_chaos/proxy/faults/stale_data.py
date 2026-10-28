"""F06 stale_data: replay an earlier unfaulted response for the same tool+args.

``snapshot_id`` (fixture snapshot replay) is out of scope: setting it raises
``NotImplementedError`` at apply time.
"""

from __future__ import annotations

import copy
from typing import Any, cast

from mcp_chaos.proxy.faults._content import result_of
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, StaleDataParams


class StaleData:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        params = ctx.spec.params
        if not isinstance(params, StaleDataParams):
            return FaultOutcome()
        if params.snapshot_id is not None:
            raise NotImplementedError("stale_data.snapshot_id is not supported; use lag_calls")
        if result_of(response) is None or len(ctx.history) < params.lag_calls:
            return FaultOutcome()
        index = len(ctx.history) - params.lag_calls
        old: Any = ctx.history[index].get("result")
        if not isinstance(old, dict):
            return FaultOutcome()
        new: dict[str, Any] = {**response, "result": copy.deepcopy(cast("dict[str, Any]", old))}
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.STALE_DATA,
            detail={"lag_calls": params.lag_calls, "replayed_from_index": index},
        )
        return FaultOutcome(response_obj=new, applied=applied)


REGISTRY[FaultMode.STALE_DATA] = StaleData()
