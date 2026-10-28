"""F10 lying_status: make ``isError`` disagree with the content."""

from __future__ import annotations

import copy
from typing import Any, cast

from mcp_chaos.proxy.faults._content import result_of
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, LyingStatusParams

ERROR_TEXT = "Error: operation failed: resource temporarily unavailable"


class LyingStatus:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        params = ctx.spec.params
        if not isinstance(params, LyingStatusParams) or result_of(response) is None:
            return FaultOutcome()
        new = copy.deepcopy(response)
        result = cast("dict[str, Any]", new["result"])
        if params.direction == "error_as_ok":
            result.pop("structuredContent", None)
            result["content"] = [{"type": "text", "text": ERROR_TEXT}]
            result["isError"] = False
        else:
            result["isError"] = True
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.LYING_STATUS,
            detail={"direction": params.direction},
        )
        return FaultOutcome(response_obj=new, applied=applied)


REGISTRY[FaultMode.LYING_STATUS] = LyingStatus()
