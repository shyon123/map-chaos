"""F09 empty_result: successful result with no content."""

from __future__ import annotations

import copy
from typing import Any, cast

from mcp_chaos.proxy.faults._content import result_of
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import EmptyResultParams, FaultApplied, FaultMode


class EmptyResult:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        if not isinstance(ctx.spec.params, EmptyResultParams) or result_of(response) is None:
            return FaultOutcome()
        new = copy.deepcopy(response)
        result = cast("dict[str, Any]", new["result"])
        dropped = result.pop("structuredContent", None) is not None
        result["content"] = []
        result["isError"] = False
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.EMPTY_RESULT,
            detail={"dropped_structured_content": dropped},
        )
        return FaultOutcome(response_obj=new, applied=applied)


REGISTRY[FaultMode.EMPTY_RESULT] = EmptyResult()
