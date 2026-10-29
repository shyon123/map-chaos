"""F03 server_error: replace the response with a JSON-RPC error, or kill the upstream."""

from __future__ import annotations

from typing import Any

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, ServerErrorParams

MESSAGE = "Internal server error"


class ServerError:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        if "error" in response:
            return FaultOutcome()
        params = ctx.spec.params
        if not isinstance(params, ServerErrorParams):
            return FaultOutcome()
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.SERVER_ERROR,
            detail={"code": params.code, "crash": params.crash},
        )
        if params.crash:
            return FaultOutcome(crash=True, applied=applied)
        new: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": response.get("id"),
            "error": {"code": params.code, "message": MESSAGE},
        }
        return FaultOutcome(response_obj=new, applied=applied)


REGISTRY[FaultMode.SERVER_ERROR] = ServerError()
