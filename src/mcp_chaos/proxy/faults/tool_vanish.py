"""F11 tool_vanish: drop a tool from tools/list once it has been called ``after_calls`` times."""

from __future__ import annotations

from typing import Any

from mcp_chaos.proxy.faults.base import CATALOG_REGISTRY, CatalogContext, CatalogOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, ToolVanishParams


class ToolVanish:
    def apply_list(self, ctx: CatalogContext, tools: list[dict[str, Any]]) -> CatalogOutcome:
        params = ctx.spec.params
        if not isinstance(params, ToolVanishParams):
            return CatalogOutcome(tools)
        kept: list[dict[str, Any]] = []
        gone: list[str] = []
        applied: list[FaultApplied] = []
        for tool in tools:
            name = tool.get("name")
            if isinstance(name, str) and ctx.tool_call_counts.get(name, 0) >= params.after_calls:
                gone.append(name)
                applied.append(
                    FaultApplied(
                        fault_id=ctx.spec.id,
                        mode=FaultMode.TOOL_VANISH,
                        detail={"tool": name, "after_calls": params.after_calls},
                    )
                )
            else:
                kept.append(tool)
        return CatalogOutcome(
            kept, notify_list_changed=bool(gone), vanished=frozenset(gone), applied=applied
        )


CATALOG_REGISTRY[FaultMode.TOOL_VANISH] = ToolVanish()
