"""Sans-IO response pipeline: size cap, then injection filter, per ``HardeningConfig`` toggles."""

from __future__ import annotations

from typing import Any, cast

from pydantic import JsonValue

from mcp_chaos.harden.inject_filter import filter_result
from mcp_chaos.harden.size_cap import cap_result
from mcp_chaos.harden.validate import is_incomplete, validate_tool_response
from mcp_chaos.schemas.runs import HardeningConfig

Detail = dict[str, JsonValue]


def validation_problems(
    h: HardeningConfig, obj: Any, output_schema: dict[str, Any] | None
) -> list[str]:
    """Problems with a tools/call response; empty when validation is off or the response is ok."""
    if not h.schema_validation:
        return []
    return validate_tool_response(obj, output_schema)


def transform_response(
    h: HardeningConfig, obj: dict[str, Any]
) -> tuple[dict[str, Any], list[Detail]]:
    """Apply size cap then injection filter to a validated tools/call response."""
    result: Any = obj.get("result")
    if not isinstance(result, dict):
        return obj, []
    res = cast("dict[str, Any]", result)
    if is_incomplete(res):
        return obj, []
    notes: list[Detail] = []
    if h.size_cap:
        res, lost = cap_result(res, h.size_cap_bytes)
        if lost:
            notes.append({"guard_action": "truncated", "bytes": lost, "cap": h.size_cap_bytes})
    if h.injection_filter:
        res, findings = filter_result(res)
        if findings:
            notes.append(
                {
                    "guard_action": "injection_flagged",
                    "spans": len(findings),
                    "chars": sum(f.end - f.start for f in findings),
                    "patterns": cast("list[JsonValue]", sorted({f.pattern for f in findings})),
                }
            )
    return {**obj, "result": res}, notes
