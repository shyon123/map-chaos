"""F05 schema_drift (catalog layer): alter a tool's inputSchema in tools/list."""

from __future__ import annotations

import copy
from typing import Any, cast

from mcp_chaos.proxy.faults.base import CATALOG_REGISTRY, CatalogContext, CatalogOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, SchemaDriftParams

_RETYPE = {
    "string": "integer",
    "integer": "string",
    "number": "string",
    "boolean": "string",
    "array": "object",
    "object": "array",
}


def _dict(v: Any) -> dict[str, Any] | None:
    return cast("dict[str, Any]", v) if isinstance(v, dict) else None


def _locate(schema: dict[str, Any], path: str) -> tuple[dict[str, Any], str] | None:
    """Return (parent object schema, final property name) for a dotted path, if it exists."""
    parts = path.split(".")
    node = schema
    for part in parts[:-1]:
        props = _dict(node.get("properties"))
        child = _dict(props.get(part)) if props is not None else None
        if child is None:
            return None
        node = child
    props = _dict(node.get("properties"))
    if props is None or parts[-1] not in props:
        return None
    return node, parts[-1]


def _required(node: dict[str, Any]) -> list[Any] | None:
    req: Any = node.get("required")
    return cast("list[Any]", req) if isinstance(req, list) else None


def apply_op(schema: dict[str, Any], op: str, path: str) -> bool:
    """Mutate ``schema`` in place; returns whether anything changed."""
    loc = _locate(schema, path)
    if loc is None:
        return False
    node, name = loc
    props = cast("dict[str, Any]", node["properties"])
    req = _required(node)
    if op == "drop":
        del props[name]
        if req is not None:
            node["required"] = [r for r in req if r != name]
    elif op == "rename":
        new = f"{name}_v2"
        if new in props:
            return False
        node["properties"] = {(new if k == name else k): v for k, v in props.items()}
        if req is not None:
            node["required"] = [new if r == name else r for r in req]
    elif op == "retype":
        prop = _dict(props[name])
        if prop is None:
            return False
        old = prop.get("type")
        prop["type"] = _RETYPE.get(old, "string") if isinstance(old, str) else "string"
        prop.pop("enum", None)
    else:  # nest: wrap the property in a single-field object
        inner: dict[str, Any] = {"type": "object", "properties": {name: props[name]}}
        if req is not None and name in req:
            inner["required"] = [name]
        props[name] = inner
    return True


class CatalogDrift:
    def apply_list(self, ctx: CatalogContext, tools: list[dict[str, Any]]) -> CatalogOutcome:
        params = ctx.spec.params
        if not isinstance(params, SchemaDriftParams):
            return CatalogOutcome(tools)
        out: list[dict[str, Any]] = []
        applied: list[FaultApplied] = []
        for tool in tools:
            if _dict(tool.get("inputSchema")) is None:
                out.append(tool)
                continue
            new_tool = copy.deepcopy(tool)
            if apply_op(cast("dict[str, Any]", new_tool["inputSchema"]), params.op, params.path):
                out.append(new_tool)
                applied.append(
                    FaultApplied(
                        fault_id=ctx.spec.id,
                        mode=FaultMode.SCHEMA_DRIFT,
                        detail={
                            "layer": "catalog",
                            "tool": str(tool.get("name")),
                            "op": params.op,
                            "path": params.path,
                        },
                        parseable=True,
                    )
                )
            else:
                out.append(tool)
        return CatalogOutcome(out, applied=applied)


CATALOG_REGISTRY[FaultMode.SCHEMA_DRIFT] = CatalogDrift()
