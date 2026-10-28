"""F05 schema_drift (content layer): rename/drop/retype/nest fields along a dotted path.

Path syntax: ``a.b`` descends dict keys; ``[]`` after a key iterates list elements
(``rows[].id``). The drift is applied to ``structuredContent`` and to every JSON text block
whose parsed value contains the path (re-serialized compact, sorted keys). The catalog layer
(``tools/list``) is handled elsewhere; ``layer == "catalog"`` is a no-op here.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal, cast

from mcp_chaos.proxy.faults._content import compact, content_of, result_of, text_block
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, SchemaDriftParams

_LIST = object()
Op = Literal["rename", "drop", "retype", "nest"]


def tokenize(path: str) -> list[Any]:
    """``rows[].id`` -> ``["rows", _LIST, "id"]``."""
    tokens: list[Any] = []
    for seg in path.split("."):
        lists = 0
        while seg.endswith("[]"):
            seg = seg[:-2]
            lists += 1
        if seg:
            tokens.append(seg)
        tokens.extend([_LIST] * lists)
    return tokens


def retype(v: Any) -> Any:
    """int/float -> str, numeric str -> number, other str -> [str], bool -> "true"/"false",
    None -> "null", list -> {"items": list}, dict -> [dict]."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        for conv in (int, float):
            try:
                return conv(v)
            except ValueError:
                continue
        return [v]
    if v is None:
        return "null"
    if isinstance(v, list):
        return {"items": cast("list[Any]", v)}
    return [v]


def _transform(op: Op, v: Any) -> Any:
    return retype(v) if op == "retype" else {"value": v}


def _apply(node: Any, tokens: list[Any], op: Op) -> int:
    """Mutate ``node`` in place; return the number of matches."""
    tok, rest = tokens[0], tokens[1:]
    if tok is _LIST:
        if not isinstance(node, list):
            return 0
        items = cast("list[Any]", node)
        if rest:
            return sum(_apply(x, rest, op) for x in items)
        if op == "rename":
            return 0
        n = len(items)
        if op == "drop":
            items.clear()
        else:
            items[:] = [_transform(op, x) for x in items]
        return n
    if not isinstance(node, dict):
        return 0
    d = cast("dict[str, Any]", node)
    if tok not in d:
        return 0
    if rest:
        return _apply(d[tok], rest, op)
    if op == "drop":
        del d[tok]
    elif op == "rename":
        renamed = {(tok + "_v2" if k == tok else k): v for k, v in d.items()}
        d.clear()
        d.update(renamed)
    else:
        d[tok] = _transform(op, d[tok])
    return 1


def drift(doc: Any, path: str, op: Op) -> tuple[Any, int]:
    """Return a drifted deep copy of ``doc`` and the match count. Never mutates ``doc``."""
    out = copy.deepcopy(doc)
    tokens = tokenize(path)
    return out, (_apply(out, tokens, op) if tokens else 0)


class SchemaDrift:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        params = ctx.spec.params
        if (
            not isinstance(params, SchemaDriftParams)
            or params.layer == "catalog"
            or result_of(response) is None
        ):
            return FaultOutcome()
        new = copy.deepcopy(response)
        result = cast("dict[str, Any]", new["result"])
        has_sc = "structuredContent" in result
        sc_matched = 0
        if has_sc:
            result["structuredContent"], sc_matched = drift(
                result["structuredContent"], params.path, params.op
            )
        text_matched = 0
        content = content_of(result)
        for i, b in enumerate(content):
            text = text_block(b)
            if text is None:
                continue
            try:
                parsed = json.loads(text)
            except ValueError:
                continue
            if not isinstance(parsed, (dict, list)):
                continue
            new_doc, n = drift(parsed, params.path, params.op)
            if n:
                content[i] = {**cast("dict[str, Any]", b), "text": compact(new_doc)}
                text_matched += n
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.SCHEMA_DRIFT,
            detail={
                "op": params.op,
                "path": params.path,
                "matched": sc_matched if has_sc else text_matched,
                "structured_matched": sc_matched,
                "text_matched": text_matched,
            },
            parseable=True,
        )
        changed = bool(sc_matched or text_matched)
        return FaultOutcome(response_obj=new if changed else None, applied=applied)


REGISTRY[FaultMode.SCHEMA_DRIFT] = SchemaDrift()
