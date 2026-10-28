"""F04 malformed_json: corrupt the JSON-RPC line (frame) or JSON inside a text block (content)."""

from __future__ import annotations

import copy
import json
import random
from typing import Any, cast

from mcp_chaos.proxy.faults._content import content_of, is_json, result_of, text_block
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, MalformedJsonParams

_GARBAGE = ("}garbage", "]]]", "\x00\x00", "{{", ' {"extra":')


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def corrupt_text(text: str, variant: str, rng: random.Random, min_cut: int = 0) -> str:
    """Corrupt serialized JSON text (truncate / bad_escape / trailing_garbage)."""
    if variant == "truncate":
        return text[: rng.randrange(min_cut, max(len(text), min_cut + 1))]
    if variant == "bad_escape":
        pos = rng.randrange(0, len(text) + 1)
        out = text[:pos] + "\\x" + text[pos:]
        # an insertion right after a lone backslash can still be valid; position 0 never is
        return out if not is_json(out) else "\\x" + text
    return text + rng.choice(_GARBAGE)


def _frame(
    ctx: FaultContext, params: MalformedJsonParams, response: dict[str, Any]
) -> FaultOutcome:
    if params.variant == "wrong_type":
        new = copy.deepcopy(response)
        new["result"] = _dumps(response["result"])
        raw = _dumps(new).encode("utf-8")
    else:
        raw = corrupt_text(_dumps(response), params.variant, ctx.rng, min_cut=1).encode("utf-8")
    applied = FaultApplied(
        fault_id=ctx.spec.id,
        mode=FaultMode.MALFORMED_JSON,
        detail={"layer": "frame", "variant": params.variant},
        parseable=is_json(raw),
    )
    return FaultOutcome(response_bytes=raw, applied=applied)


def _wrong_type(value: Any) -> Any:
    return [value] if isinstance(value, str) else _dumps(value)


def _content(
    ctx: FaultContext, params: MalformedJsonParams, response: dict[str, Any]
) -> FaultOutcome:
    new = copy.deepcopy(response)
    result = cast("dict[str, Any]", new["result"])
    content = content_of(result)
    texts = [(i, t) for i, b in enumerate(content) if (t := text_block(b)) is not None]
    if not texts:
        return FaultOutcome()
    target = next(((i, t) for i, t in texts if is_json(t)), None)
    found_json = target is not None
    variant = params.variant
    if target is None:
        target = texts[0]
        if variant == "wrong_type":
            variant = "truncate"  # nothing JSON to retype; corrupt the prose instead
    idx, text = target
    if variant == "wrong_type":
        new_text = _dumps(_wrong_type(json.loads(text)))
    else:
        new_text = corrupt_text(text, variant, ctx.rng)
    content[idx] = {**cast("dict[str, Any]", content[idx]), "text": new_text}
    dropped = result.pop("structuredContent", None) is not None
    applied = FaultApplied(
        fault_id=ctx.spec.id,
        mode=FaultMode.MALFORMED_JSON,
        detail={
            "layer": "content",
            "variant": variant,
            "block_index": idx,
            "found_json": found_json,
            "dropped_structured_content": dropped,
        },
        parseable=is_json(new_text),
    )
    return FaultOutcome(response_obj=new, applied=applied)


class MalformedJson:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        params = ctx.spec.params
        if not isinstance(params, MalformedJsonParams) or result_of(response) is None:
            return FaultOutcome()
        if params.layer == "frame":
            return _frame(ctx, params, response)
        return _content(ctx, params, response)


REGISTRY[FaultMode.MALFORMED_JSON] = MalformedJson()
