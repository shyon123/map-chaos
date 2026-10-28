"""F07 oversized_output: append a filler text block so the serialized result is ~N bytes."""

from __future__ import annotations

import copy
import json
import random
from typing import Any, cast

from mcp_chaos.proxy.faults._content import compact, result_of
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, OversizedOutputParams

_WORDS = [
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "do",
    "eiusmod",
    "tempor",
    "incididunt",
    "ut",
    "labore",
    "et",
    "dolore",
    "magna",
    "aliqua",
    "enim",
    "ad",
    "minim",
    "veniam",
    "quis",
    "nostrud",
    "exercitation",
]


def _escaped_len(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _fit(pieces: list[str], sep: str, n: int) -> str:
    """Join pieces (already >= n escaped bytes), trim, then pad with spaces to exactly n."""
    text = sep.join(pieces)
    while text and _escaped_len(text) > n:
        text = text[: max(len(text) - max(_escaped_len(text) - n, 1), 0)]
    return text + " " * (n - _escaped_len(text))


def _filler(kind: str, n: int, rng: random.Random) -> str:
    if n <= 0:
        return ""
    if kind == "zeros":
        return "0" * n
    pieces: list[str] = []
    size = 0
    while size < n:
        if kind == "lorem":
            piece = rng.choice(_WORDS)
        else:
            note = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 6)))
            piece = json.dumps({"id": rng.randrange(10**6), "note": note})
        pieces.append(piece)
        size += _escaped_len(piece) + (2 if kind == "rows" else 1)
    return _fit(pieces, "\n" if kind == "rows" else " ", n)


class OversizedOutput:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        params = ctx.spec.params
        if not isinstance(params, OversizedOutputParams) or result_of(response) is None:
            return FaultOutcome()
        new = copy.deepcopy(response)
        result = cast("dict[str, Any]", new["result"])
        raw: Any = result.get("content")
        content: list[Any] = cast("list[Any]", raw) if isinstance(raw, list) else []
        result["content"] = content
        content.append({"type": "text", "text": ""})
        base = len(compact(result).encode("utf-8"))
        text = _filler(params.filler, params.bytes - base, ctx.rng)
        content[-1] = {"type": "text", "text": text}
        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.OVERSIZED_OUTPUT,
            detail={"bytes": params.bytes, "filler": params.filler},
        )
        return FaultOutcome(response_obj=new, applied=applied)


REGISTRY[FaultMode.OVERSIZED_OUTPUT] = OversizedOutput()
