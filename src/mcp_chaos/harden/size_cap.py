"""Size cap: truncate oversized text results with an explicit marker."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

MARKER = "\n[mcp-chaos guard: truncated {n} bytes]"


def _trunc(text: str, keep: int) -> str:
    return text.encode("utf-8")[:keep].decode("utf-8", errors="ignore")


def cap_result(result: dict[str, Any], cap_bytes: int) -> tuple[dict[str, Any], int]:
    """Return ``(result, truncated_bytes)``; text beyond ``cap_bytes`` in total is cut.

    An oversized ``structuredContent`` is dropped (its size is reported in a marker block)
    because a truncated object would no longer be valid.
    """
    content: Any = result.get("content")
    if not isinstance(content, list):
        return result, 0
    out = copy.deepcopy(result)
    blocks = cast("list[Any]", out["content"])
    remaining = cap_bytes
    removed = 0
    for b in list(blocks):
        if not (isinstance(b, dict) and cast("dict[str, Any]", b).get("type") == "text"):
            continue
        blk = cast("dict[str, Any]", b)
        text: Any = blk.get("text")
        if not isinstance(text, str):
            continue
        size = len(text.encode("utf-8"))
        if size <= remaining:
            remaining -= size
            continue
        cut = _trunc(text, remaining)
        lost = size - len(cut.encode("utf-8"))
        blk["text"] = cut + MARKER.format(n=lost)
        removed += lost
        remaining = 0
    sc: Any = out.get("structuredContent")
    if isinstance(sc, dict):
        sc_bytes = len(json.dumps(sc, ensure_ascii=False).encode("utf-8"))
        if sc_bytes > cap_bytes:
            del out["structuredContent"]
            note = MARKER.format(n=sc_bytes).lstrip() + " (structuredContent dropped)"
            blocks.append({"type": "text", "text": note})
            removed += sc_bytes
    return (out, removed) if removed else (result, 0)
