"""Shared helpers for content-layer faults (not a fault itself; skipped by discovery)."""

from __future__ import annotations

import json
from typing import Any, cast


def compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def is_json(text: str | bytes) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def result_of(response: dict[str, Any]) -> dict[str, Any] | None:
    """The ``result`` object of a successful JSON-RPC response, else None."""
    result: Any = response.get("result")
    if "error" in response or not isinstance(result, dict):
        return None
    return cast("dict[str, Any]", result)


def content_of(result: dict[str, Any]) -> list[Any]:
    """The ``content`` list of a result, or an empty (detached) list if absent."""
    raw: Any = result.get("content")
    return cast("list[Any]", raw) if isinstance(raw, list) else []


def text_block(block: Any) -> str | None:
    """The text of a ``type: text`` content block, else None."""
    if not isinstance(block, dict):
        return None
    blk = cast("dict[str, Any]", block)
    text = blk.get("text")
    return text if blk.get("type") == "text" and isinstance(text, str) else None
