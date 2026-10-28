"""Request-body normalization and cache-key computation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Final, cast

Rule = tuple[str, str]

_ISO = r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
DEFAULT_RULES: Final[list[Rule]] = [
    (r"Today's date is [^.\n]*", "Today's date is <DATE>"),
    (_ISO, "<DATE>"),
    (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "<UUID>"),
    (r"/private/var/[^\s\"'`)\]]*", "<TMP>"),
    (r"(?:/private)?/(?:tmp|var/folders)/[^\s\"'`)\]]*", "<TMP>"),
]


class BodyError(ValueError):
    """Request body is not a JSON object."""


def parse_body(raw: bytes) -> dict[str, Any]:
    try:
        obj: Any = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise BodyError("request body is not valid JSON") from e
    if not isinstance(obj, dict):
        raise BodyError("request body must be a JSON object")
    return cast("dict[str, Any]", obj)


def _apply(value: Any, rules: Sequence[tuple[re.Pattern[str], str]]) -> Any:
    if isinstance(value, str):
        for pat, rep in rules:
            value = pat.sub(rep, value)
        return value
    if isinstance(value, list):
        return [_apply(v, rules) for v in cast("list[Any]", value)]
    if isinstance(value, dict):
        return {k: _apply(v, rules) for k, v in cast("dict[str, Any]", value).items()}
    return value


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def normalize(body: dict[str, Any], rules: Sequence[Rule] | None = None) -> bytes:
    """Drop ``metadata``, regex-rewrite string leaves of ``system`` ONLY, canonicalize.

    ``messages`` (including tool_use/tool_result content) are hashed verbatim: rewriting dates or
    paths there would let requests with different tool results share a key.
    """
    compiled = [(re.compile(p), r) for p, r in (DEFAULT_RULES if rules is None else rules)]
    out = {k: v for k, v in body.items() if k != "metadata"}
    if "system" in out:
        out["system"] = _apply(out["system"], compiled)
    return canonical_json(out)


def compute_key(model: str, normalized: bytes, seed: int) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(normalized)
    h.update(b"\x00")
    h.update(str(seed).encode())
    return h.hexdigest()
