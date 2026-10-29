"""Sanitizing untrusted run output before it reaches the red-team prompt."""

from __future__ import annotations

import re

MAX_ANSWER_CHARS = 200
REDACTED = "[redacted]"
_SEGMENT = re.compile(r"(?<=[.!?;])\s+|\n+")
_INSTRUCTIONISH = re.compile(
    r"CANARY|\b(system|assistant|developer|user|human)\s*:|"
    r"\b(ignore|disregard|forget|override|obey|follow|pretend|instructions?|prompt)\b|"
    r"\b(you\s+(must|should|will|are|need)|please|always|never)\b|"
    r"\b(call|invoke|execute|run|write|append|delete|remove|send|tell|insert|include|output|"
    r"respond|reply|say|print|echo|exfiltrate|fetch|download)\b",
    re.IGNORECASE,
)
_STRIP = re.compile(r"[^\x20-\x7e]|[<>{}`\[\]]")


def sanitize_text(text: str | None, limit: int = MAX_ANSWER_CHARS) -> str:
    """Drop instruction-like sentences and markup/non-ASCII characters; clip to ``limit``."""
    if not text:
        return ""
    kept: list[str] = []
    for raw in _SEGMENT.split(text):
        seg = _STRIP.sub("", raw).strip()
        if not seg:
            continue
        piece = REDACTED if _INSTRUCTIONISH.search(seg) else seg
        if piece == REDACTED and kept and kept[-1] == REDACTED:
            continue
        kept.append(piece)
    out = " ".join(kept)
    return out if len(out) <= limit else out[: limit - 3] + "..."
