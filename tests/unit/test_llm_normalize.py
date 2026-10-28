from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_chaos.llm.normalize import BodyError, compute_key, normalize, parse_body

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"


def _req() -> dict[str, Any]:
    return json.loads((FIXTURES / "request.json").read_text())


def test_metadata_removed_and_system_placeholders() -> None:
    out = normalize(_req()).decode()
    assert "metadata" not in out
    assert "session-abc" not in out
    assert "Today's date is <DATE>" in out
    assert "<UUID>" in out
    assert "Today's date is 2026-10-28" not in out


def test_messages_hashed_verbatim() -> None:
    out = normalize(_req()).decode()
    assert "/tmp/pytest-of-user/pytest-3/work" in out  # noqa: S108
    assert "2026-10-28T09:15:00Z" in out
    assert "<TMP>" not in out


def test_system_text_blocks_normalized() -> None:
    body = {
        "model": "m",
        "system": [{"type": "text", "text": "Today's date is 2026-01-01. cwd /tmp/abc/x"}],
        "messages": [],
    }
    out = normalize(body).decode()
    assert "<DATE>" in out and "<TMP>" in out


def test_tool_result_date_differs_different_key() -> None:
    def req(date: str) -> dict[str, Any]:
        return {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t", "content": f"commit {date}"}
                    ],
                }
            ],
        }

    ka = compute_key("m", normalize(req("2026-01-01")), 1)
    kb = compute_key("m", normalize(req("2026-02-02")), 1)
    assert ka != kb


def test_system_date_differs_same_key() -> None:
    a = _req()
    b = _req()
    b["system"] = (
        "You are a test agent. Today's date is 2027-01-02. "
        "Session aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee."
    )
    b["metadata"] = {"user_id": "other"}
    assert compute_key("m", normalize(a), 1) == compute_key("m", normalize(b), 1)
    assert compute_key("m", normalize(a), 2) != compute_key("m", normalize(a), 1)
    assert compute_key("n", normalize(a), 1) != compute_key("m", normalize(a), 1)


def test_custom_rules_and_key_order_independent() -> None:
    a = {"model": "m", "system": "id 42", "messages": [], "max_tokens": 1}
    b = {"max_tokens": 1, "messages": [], "system": "id 99", "model": "m"}
    rules = [(r"\d+", "N")]
    assert normalize(a, rules) == normalize(b, rules)
    assert normalize(a) != normalize(b)


def test_non_string_leaves_untouched() -> None:
    body = {"model": "m", "system": [{"n": 5, "ok": True, "l": ["2026-01-01"]}]}
    assert json.loads(normalize(body))["system"][0] == {"n": 5, "ok": True, "l": ["<DATE>"]}


def test_parse_body_errors() -> None:
    with pytest.raises(BodyError):
        parse_body(b"not json")
    with pytest.raises(BodyError):
        parse_body(b"[1]")
    assert parse_body(b'{"a":1}') == {"a": 1}
