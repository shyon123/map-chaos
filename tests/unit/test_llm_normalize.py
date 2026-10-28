from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_chaos.llm.normalize import BodyError, compute_key, normalize, parse_body

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"


def _req() -> dict[str, Any]:
    return json.loads((FIXTURES / "request.json").read_text())


def test_metadata_removed_and_placeholders() -> None:
    out = normalize(_req()).decode()
    assert "metadata" not in out
    assert "session-abc" not in out
    assert "Today's date is <DATE>" in out
    assert "<UUID>" in out
    assert "<TMP>" in out
    assert "2026-10-28" not in out


@pytest.mark.parametrize(
    "text",
    [
        "/private/var/folders/ab/cd/T/xyz",
        "/tmp/abc/def.txt",  # noqa: S108
        "/var/folders/ab/cd",
    ],
)
def test_tmp_paths(text: str) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": f"see {text} now"}]}
    assert normalize(body).decode().count("<TMP>") == 1


def test_differing_only_by_volatile_values_same_key() -> None:
    a = _req()
    b = _req()
    b["system"] = (
        "You are a test agent. Today's date is 2027-01-02. "
        "Session aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee."
    )
    b["messages"][0]["content"][0]["text"] = (
        "List files in /tmp/pytest-of-other/pytest-9/work "
        "and note the time 2027-01-02T01:02:03+00:00."
    )
    b["metadata"] = {"user_id": "other"}
    ka = compute_key("m", normalize(a), 1)
    kb = compute_key("m", normalize(b), 1)
    assert ka == kb
    assert compute_key("m", normalize(a), 2) != ka
    assert compute_key("n", normalize(a), 1) != ka


def test_custom_rules_and_key_order_independent() -> None:
    a = {"model": "m", "messages": [{"role": "user", "content": "id 42"}], "max_tokens": 1}
    b = {"max_tokens": 1, "messages": [{"role": "user", "content": "id 99"}], "model": "m"}
    rules = [(r"\d+", "N")]
    assert normalize(a, rules) == normalize(b, rules)
    assert normalize(a) != normalize(b)


def test_non_string_leaves_untouched() -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": [{"n": 5, "ok": True}]}]}
    assert json.loads(normalize(body))["messages"][0]["content"][0] == {"n": 5, "ok": True}


def test_parse_body_errors() -> None:
    with pytest.raises(BodyError):
        parse_body(b"not json")
    with pytest.raises(BodyError):
        parse_body(b"[1]")
    assert parse_body(b'{"a":1}') == {"a": 1}
