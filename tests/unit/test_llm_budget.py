from __future__ import annotations

import json
from pathlib import Path

from mcp_chaos.llm.budget import BudgetTracker, TokenCounts, usage_from_body, usage_from_sse

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"


def test_usage_from_body() -> None:
    t = usage_from_body((FIXTURES / "response.json").read_bytes())
    assert (t.input, t.output, t.cache_write, t.cache_read) == (1000, 200, 0, 0)


def test_usage_from_body_garbage() -> None:
    for raw in (b"nope", b"[1]", b'{"usage": 3}', b'{"usage":{"input_tokens":true}}'):
        t = usage_from_body(raw)
        assert (t.input, t.output) == (0, 0)


def test_usage_from_sse_any_chunking() -> None:
    raw = (FIXTURES / "sse.txt").read_bytes()
    chunks = [raw[i : i + 37] for i in range(0, len(raw), 37)]
    t = usage_from_sse(chunks)
    assert (t.input, t.output) == (1000, 200)
    crlf = raw.replace(b"\n", b"\r\n")
    assert usage_from_sse([crlf]).output == 200


def test_usage_from_sse_ignores_junk() -> None:
    raw = b'data: nope\n\ndata: [1]\n\ndata: {"type":"message_start","message":3}\n\n'
    assert usage_from_sse([raw]).input == 0


def test_tracker_spend_caps_and_usage() -> None:
    bt = BudgetTracker(global_cap_usd=1.0)
    bt.register("a", 0.5)
    bt.register("b", 0.5)
    assert bt.known("a") and not bt.known("z")
    assert not bt.exceeded("a")
    t = TokenCounts()
    t.absorb(json.loads('{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":2}'))
    bt.record("a", t, 0.5)
    assert bt.exceeded("a")
    assert not bt.exceeded("b")
    bt.record("b", t, 0.5)
    assert bt.exceeded("b")  # global cap
    bt.record_cache_hit("a")
    u = bt.get_usage("a")
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_hits) == (10, 5, 2, 1)
    assert u.cost_usd == 0.5
    assert bt.get_usage("unknown").cost_usd == 0.0
    assert not bt.is_aborted("a")
    bt.mark_aborted("a")
    assert bt.is_aborted("a")
