# ruff: noqa: S311
import copy
import json
import random
from typing import Any

import pytest

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext
from mcp_chaos.schemas import FaultMode

from .proxy_helpers import Position, injection_spec


def run(
    resp: dict[str, Any], position: Position, canary: str | None = None, seed: int = 3
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    spec = injection_spec(position, canary=canary)
    ctx = FaultContext(spec, random.Random(seed), "t", 0, None)
    before = copy.deepcopy(resp)
    out = REGISTRY[FaultMode.PROMPT_INJECTION].apply(ctx, resp)
    assert resp == before, "input mutated"
    return out.response_obj, ({} if out.applied is None else dict(out.applied.detail))


def ok(structured: Any = None, texts: tuple[str, ...] = ("hi",)) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": t} for t in texts]}
    if structured is not None:
        result["structuredContent"] = structured
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def test_prefix() -> None:
    out, d = run(ok(), "prefix", "CANARY-aaaaaaaa")
    assert out is not None
    c = out["result"]["content"]
    assert len(c) == 2 and "CANARY-aaaaaaaa" in c[0]["text"] and c[1]["text"] == "hi"
    assert d == {
        "template_id": "echo_in_answer",
        "position": "prefix",
        "canary": "CANARY-aaaaaaaa",
        "fallback": False,
    }


def test_suffix_and_derived_canary() -> None:
    out, d = run(ok(), "suffix")
    assert out is not None
    c = out["result"]["content"]
    assert c[0]["text"] == "hi" and d["canary"] in c[1]["text"]
    assert str(d["canary"]).startswith("CANARY-")
    assert run(ok(), "suffix")[1] == d


def test_field_rewrites_matching_text_block() -> None:
    sc = {"rows": [{"name": "a", "id": 1}], "count": 1, "z": "tail"}
    text = json.dumps(sc, sort_keys=True, separators=(",", ":"))
    out, d = run(ok(sc, (text,)), "field", "CANARY-bbbbbbbb")
    assert out is not None
    new_sc = out["result"]["structuredContent"]
    assert (
        new_sc["rows"][0]["name"].startswith("a ")
        and "CANARY-bbbbbbbb" in new_sc["rows"][0]["name"]
    )
    assert new_sc["z"] == "tail"
    assert json.loads(out["result"]["content"][0]["text"]) == new_sc
    assert len(out["result"]["content"]) == 1
    assert d["position"] == "field" and d["fallback"] is False


def test_field_unmatched_text_left_alone() -> None:
    out, _ = run(ok({"k": "v"}, ("not json",)), "field")
    assert out is not None and out["result"]["content"][0]["text"] == "not json"


def test_field_skips_non_text_blocks() -> None:
    resp = ok({"k": "v"})
    resp["result"]["content"] = ["junk", {"type": "image"}, {"type": "text", "text": 5}]
    out, d = run(resp, "field")
    assert out is not None and d["fallback"] is False


@pytest.mark.parametrize("sc", [None, {"n": 1, "l": [2, None]}, {}])
def test_field_fallback(sc: Any) -> None:
    out, d = run(ok(sc), "field")
    assert out is not None
    assert d["fallback"] is True and d["position"] == "suffix"
    assert len(out["result"]["content"]) == 2


def test_missing_content_created() -> None:
    out, _ = run({"jsonrpc": "2.0", "id": 1, "result": {}}, "prefix")
    assert out is not None and len(out["result"]["content"]) == 1


def test_noop_on_errors() -> None:
    for resp in (
        {"jsonrpc": "2.0", "id": 1, "error": {"code": 1, "message": "x"}},
        {"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": []}},
        {"jsonrpc": "2.0", "id": 1, "result": "weird"},
    ):
        spec = injection_spec()
        out = REGISTRY[FaultMode.PROMPT_INJECTION].apply(
            FaultContext(spec, random.Random(0), "t", 0, None), resp
        )
        assert out.applied is None and out.response_obj is None


def test_wrong_params_noop() -> None:
    from mcp_chaos.schemas import EmptyResultParams

    spec = injection_spec().model_copy(update={"params": EmptyResultParams()})
    out = REGISTRY[FaultMode.PROMPT_INJECTION].apply(
        FaultContext(spec, random.Random(0), "t", 0, None), ok()
    )
    assert out.applied is None


def test_dfs_order_sorted_keys() -> None:
    out, _ = run(ok({"b": "second", "a": {"y": "first"}}), "field", "CANARY-cccccccc")
    assert out is not None
    sc = out["result"]["structuredContent"]
    assert "CANARY" in sc["a"]["y"] and sc["b"] == "second"
