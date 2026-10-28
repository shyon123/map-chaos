# ruff: noqa: S311
import copy
import json
import random
from typing import Any

import pytest

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.proxy.faults.discover import load_all
from mcp_chaos.proxy.faults.lying_status import ERROR_TEXT
from mcp_chaos.proxy.faults.schema_drift import drift, tokenize
from mcp_chaos.schemas import (
    EmptyResultParams,
    FaultSpec,
    LyingStatusParams,
    MalformedJsonParams,
    OversizedOutputParams,
    SchemaDriftParams,
    StaleDataParams,
)

load_all()

SC = {"rows": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], "count": 2, "ok": True}


def resp(
    structured: Any = None, texts: tuple[str, ...] | None = None, is_error: bool = False
) -> dict[str, Any]:
    if texts is None:
        texts = (json.dumps(structured, sort_keys=True, separators=(",", ":")),)
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": t} for t in texts],
        "isError": is_error,
    }
    if structured is not None:
        result["structuredContent"] = structured
    return {"jsonrpc": "2.0", "id": 5, "result": result}


def run(
    params: Any, response: dict[str, Any], seed: int = 1, history: tuple[dict[str, Any], ...] = ()
) -> FaultOutcome:
    spec = FaultSpec(id="f", mode=params.mode, params=params)
    ctx = FaultContext(spec, random.Random(seed), "t", 0, None, history)
    before = copy.deepcopy(response)
    out = REGISTRY[params.mode].apply(ctx, response)
    assert response == before, "input mutated"
    return out


ERR: dict[str, Any] = {"jsonrpc": "2.0", "id": 5, "error": {"code": -1, "message": "x"}}
ALL_PARAMS: list[Any] = [
    MalformedJsonParams(layer="frame"),
    MalformedJsonParams(layer="content"),
    SchemaDriftParams(path="count"),
    StaleDataParams(),
    OversizedOutputParams(bytes=500),
    EmptyResultParams(),
    LyingStatusParams(),
]


@pytest.mark.parametrize("params", ALL_PARAMS, ids=lambda p: p.mode.value)
def test_registered_and_noop_on_error(params: Any) -> None:
    assert params.mode in REGISTRY
    out = run(params, ERR, history=(resp(SC),))
    assert out == FaultOutcome()


# ---------------- F04 ----------------


@pytest.mark.parametrize("variant", ["truncate", "bad_escape", "trailing_garbage"])
@pytest.mark.parametrize("seed", range(20))
def test_frame_variants_unparseable(variant: str, seed: int) -> None:
    out = run(MalformedJsonParams(variant=variant, layer="frame"), resp(SC), seed)  # type: ignore[arg-type]
    assert out.response_bytes is not None and out.applied is not None
    assert out.applied.parseable is False
    with pytest.raises(ValueError):
        json.loads(out.response_bytes)


def test_frame_wrong_type_valid_json() -> None:
    out = run(MalformedJsonParams(variant="wrong_type", layer="frame"), resp(SC))
    assert out.response_bytes is not None and out.applied is not None
    assert out.applied.parseable is True
    assert isinstance(json.loads(out.response_bytes)["result"], str)


def test_bad_escape_after_backslash_still_invalid() -> None:
    r = resp(texts=("\\n\\n\\n\\n",))
    for seed in range(60):
        out = run(MalformedJsonParams(variant="bad_escape", layer="frame"), r, seed)
        assert out.applied is not None and out.applied.parseable is False
        c = run(MalformedJsonParams(variant="bad_escape", layer="content"), r, seed)
        assert c.applied is not None and c.applied.parseable is False


@pytest.mark.parametrize("variant", ["truncate", "bad_escape", "trailing_garbage"])
def test_content_corrupts_json_block_and_drops_structured(variant: str) -> None:
    r = resp(SC, texts=("prose", json.dumps(SC)))
    out = run(MalformedJsonParams(variant=variant, layer="content"), r)  # type: ignore[arg-type]
    assert out.response_obj is not None and out.applied is not None
    res = out.response_obj["result"]
    assert "structuredContent" not in res
    assert res["content"][0]["text"] == "prose"
    assert out.applied.detail["block_index"] == 1
    assert out.applied.detail["found_json"] is True
    assert out.applied.detail["dropped_structured_content"] is True
    assert out.applied.parseable is False
    with pytest.raises(ValueError):
        json.loads(res["content"][1]["text"])


def test_content_wrong_type() -> None:
    out = run(MalformedJsonParams(variant="wrong_type", layer="content"), resp(SC))
    assert out.response_obj is not None and out.applied is not None
    text = out.response_obj["result"]["content"][0]["text"]
    assert json.loads(json.loads(text)) == SC
    assert out.applied.parseable is True
    out2 = run(MalformedJsonParams(variant="wrong_type", layer="content"), resp(texts=('"s"',)))
    assert out2.response_obj is not None
    assert json.loads(out2.response_obj["result"]["content"][0]["text"]) == ["s"]


@pytest.mark.parametrize("variant", ["truncate", "bad_escape", "trailing_garbage", "wrong_type"])
def test_content_no_json_block_falls_back(variant: str) -> None:
    out = run(
        MalformedJsonParams(variant=variant, layer="content"),  # type: ignore[arg-type]
        resp(texts=("hello world, plain prose",)),
    )
    assert out.response_obj is not None and out.applied is not None
    assert out.applied.detail["found_json"] is False
    assert out.applied.detail["dropped_structured_content"] is False
    assert out.applied.parseable is False
    assert out.response_obj["result"]["content"][0]["text"] != "hello world, plain prose"


def test_content_no_text_block_noop() -> None:
    r = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "image", "data": ""}]}}
    assert run(MalformedJsonParams(layer="content"), r) == FaultOutcome()
    assert run(MalformedJsonParams(layer="content"), {"id": 1, "result": {}}) == FaultOutcome()


def test_malformed_determinism() -> None:
    for layer in ("frame", "content"):
        p = MalformedJsonParams(variant="truncate", layer=layer)  # type: ignore[arg-type]
        assert run(p, resp(SC), 9) == run(p, resp(SC), 9)
    outs = {
        run(MalformedJsonParams(variant="truncate", layer="frame"), resp(SC), s).response_bytes
        for s in range(10)
    }
    assert len(outs) > 1


# ---------------- F05 ----------------


def test_tokenize() -> None:
    assert tokenize("count") == ["count"]
    assert tokenize("rows[].id")[0] == "rows" and tokenize("rows[].id")[2] == "id"
    assert len(tokenize("a[][]")) == 3


def sd(op: str, path: str, r: dict[str, Any] | None = None, layer: str = "content") -> FaultOutcome:
    return run(SchemaDriftParams(op=op, path=path, layer=layer), r or resp(SC))  # type: ignore[arg-type]


def test_rename_top_and_nested_keeps_order() -> None:
    out = sd("rename", "count")
    assert out.response_obj is not None and out.applied is not None
    sc = out.response_obj["result"]["structuredContent"]
    assert list(sc) == ["rows", "count_v2", "ok"] and sc["count_v2"] == 2
    assert json.loads(out.response_obj["result"]["content"][0]["text"]) == sc
    assert out.applied.detail["matched"] == 1 and out.applied.parseable is True
    nested = sd("rename", "rows[].id").response_obj
    assert nested is not None
    assert nested["result"]["structuredContent"]["rows"][1] == {"id_v2": 2, "name": "b"}


def test_drop() -> None:
    out = sd("drop", "rows[].name")
    assert out.response_obj is not None and out.applied is not None
    assert out.response_obj["result"]["structuredContent"]["rows"] == [{"id": 1}, {"id": 2}]
    assert out.applied.detail["matched"] == 2
    whole = sd("drop", "rows[]").response_obj
    assert whole is not None and whole["result"]["structuredContent"]["rows"] == []
    top = sd("drop", "count").response_obj
    assert top is not None and "count" not in top["result"]["structuredContent"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, "3"),
        (1.5, "1.5"),
        (True, "true"),
        (False, "false"),
        ("7", 7),
        ("7.5", 7.5),
        ("abc", ["abc"]),
        (None, "null"),
        ([1], {"items": [1]}),
        ({"a": 1}, [{"a": 1}]),
    ],
)
def test_retype(value: Any, expected: Any) -> None:
    doc, n = drift({"v": value}, "v", "retype")
    assert n == 1 and doc == {"v": expected}


def test_retype_list_elements_and_nest() -> None:
    out = sd("retype", "rows[].id").response_obj
    assert out is not None
    assert [r["id"] for r in out["result"]["structuredContent"]["rows"]] == ["1", "2"]
    nested = sd("nest", "count").response_obj
    assert nested is not None and nested["result"]["structuredContent"]["count"] == {"value": 2}
    elems = sd("nest", "rows[]").response_obj
    assert elems is not None
    assert elems["result"]["structuredContent"]["rows"][0] == {"value": {"id": 1, "name": "a"}}


def test_list_element_rename_not_matched() -> None:
    out = sd("rename", "rows[]")
    assert out.response_obj is None and out.applied is not None
    assert out.applied.detail["matched"] == 0


def test_path_not_found_still_applied() -> None:
    for path in ("nope", "rows.id", "count[]", "count.x", "rows[].nope", "[]"):
        out = sd("rename", path)
        assert out.response_obj is None and out.applied is not None, path
        assert out.applied.detail["matched"] == 0
        assert out.applied.parseable is True


def test_text_block_only_no_structured() -> None:
    r = resp(texts=("note", json.dumps({"count": 1, "b": 2}), "[1]", "7"))
    out = sd("rename", "count", r)
    assert out.response_obj is not None and out.applied is not None
    texts = [b["text"] for b in out.response_obj["result"]["content"]]
    assert texts == ["note", '{"b":2,"count_v2":1}', "[1]", "7"]
    assert out.applied.detail["matched"] == 1
    assert out.applied.detail["text_matched"] == 1


def test_catalog_layer_noop() -> None:
    assert sd("rename", "count", layer="catalog") == FaultOutcome()


def test_drift_does_not_mutate() -> None:
    doc = {"a": {"b": [1]}}
    before = copy.deepcopy(doc)
    drift(doc, "a.b", "retype")
    assert doc == before
    assert drift(doc, "", "drop") == (doc, 0)


# ---------------- F06 ----------------


def test_stale_replays_with_current_id() -> None:
    old1, old2 = resp(texts=("v1",)), resp(texts=("v2",))
    old1["id"], old2["id"] = 1, 2
    cur = resp(texts=("v3",))
    out = run(StaleDataParams(lag_calls=1), cur, history=(old1, old2))
    assert out.response_obj is not None and out.applied is not None
    assert out.response_obj["id"] == 5
    assert out.response_obj["result"] == old2["result"]
    assert out.response_obj["result"] is not old2["result"]
    assert out.applied.detail == {"lag_calls": 1, "replayed_from_index": 1}
    out2 = run(StaleDataParams(lag_calls=2), cur, history=(old1, old2))
    assert out2.response_obj is not None and out2.applied is not None
    assert out2.response_obj["result"] == old1["result"]
    assert out2.applied.detail == {"lag_calls": 2, "replayed_from_index": 0}


def test_stale_short_history_noop() -> None:
    assert run(StaleDataParams(lag_calls=2), resp(SC), history=(resp(SC),)) == FaultOutcome()
    assert run(StaleDataParams(), resp(SC)) == FaultOutcome()


def test_stale_bad_history_entry_noop() -> None:
    assert run(StaleDataParams(), resp(SC), history=({"id": 1},)) == FaultOutcome()


def test_stale_snapshot_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="snapshot_id"):
        run(StaleDataParams(snapshot_id="s1"), resp(SC))


# ---------------- F07 ----------------


def size(r: dict[str, Any]) -> int:
    return len(
        json.dumps(r["result"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


@pytest.mark.parametrize("filler", ["zeros", "lorem", "rows"])
@pytest.mark.parametrize("target", [400, 1000, 50_000, 1_000_000])
def test_oversized_exact_size(filler: str, target: int) -> None:
    out = run(OversizedOutputParams(bytes=target, filler=filler), resp(SC))  # type: ignore[arg-type]
    assert out.response_obj is not None and out.applied is not None
    assert size(out.response_obj) == target
    assert out.applied.detail == {"bytes": target, "filler": filler}
    assert out.response_obj["result"]["content"][0] == resp(SC)["result"]["content"][0]
    assert out.response_obj["result"]["structuredContent"] == SC


def test_oversized_filler_shapes() -> None:
    def last_text(filler: str) -> str:
        out = run(OversizedOutputParams(bytes=3000, filler=filler), resp(SC))  # type: ignore[arg-type]
        assert out.response_obj is not None
        return out.response_obj["result"]["content"][-1]["text"]

    assert set(last_text("zeros")) == {"0"}
    assert "lorem" in last_text("lorem") or "ipsum" in last_text("lorem")
    rows = [ln for ln in last_text("rows").split("\n") if ln.strip()]
    parsed = [json.loads(ln) for ln in rows[:-1]]
    assert all(set(p) == {"id", "note"} for p in parsed) and parsed


def test_oversized_determinism_and_small_target() -> None:
    p = OversizedOutputParams(bytes=5000, filler="rows")
    assert run(p, resp(SC), 4) == run(p, resp(SC), 4)
    assert run(p, resp(SC), 4) != run(p, resp(SC), 5)
    tiny = run(OversizedOutputParams(bytes=1), resp(SC))
    assert tiny.response_obj is not None
    assert tiny.response_obj["result"]["content"][-1]["text"] == ""
    bare = run(OversizedOutputParams(bytes=100, filler="zeros"), {"id": 1, "result": {}})
    assert bare.response_obj is not None and size(bare.response_obj) == 100


# ---------------- F09 / F10 ----------------


def test_empty_result() -> None:
    out = run(EmptyResultParams(), resp(SC, is_error=True))
    assert out.response_obj is not None and out.applied is not None
    assert out.response_obj["result"] == {"content": [], "isError": False}
    assert out.applied.detail == {"dropped_structured_content": True}
    plain = run(EmptyResultParams(), resp(texts=("x",)))
    assert plain.applied is not None
    assert plain.applied.detail == {"dropped_structured_content": False}


def test_lying_error_as_ok() -> None:
    out = run(LyingStatusParams(direction="error_as_ok"), resp(SC, is_error=True))
    assert out.response_obj is not None and out.applied is not None
    assert out.response_obj["result"] == {
        "content": [{"type": "text", "text": ERROR_TEXT}],
        "isError": False,
    }
    assert out.applied.detail == {"direction": "error_as_ok"}


def test_lying_ok_as_error() -> None:
    r = resp(SC)
    out = run(LyingStatusParams(direction="ok_as_error"), r)
    assert out.response_obj is not None
    assert out.response_obj["result"]["isError"] is True
    assert out.response_obj["result"]["content"] == r["result"]["content"]
    assert out.response_obj["result"]["structuredContent"] == SC


def test_core_trace_describes_emitted_bytes() -> None:
    import hashlib

    from mcp_chaos.proxy.core import ProxyCore, SendToClient, Trace
    from mcp_chaos.proxy.engine import FaultEngine

    from .proxy_helpers import plan

    class Clock:
        def now_ms(self) -> int:
            return 0

    params = MalformedJsonParams(variant="trailing_garbage", layer="frame")
    fs = FaultSpec(id="f", mode=params.mode, params=params)
    core = ProxyCore(
        FaultEngine(plan(fs), "srv"), run_id="r", agent="a", server="srv", clock=Clock()
    )
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "t"}}
    core.on_client_line(json.dumps(req).encode())
    actions = core.on_server_line(json.dumps(resp(SC) | {"id": 1}).encode())
    sent = next(a for a in actions if isinstance(a, SendToClient))
    ev = next(a.event for a in actions if isinstance(a, Trace))
    assert ev.result_bytes == len(sent.line)
    assert ev.result_sha256 == hashlib.sha256(sent.line).hexdigest()
