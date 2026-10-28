# ruff: noqa: S311
import copy
import json
import random
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.proxy.faults.discover import load_all
from mcp_chaos.proxy.faults.schema_drift import drift
from mcp_chaos.schemas import (
    FaultSpec,
    MalformedJsonParams,
    OversizedOutputParams,
    SchemaDriftParams,
)

load_all()

leaf = st.none() | st.booleans() | st.integers(-5, 5) | st.text(max_size=8)
key = st.sampled_from(["a", "b", "rows", "id", "x_y"])
json_val = st.recursive(
    leaf,
    lambda c: st.lists(c, max_size=3) | st.dictionaries(key, c, max_size=3),
    max_leaves=8,
)
text_block = st.one_of(
    st.text(max_size=20),
    json_val.map(lambda v: json.dumps(v)),
)


@st.composite
def results(draw: st.DrawFn) -> dict[str, Any]:
    texts = draw(st.lists(text_block, max_size=3))
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": t} for t in texts],
        "isError": draw(st.booleans()),
    }
    if draw(st.booleans()):
        result["structuredContent"] = draw(st.dictionaries(key, json_val, max_size=3))
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def run(params: Any, response: dict[str, Any], seed: int) -> FaultOutcome:
    spec = FaultSpec(id="f", mode=params.mode, params=params)
    ctx = FaultContext(spec, random.Random(seed), "t", 0, None)
    before = copy.deepcopy(response)
    out = REGISTRY[params.mode].apply(ctx, response)
    assert response == before
    return out


def valid(data: str | bytes) -> bool:
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


variants = st.sampled_from(["truncate", "bad_escape", "trailing_garbage", "wrong_type"])


@given(results(), variants, st.integers(0, 2**16))
@settings(max_examples=200, deadline=None)
def test_p3_malformed_frame(resp: dict[str, Any], variant: str, seed: int) -> None:
    out = run(MalformedJsonParams(variant=variant, layer="frame"), resp, seed)  # type: ignore[arg-type]
    assert out.response_bytes is not None and out.applied is not None
    assert out.applied.parseable is valid(out.response_bytes)
    assert out.applied.parseable is (variant == "wrong_type")


@given(results(), variants, st.integers(0, 2**16))
@settings(max_examples=200, deadline=None)
def test_p3_malformed_content(resp: dict[str, Any], variant: str, seed: int) -> None:
    out = run(MalformedJsonParams(variant=variant, layer="content"), resp, seed)  # type: ignore[arg-type]
    if out.applied is None:
        assert out.response_obj is None
        return
    assert out.response_obj is not None and out.applied.parseable is not None
    idx = out.applied.detail["block_index"]
    assert isinstance(idx, int)
    text = out.response_obj["result"]["content"][idx]["text"]
    # P3: valid JSON, or the trace says it is not
    assert out.applied.parseable is valid(text)
    if variant != "wrong_type":
        assert out.applied.parseable is False


drift_ops = st.sampled_from(["rename", "drop", "retype", "nest"])
paths = st.sampled_from(["a", "b", "rows", "rows[].id", "a.b", "a[].b", "nope", "x_y"])


@given(results(), drift_ops, paths, st.integers(0, 2**16))
@settings(max_examples=200, deadline=None)
def test_p3_schema_drift_valid_json(resp: dict[str, Any], op: str, path: str, seed: int) -> None:
    out = run(SchemaDriftParams(op=op, path=path), resp, seed)  # type: ignore[arg-type]
    assert out.applied is not None and out.applied.parseable is True
    final = out.response_obj if out.response_obj is not None else resp
    json.dumps(final)  # serializable
    for block in final["result"]["content"]:
        if valid(block["text"]):
            continue
        assert block in resp["result"]["content"]  # prose was already non-JSON, untouched


@given(json_val, drift_ops, paths)
def test_drift_only_touches_path(doc: Any, op: str, path: str) -> None:
    before = copy.deepcopy(doc)
    out, n = drift(doc, path, op)  # type: ignore[arg-type]
    assert doc == before
    if n == 0:
        assert out == doc


@given(
    st.dictionaries(key, json_val, min_size=1, max_size=4),
    drift_ops,
    st.sampled_from(["a", "b", "rows", "id", "x_y"]),
)
def test_drift_leaves_siblings(doc: dict[str, Any], op: str, path: str) -> None:
    out, _ = drift(doc, path, op)  # type: ignore[arg-type]
    for k, v in doc.items():
        if k != path:
            assert out[k] == v  # rename keeps other keys; drop/retype/nest only touch `path`


@given(
    st.sampled_from(["zeros", "lorem", "rows"]),
    st.integers(2_000, 20_000),
    results(),
    st.integers(0, 2**16),
)
@settings(max_examples=100, deadline=None)
def test_oversized_within_one_percent(
    filler: str, target: int, resp: dict[str, Any], seed: int
) -> None:
    params = OversizedOutputParams(bytes=target, filler=filler)  # type: ignore[arg-type]
    out = run(params, resp, seed)
    assert out.response_obj is not None
    n = len(
        json.dumps(
            out.response_obj["result"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )
    base = len(
        json.dumps(
            resp["result"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )
    if base + 40 <= target:
        assert abs(n - target) <= target * 0.01
