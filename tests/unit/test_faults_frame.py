# ruff: noqa: S311
import copy
import random
from typing import Any

import pytest

from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.proxy.faults.discover import load_all
from mcp_chaos.schemas import (
    FaultMode,
    FaultSpec,
    LatencyParams,
    PromptInjectionParams,
    ServerErrorParams,
    TimeoutParams,
)

load_all()

OK: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 7,
    "result": {"content": [{"type": "text", "text": "hi"}]},
}
ERR: dict[str, Any] = {"jsonrpc": "2.0", "id": 7, "error": {"code": -1, "message": "boom"}}


def run(
    mode: FaultMode,
    params: Any,
    resp: dict[str, Any] = OK,
    seed: int = 1,
) -> FaultOutcome:
    spec = (
        FaultSpec(id="f1", mode=mode, params=params)
        if params.mode == mode
        else FaultSpec.model_construct(id="f1", mode=mode, params=params)  # mismatched on purpose
    )
    before = copy.deepcopy(resp)
    out = REGISTRY[mode].apply(FaultContext(spec, random.Random(seed), "t", 0, None), resp)
    assert resp == before, "input mutated"
    return out


def test_timeout_delay() -> None:
    out = run(FaultMode.TIMEOUT, TimeoutParams(delay_s=0.2))
    assert out.delay_s == 0.2 and not out.hang and out.response_obj is None
    assert out.applied is not None and out.applied.detail == {"delay_s": 0.2, "hang": False}


def test_timeout_hang() -> None:
    out = run(FaultMode.TIMEOUT, TimeoutParams(hang=True))
    assert out.hang and out.delay_s == 0.0
    assert out.applied is not None and out.applied.detail["hang"] is True


def test_latency_no_jitter() -> None:
    out = run(FaultMode.LATENCY, LatencyParams(delay_s=0.1))
    assert out.delay_s == 0.1 and out.response_obj is None and not out.hang
    assert out.applied is not None and out.applied.detail == {"delay_s": 0.1}


def test_latency_jitter_deterministic_and_bounded() -> None:
    p = LatencyParams(delay_s=0.1, jitter_s=0.5)
    a, b = run(FaultMode.LATENCY, p, seed=5), run(FaultMode.LATENCY, p, seed=5)
    assert a.delay_s == b.delay_s
    delays = {run(FaultMode.LATENCY, p, seed=s).delay_s for s in range(20)}
    assert len(delays) > 1 and all(0.1 <= d <= 0.6 for d in delays)
    assert all(round(d, 3) == d for d in delays)


def test_server_error_replaces_response() -> None:
    out = run(FaultMode.SERVER_ERROR, ServerErrorParams(code=-32603))
    assert out.response_obj == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32603, "message": "Internal server error"},
    }
    assert out.applied is not None and out.applied.detail == {"code": -32603, "crash": False}


def test_server_error_crash() -> None:
    out = run(FaultMode.SERVER_ERROR, ServerErrorParams(crash=True))
    assert out.crash and out.response_obj is None
    assert out.applied is not None and out.applied.detail == {"code": -32000, "crash": True}


@pytest.mark.parametrize(
    ("mode", "params"),
    [
        (FaultMode.TIMEOUT, TimeoutParams(hang=True)),
        (FaultMode.LATENCY, LatencyParams()),
        (FaultMode.SERVER_ERROR, ServerErrorParams(crash=True)),
    ],
)
def test_noop_on_upstream_error(mode: FaultMode, params: Any) -> None:
    out = run(mode, params, ERR)
    assert out == FaultOutcome() and out.applied is None


@pytest.mark.parametrize("mode", [FaultMode.TIMEOUT, FaultMode.LATENCY, FaultMode.SERVER_ERROR])
def test_noop_on_wrong_params(mode: FaultMode) -> None:
    out = run(mode, PromptInjectionParams(template_id="delegate", position="suffix"))
    assert out == FaultOutcome()
