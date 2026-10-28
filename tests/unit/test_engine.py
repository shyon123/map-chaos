import pytest

from mcp_chaos.proxy.engine import Decision, FaultEngine, UnsupportedFaultError, derive_rng
from mcp_chaos.schemas import FaultMode, FaultSpec, ToolVanishParams

from .proxy_helpers import injection_spec, plan


def step(e: FaultEngine, tool: str) -> Decision:
    return e.decide(tool, e.reserve_index(tool))


def test_derive_rng_deterministic_and_distinct() -> None:
    a = derive_rng(1, "s", "t", 0).random()
    assert a == derive_rng(1, "s", "t", 0).random()
    others = {
        derive_rng(2, "s", "t", 0).random(),
        derive_rng(1, "x", "t", 0).random(),
        derive_rng(1, "s", "u", 0).random(),
        derive_rng(1, "s", "t", 1).random(),
    }
    assert a not in others
    # length-prefixing: boundary shifts must not collide
    assert derive_rng(1, "ab", "c", 0).random() != derive_rng(1, "a", "bc", 0).random()


def test_counters_and_calls_membership() -> None:
    e = FaultEngine(plan(injection_spec(calls=[1])), "srv")
    ds = [step(e, "t") for _ in range(3)]
    assert [d.call_index for d in ds] == [0, 1, 2]
    assert [d.spec is not None for d in ds] == [False, True, False]
    assert step(e, "other").call_index == 0


def test_glob_matching() -> None:
    e = FaultEngine(plan(injection_spec(tool="read_*", server="db*")), "dbx")
    assert step(e, "read_rows").spec is not None
    assert step(e, "write_rows").spec is None
    e2 = FaultEngine(plan(injection_spec(server="db*")), "other")
    assert step(e2, "t").spec is None


def test_first_match_wins() -> None:
    e = FaultEngine(plan(injection_spec(fid="a", tool="x"), injection_spec(fid="b")), "s")
    d = step(e, "x")
    assert d.spec is not None and d.spec.id == "a"
    d = step(e, "y")
    assert d.spec is not None and d.spec.id == "b"


def test_probability_extremes_and_determinism() -> None:
    e0 = FaultEngine(plan(injection_spec(probability=0.0)), "s")
    assert all(step(e0, "t").spec is None for _ in range(20))
    e1 = FaultEngine(plan(injection_spec(probability=1.0)), "s")
    assert all(step(e1, "t").spec is not None for _ in range(20))
    pa = [FaultEngine(plan(injection_spec(probability=0.5)), "s") for _ in range(2)]
    seqs = [[step(e, "t").spec is not None for _ in range(50)] for e in pa]
    assert seqs[0] == seqs[1]
    assert 5 < sum(seqs[0]) < 45


def test_unsupported_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_chaos.proxy.faults import base

    monkeypatch.delitem(base.CATALOG_REGISTRY, FaultMode.TOOL_VANISH)
    spec = FaultSpec(id="e", mode=FaultMode.TOOL_VANISH, params=ToolVanishParams())
    with pytest.raises(UnsupportedFaultError, match="tool_vanish"):
        FaultEngine(plan(spec), "s")


def test_every_fault_mode_is_implemented() -> None:
    from mcp_chaos.proxy.faults import base

    assert set(FaultMode) <= set(base.REGISTRY) | set(base.CATALOG_REGISTRY)


def test_decide_with_explicit_out_of_order_index() -> None:
    e = FaultEngine(plan(injection_spec(calls=[1])), "s")
    i0, i1 = e.reserve_index("t"), e.reserve_index("t")
    assert (i0, i1) == (0, 1)
    assert e.decide("t", i1).spec is not None
    assert e.decide("t", i0).spec is None
