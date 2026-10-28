import itertools
import json
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mcp_chaos.proxy.core import Action, ProxyCore, SendToClient, SendToServer, Trace
from mcp_chaos.proxy.engine import FaultEngine
from mcp_chaos.proxy.framing import LineSplitter
from tests.unit.proxy_helpers import injection_spec, plan


class Clock:
    def now_ms(self) -> int:
        return 0


def core_for(p: Any, allow: frozenset[str] | None = None) -> ProxyCore:
    return ProxyCore(
        FaultEngine(p, "srv"), run_id="r", agent="a", server="srv", clock=Clock(), allow=allow
    )


json_leaf = st.none() | st.booleans() | st.integers(-5, 5) | st.text(max_size=8)
json_val = st.recursive(
    json_leaf,
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=4), c, max_size=3),
    max_leaves=8,
)
ids = st.integers(0, 4) | st.sampled_from(["a", "b"])
tools = st.sampled_from(["t1", "t2", "other"])


def call_msg(i: Any, tool: str) -> bytes:
    msg = {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": tool}}
    return json.dumps(msg).encode()


@st.composite
def result_msgs(draw: st.DrawFn) -> bytes:
    i = draw(ids)
    kind = draw(st.integers(0, 3))
    if kind == 0:
        msg: Any = {"jsonrpc": "2.0", "id": i, "error": {"code": -1, "message": "x"}}
    else:
        res: dict[str, Any] = {
            "content": [{"type": "text", "text": draw(st.text(max_size=10))}],
            "isError": draw(st.booleans()),
        }
        if kind == 3:
            res["structuredContent"] = draw(json_val)
        msg = {"jsonrpc": "2.0", "id": i, "result": res}
    return json.dumps(msg, ensure_ascii=draw(st.booleans())).encode()


bad_names = st.one_of(
    st.none(), st.integers(), st.lists(st.text(max_size=3), max_size=2), st.just({})
)


@st.composite
def odd_calls(draw: st.DrawFn) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": "tools/call"}
    if draw(st.booleans()):
        msg["id"] = draw(ids)
    shape = draw(st.integers(0, 3))
    if shape == 0:
        msg["params"] = draw(json_val)
    elif shape == 1:
        msg["params"] = {"name": draw(bad_names)}
    elif shape == 2:
        msg["params"] = {"arguments": {"a": 1}}  # missing name
    return json.dumps(msg).encode()


def with_eol(line: bytes, eol: str) -> bytes:
    # lines handed to the core exclude "\n" but keep any trailing "\r"
    return line + (b"\r" if eol == "crlf" else b"")


noise = st.builds(
    with_eol,
    st.one_of(
        st.binary(max_size=30).filter(lambda b: b"\n" not in b),
        st.builds(lambda v: json.dumps(v).encode(), json_val),
        st.builds(
            lambda i, m: json.dumps({"jsonrpc": "2.0", "id": i, "method": m}).encode(),
            ids,
            st.text(max_size=5) | st.just("sampling/createMessage") | st.just("tools/call"),
        ),
        st.builds(
            lambda m: json.dumps({"jsonrpc": "2.0", "method": m}).encode(), st.text(max_size=5)
        ),
        st.builds(lambda i: json.dumps([{"id": i, "method": "tools/call"}]).encode(), ids),
        st.builds(call_msg, ids, tools),
        odd_calls(),
        result_msgs(),
    ),
    st.sampled_from(["lf", "crlf"]),
)
lines_st = st.lists(noise, max_size=15)


def to_client(actions: list[Action]) -> list[bytes]:
    return [a.line for a in actions if isinstance(a, SendToClient)]


def to_server(actions: list[Action]) -> list[bytes]:
    return [a.line for a in actions if isinstance(a, SendToServer)]


@settings(max_examples=200, deadline=None)
@given(lines=lines_st)
def test_p2_passthrough(lines: list[bytes]) -> None:
    # triggers never match: tool glob matches nothing we generate
    core = core_for(plan(injection_spec(tool="never-*")))
    out: list[bytes] = []
    for line in lines:
        # interleave requests to build pending state, then feed to the chosen side
        acts = core.on_client_line(line)
        out += to_server(acts)
    assert out == lines
    out = []
    for line in lines:
        out += to_client(core.on_server_line(line))
    assert out == lines


@settings(max_examples=150, deadline=None)
@given(
    traffic=st.lists(st.tuples(st.booleans(), noise), max_size=20),
    seed=st.integers(0, 10),
    prob=st.sampled_from([0.3, 0.7, 1.0]),
)
def test_p1_determinism(traffic: list[tuple[bool, bytes]], seed: int, prob: float) -> None:
    p = plan(
        injection_spec("field", probability=prob),
        injection_spec("prefix", fid="f2", tool="t2"),
        seed=seed,
    )

    def run() -> tuple[list[bytes], list[str]]:
        core = core_for(p)
        out: list[bytes] = []
        faults: list[str] = []
        for from_client, line in traffic:
            acts = core.on_client_line(line) if from_client else core.on_server_line(line)
            out += to_client(acts) + to_server(acts)
            for a in acts:
                if isinstance(a, Trace):
                    dumped = a.event.model_dump(exclude={"t_ms"})
                    faults.append(json.dumps(dumped, sort_keys=True))
        return out, faults

    assert run() == run()


@settings(max_examples=200, deadline=None)
@given(
    data=st.binary(max_size=200),
    cuts=st.lists(st.integers(0, 200), max_size=8),
)
def test_p4_framing(data: bytes, cuts: list[int]) -> None:
    whole = LineSplitter()
    expect = whole.feed(data)
    tail = whole.flush()
    pts = sorted({min(c, len(data)) for c in cuts} | {0, len(data)})
    s = LineSplitter()
    got: list[bytes] = []
    for a, b in itertools.pairwise(pts):
        got += s.feed(data[a:b])
    assert got == expect
    assert s.flush() == tail
    rebuilt = b"".join(line + b"\n" for line in expect) + (tail or b"")
    assert rebuilt == data


@settings(max_examples=150, deadline=None)
@given(lines=lines_st)
def test_allowlist_never_forwards_disallowed(lines: list[bytes]) -> None:
    allow = frozenset({"t1"})
    core = core_for(plan(injection_spec(tool="never-*")), allow)
    for line in lines:
        for fwd in to_server(core.on_client_line(line)):
            try:
                msg = json.loads(fwd)
            except ValueError:
                continue  # non-JSON junk is forwarded verbatim by design
            assert not isinstance(msg, list)
            if isinstance(msg, dict) and msg.get("method") == "tools/call":
                params = msg.get("params")
                assert isinstance(params, dict) and params.get("name") in allow


@settings(max_examples=100, deadline=None)
@given(lines=lines_st)
def test_allowlist_junk_is_not_special(lines: list[bytes]) -> None:
    # non-JSON lines are still forwarded verbatim even with an allowlist
    core = core_for(plan(), frozenset({"t1"}))
    for line in lines:
        try:
            json.loads(line)
        except ValueError:
            assert core.on_client_line(line) == [SendToServer(line)]


@settings(max_examples=100, deadline=None)
@given(
    lines=lines_st,
    cuts=st.lists(st.integers(0, 4000), max_size=6),
    final_newline=st.booleans(),
)
def test_p2_stream_with_partial_final_line(
    lines: list[bytes], cuts: list[int], final_newline: bool
) -> None:
    lines = [x for x in lines if b"\n" not in x]
    assume(final_newline or not lines or lines[-1] != b"")  # empty tail == no tail
    data = b"\n".join(lines) + (b"\n" if final_newline and lines else b"")
    pts = sorted({min(c, len(data)) for c in cuts} | {0, len(data)})
    core = core_for(plan(injection_spec(tool="never-*")))
    sp = LineSplitter()
    out: list[bytes] = []
    for a, b in itertools.pairwise(pts):
        for line in sp.feed(data[a:b]):
            out += to_server(core.on_client_line(line))
    tail = sp.flush()
    if tail is not None:
        out += to_server(core.on_client_line(tail))
    assert out == lines


@settings(max_examples=5, deadline=None)
@given(pad=st.integers(65_536, 200_000), eol=st.sampled_from([b"", b"\r"]))
def test_p2_long_lines(pad: int, eol: bytes) -> None:
    core = core_for(plan(injection_spec()))
    text = "x" * pad
    req = call_msg(1, "t1") + eol
    resp = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}}
    ).encode()
    # faulted response is re-serialized (valid JSON), unfaulted would be byte-identical
    assert to_server(core.on_client_line(req)) == [req]
    (out,) = to_client(core.on_server_line(resp))
    assert json.loads(out)["result"]["content"][0]["text"] == text
    plain = core_for(plan(injection_spec(tool="never-*")))
    plain.on_client_line(req)
    assert to_client(plain.on_server_line(resp + eol)) == [resp + eol]


@settings(max_examples=150, deadline=None)
@given(
    n=st.integers(2, 8),
    perm_seed=st.randoms(use_true_random=False),
    seed=st.integers(0, 20),
)
def test_interleaving_independence(n: int, perm_seed: Any, seed: int) -> None:
    p = plan(injection_spec("suffix", probability=0.5), seed=seed)

    def run(order: list[int]) -> dict[int, str]:
        core = core_for(p)
        for i in range(n):
            core.on_client_line(call_msg(i, "t1"))
        mapping: dict[int, str] = {}
        for i in order:
            resp = json.dumps(
                {"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "x"}]}}
            ).encode()
            for a in core.on_server_line(resp):
                if isinstance(a, Trace) and a.event.tool_call_index is not None:
                    assert a.event.tool_call_index == i  # index fixed at request time
                    mapping[a.event.tool_call_index] = json.dumps(
                        a.event.fault.model_dump() if a.event.fault else None, sort_keys=True
                    )
        return mapping

    order = list(range(n))
    shuffled = order[:]
    perm_seed.shuffle(shuffled)
    assert run(order) == run(shuffled)
