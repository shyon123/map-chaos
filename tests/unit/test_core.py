import hashlib
import json
from typing import Any

from mcp_chaos.proxy.core import (
    Action,
    Crash,
    Delay,
    Hang,
    ProxyCore,
    SendToClient,
    SendToServer,
    Trace,
)
from mcp_chaos.proxy.engine import FaultEngine
from mcp_chaos.schemas import TraceEvent

from .proxy_helpers import injection_spec, plan


class FakeClock:
    def __init__(self) -> None:
        self.t = 0

    def now_ms(self) -> int:
        self.t += 10
        return self.t


def make(*specs: Any, allow: frozenset[str] | None = None) -> ProxyCore:
    return ProxyCore(
        FaultEngine(plan(*specs), "srv"),
        run_id="r1",
        agent="a",
        server="srv",
        clock=FakeClock(),
        allow=allow,
    )


def b(obj: Any) -> bytes:
    return json.dumps(obj).encode()


def call_req(i: Any = 1, name: str = "read_rows", args: Any = None) -> bytes:
    return b(
        {
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {"name": name, "arguments": args or {"secret": "hunter2"}},
        }
    )


def call_resp(i: Any = 1, text: str = "hello", **result: Any) -> bytes:
    return b(
        {
            "jsonrpc": "2.0",
            "id": i,
            "result": {"content": [{"type": "text", "text": text}], **result},
        }
    )


def traces(actions: list[Action]) -> list[TraceEvent]:
    return [a.event for a in actions if isinstance(a, Trace)]


def sends(actions: list[Action]) -> list[bytes]:
    return [a.line for a in actions if isinstance(a, SendToClient | SendToServer)]


def test_passthrough_cases() -> None:
    core = make(injection_spec())
    weird = [
        b"not json \xff\xfe",
        b"[1,2,3]",
        b'[{"jsonrpc":"2.0","id":1,"method":"tools/call"}]',
        b'{"jsonrpc":"2.0","method":"notifications/initialized"}',
        b'{"jsonrpc":"2.0",  "id":5,"method":"ping"}\r',
        b"",
        b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":"x"}',
        b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":5}}',
        b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"t"}}',
        b'{"jsonrpc":"2.0","id":true,"method":"tools/call","params":{"name":"t"}}',
    ]
    for line in weird:
        assert core.on_client_line(line) == [SendToServer(line)]
    # server lines with no pending
    for line in [*weird, call_resp(99)]:
        assert core.on_server_line(line) == [SendToClient(line)]


def test_server_passthrough_with_pending() -> None:
    core = make()
    core.on_client_line(call_req(1))
    for line in [b"junk", b'{"id":1,"method":"sampling/x"}', b'{"id":"zz","result":{}}', b"[1]"]:
        assert core.on_server_line(line) == [SendToClient(line)]
    assert core.on_server_line(b'{"result":{}}') == [SendToClient(b'{"result":{}}')]


def test_no_matching_fault_is_byte_identical_with_trace() -> None:
    core = make(injection_spec(calls=[5]))
    core.on_client_line(call_req(1))
    line = b'{"jsonrpc": "2.0",\n "id":1, "result":{"content":[]}}'.replace(b"\n", b" ") + b"\r"
    # trailing \r is valid JSON whitespace
    acts = core.on_server_line(line)
    assert sends(acts) == [line]
    ev = traces(acts)[0]
    assert ev.fault is None and ev.tool == "read_rows" and ev.tool_call_index == 0


def test_fault_applied_and_reserialized() -> None:
    core = make(injection_spec("suffix", canary="CANARY-12345678"))
    core.on_client_line(call_req("abc"))
    acts = core.on_server_line(call_resp("abc", "unicodé ✓"))
    (out,) = sends(acts)
    obj = json.loads(out)
    assert b" " not in out.split(b'"text"')[0]
    assert "unicodé ✓".encode() in out
    assert "CANARY-12345678" in obj["result"]["content"][1]["text"]
    ev = traces(acts)[0]
    assert ev.fault is not None and ev.fault.mode == "prompt_injection"
    assert ev.fault.detail["canary"] == "CANARY-12345678"
    assert isinstance(acts[0], Trace)


def test_int_and_str_ids_distinct() -> None:
    core = make()
    core.on_client_line(call_req(1))
    assert traces(core.on_server_line(call_resp("1"))) == []
    assert len(traces(core.on_server_line(call_resp(1)))) == 1


def test_per_tool_call_index() -> None:
    core = make(injection_spec(calls=[1], tool="read_rows"))
    outs: list[bytes] = []
    for i in range(3):
        core.on_client_line(call_req(i))
        outs.append(sends(core.on_server_line(call_resp(i)))[0])
    assert [b"CANARY" in o for o in outs] == [False, True, False]


def test_error_response_noop_but_counted() -> None:
    core = make(injection_spec(calls=[1]))
    core.on_client_line(call_req(1))
    err = b'{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"x"}}'
    acts = core.on_server_line(err)
    assert sends(acts) == [err]
    assert traces(acts)[0].is_error is True
    core.on_client_line(call_req(2))
    assert b"CANARY" in sends(core.on_server_line(call_resp(2)))[0]


def test_iserror_flag_traced() -> None:
    core = make()
    core.on_client_line(call_req(1))
    assert traces(core.on_server_line(call_resp(1, isError=True)))[0].is_error is True


def test_trace_never_contains_raw_args_or_results() -> None:
    core = make(injection_spec())
    acts = core.on_client_line(call_req(1, args={"secret": "hunter2"}))
    acts += core.on_server_line(call_resp(1, "TOPSECRETRESULT"))
    dumped = "".join(e.model_dump_json() for e in traces(acts))
    assert "hunter2" not in dumped and "TOPSECRETRESULT" not in dumped
    c2s, s2c = traces(acts)
    assert c2s.direction == "c2s" and s2c.direction == "s2c"
    assert c2s.args_sha256 == s2c.args_sha256
    expected = hashlib.sha256(b'{"secret":"hunter2"}').hexdigest()
    assert c2s.args_sha256 == expected
    assert s2c.result_sha256 and s2c.result_bytes and s2c.latency_ms is not None
    assert [c2s.seq, s2c.seq] == [0, 1]


def test_allow_deny() -> None:
    core = make(allow=frozenset({"read_rows"}))
    acts = core.on_client_line(call_req(7, "echo"))
    (reply,) = sends(acts)
    assert SendToServer not in {type(a) for a in acts}
    assert json.loads(reply) == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32001, "message": "tool not allowed by mcp-chaos policy"},
    }
    ev = traces(acts)[0]
    assert ev.method == "tools/call/denied" and ev.tool == "echo" and ev.is_error is True
    # allowed passes
    assert isinstance(core.on_client_line(call_req(8))[-1], SendToServer)
    # denied notification: dropped, no reply
    note = b(
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "echo", "arguments": {}}}
    )
    acts = core.on_client_line(note)
    assert sends(acts) == [] and len(traces(acts)) == 1


def test_tools_list_filtering() -> None:
    core = make(allow=frozenset({"a"}))
    req = b'{"jsonrpc":"2.0","id":3,"method":"tools/list"}'
    assert core.on_client_line(req) == [SendToServer(req)]
    resp = b({"jsonrpc": "2.0", "id": 3, "result": {"tools": [{"name": "a"}, {"name": "b"}, 4]}})
    acts = core.on_server_line(resp)
    assert json.loads(sends(acts)[0])["result"]["tools"] == [{"name": "a"}]
    ev = traces(acts)[0]
    assert ev.method == "tools/list" and ev.direction == "s2c"


def test_tools_list_unfiltered_when_all_allowed_or_no_allow() -> None:
    resp = b({"jsonrpc": "2.0", "id": 3, "result": {"tools": [{"name": "a"}]}})
    for allow in (None, frozenset({"a"})):
        core = make(allow=allow)
        core.on_client_line(b'{"jsonrpc":"2.0","id":3,"method":"tools/list"}')
        assert sends(core.on_server_line(resp)) == [resp]
    core = make(allow=frozenset({"a"}))
    core.on_client_line(b'{"jsonrpc":"2.0","id":3,"method":"tools/list"}')
    odd = b'{"id":3,"error":{"code":1,"message":"x"}}'
    assert sends(core.on_server_line(odd)) == [odd]


def test_delay_hang_crash_actions() -> None:
    from dataclasses import replace

    from mcp_chaos.proxy import core as core_mod
    from mcp_chaos.proxy.faults.base import REGISTRY, FaultOutcome
    from mcp_chaos.schemas import FaultMode

    real = REGISTRY[FaultMode.PROMPT_INJECTION]

    class Fake:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

        def apply(self, ctx: Any, response: dict[str, Any]) -> FaultOutcome:
            return replace(FaultOutcome(), **self.kw)

    try:
        for kw, kind in (
            ({"delay_s": 2.0}, Delay),
            ({"hang": True}, Hang),
            ({"crash": True}, Crash),
        ):
            REGISTRY[FaultMode.PROMPT_INJECTION] = Fake(**kw)
            core = make(injection_spec())
            core.on_client_line(call_req(1))
            acts = core.on_server_line(call_resp(1))
            assert isinstance(acts[0], Trace) and isinstance(acts[1], kind)
            if kind is Delay:
                assert isinstance(acts[1], Delay) and isinstance(acts[1].then, SendToClient)
            assert not any(isinstance(a, SendToClient) for a in acts)
    finally:
        REGISTRY[FaultMode.PROMPT_INJECTION] = real
    assert core_mod


def _denied(acts: list[Action], reply_id: Any) -> None:
    assert not any(isinstance(a, SendToServer) for a in acts)
    (reply,) = sends(acts)
    assert json.loads(reply)["error"]["code"] == -32001
    assert json.loads(reply)["id"] == reply_id
    assert traces(acts)[0].method == "tools/call/denied"


def test_allow_bypass_shapes_denied_with_id() -> None:
    core = make(allow=frozenset({"read_rows"}))
    base = {"jsonrpc": "2.0", "id": 4, "method": "tools/call"}
    for params in ("str", [1], None, {}, {"name": 5}, {"name": None}, {"name": ["read_rows"]}):
        msg = base | ({} if params is None else {"params": params})
        _denied(core.on_client_line(b(msg)), 4)
    # string id, and non-standard id kinds still get a reply
    _denied(core.on_client_line(b(base | {"id": "x", "params": {}})), "x")
    _denied(core.on_client_line(b(base | {"id": 1.5, "params": {}})), 1.5)


def test_allow_bypass_shapes_dropped_without_id() -> None:
    core = make(allow=frozenset({"read_rows"}))
    for params in ("str", {}, {"name": 5}, {"name": "echo"}):
        acts = core.on_client_line(b({"jsonrpc": "2.0", "method": "tools/call", "params": params}))
        assert sends(acts) == [] and len(traces(acts)) == 1


def test_allow_rejects_batch() -> None:
    core = make(allow=frozenset({"read_rows"}))
    line = b'[{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"read_rows"}}]'
    acts = core.on_client_line(line)
    assert sends(acts) == [
        b'{"jsonrpc":"2.0","id":null,"error":{"code":-32600,'
        b'"message":"batch requests not allowed by mcp-chaos policy"}}'
    ]
    assert not any(isinstance(a, SendToServer) for a in acts)
    assert traces(acts)[0].method == "batch/denied"


def test_allow_rejects_duplicate_keys() -> None:
    core = make(allow=frozenset({"read_rows"}))
    nested = (
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call",'
        b'"params":{"name":"read_rows","name":"echo"}}'
    )
    top = b'{"jsonrpc":"2.0","id":7,"id":8,"method":"tools/list"}'
    for line in (nested, top):
        acts = core.on_client_line(line)
        assert not any(isinstance(a, SendToServer) for a in acts)
        assert b'"code":-32600' in sends(acts)[0]
        assert traces(acts)[0].method == "duplicate_keys/denied"


def test_no_allow_passes_duplicate_keys_through() -> None:
    line = b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"a","name":"b"}}'
    acts = make().on_client_line(line)
    assert [a.line for a in acts if isinstance(a, SendToServer)] == [line]


def test_no_allow_keeps_passthrough_for_odd_shapes() -> None:
    core = make()
    for line in (b"[1]", b'{"id":1,"method":"tools/call","params":{"name":5}}', b"7"):
        assert core.on_client_line(line) == [SendToServer(line)]


def test_call_index_assigned_at_request_time() -> None:
    core = make(injection_spec(calls=[0]))
    core.on_client_line(call_req(1))
    core.on_client_line(call_req(2))
    # respond out of order: id 2 (index 1) first, then id 1 (index 0)
    assert b"CANARY" not in sends(core.on_server_line(call_resp(2)))[0]
    assert b"CANARY" in sends(core.on_server_line(call_resp(1)))[0]
