import json
from typing import Any, Literal

import pytest

from mcp_chaos.proxy.core import ProxyCore, SendToClient, SendToServer
from mcp_chaos.proxy.engine import FaultEngine, UnsupportedFaultError
from mcp_chaos.proxy.faults.base import CATALOG_REGISTRY, REGISTRY
from mcp_chaos.schemas import (
    FaultMode,
    FaultSpec,
    SchemaDriftParams,
    ToolVanishParams,
    TraceEvent,
    Trigger,
)

from .proxy_helpers import injection_spec, plan
from .test_core import FakeClock, b, call_req, call_resp, sends, traces

Op = Literal["rename", "drop", "retype", "nest"]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "table": {"type": "string"},
        "limit": {"type": "integer"},
        "opts": {
            "type": "object",
            "properties": {"deep": {"type": "boolean"}},
            "required": ["deep"],
        },
    },
    "required": ["table", "opts"],
}


def tools_resp(i: Any = 1, schema: dict[str, Any] | None = None) -> bytes:
    sch = SCHEMA if schema is None else schema
    tools = [
        {"name": "read_rows", "inputSchema": sch},
        {"name": "echo", "inputSchema": SCHEMA},
    ]
    return b({"jsonrpc": "2.0", "id": i, "result": {"tools": tools}})


def list_req(i: Any = 1) -> bytes:
    return b({"jsonrpc": "2.0", "id": i, "method": "tools/list"})


def drift(
    op: Op, path: str, tool: str = "read_rows", layer: Literal["content", "catalog"] = "catalog"
) -> FaultSpec:
    return FaultSpec(
        id=f"d-{op}",
        mode=FaultMode.SCHEMA_DRIFT,
        params=SchemaDriftParams(op=op, path=path, layer=layer),
        trigger=Trigger(tool=tool),
    )


def vanish(after: int = 2, tool: str = "read_rows") -> FaultSpec:
    return FaultSpec(
        id="v1",
        mode=FaultMode.TOOL_VANISH,
        params=ToolVanishParams(after_calls=after),
        trigger=Trigger(tool=tool),
    )


def make(*specs: FaultSpec, seed: int = 7, allow: frozenset[str] | None = None) -> ProxyCore:
    return ProxyCore(
        FaultEngine(plan(*specs, seed=seed), "srv"),
        run_id="r1",
        agent="a",
        server="srv",
        clock=FakeClock(),
        allow=allow,
    )


def list_tools(core: ProxyCore, schema: dict[str, Any] | None = None) -> tuple[Any, TraceEvent]:
    core.on_client_line(list_req())
    acts = core.on_server_line(tools_resp(1, schema))
    (line,) = sends(acts)
    (ev,) = traces(acts)
    return json.loads(line)["result"]["tools"], ev


def test_registered() -> None:
    assert FaultMode.SCHEMA_DRIFT in CATALOG_REGISTRY
    assert FaultMode.TOOL_VANISH in CATALOG_REGISTRY
    assert FaultMode.TOOL_VANISH not in REGISTRY


def test_unsupported_catalog_spec_layer_aware() -> None:
    FaultEngine(plan(drift("drop", "limit")), "srv")
    if FaultMode.SCHEMA_DRIFT not in REGISTRY:
        with pytest.raises(UnsupportedFaultError):
            FaultEngine(plan(drift("drop", "limit", layer="content")), "srv")


def test_content_layer_not_routed_to_catalog() -> None:
    eng = (
        FaultEngine(plan(drift("drop", "limit", layer="content")), "srv")
        if (FaultMode.SCHEMA_DRIFT in REGISTRY)
        else None
    )
    if eng is not None:
        assert eng.catalog_specs("read_rows") == []
        assert not eng.has_catalog


def test_catalog_specs_match_tool_and_server() -> None:
    eng = FaultEngine(plan(drift("drop", "limit", tool="read_*"), injection_spec()), "srv")
    assert [s.id for s in eng.catalog_specs("read_rows")] == ["d-drop"]
    assert eng.catalog_specs("echo") == []


def test_drop_updates_required() -> None:
    tools, ev = list_tools(make(drift("drop", "table")))
    assert "table" not in tools[0]["inputSchema"]["properties"]
    assert tools[0]["inputSchema"]["required"] == ["opts"]
    assert tools[1]["inputSchema"] == SCHEMA  # non-matching tool untouched
    assert ev.fault is not None
    assert ev.fault.mode == "schema_drift"
    assert ev.fault.detail["tool"] == "read_rows"
    assert ev.fault.parseable is True


def test_rename_updates_required_and_keeps_order() -> None:
    tools, _ = list_tools(make(drift("rename", "table")))
    sch = tools[0]["inputSchema"]
    assert list(sch["properties"]) == ["table_v2", "limit", "opts"]
    assert sch["required"] == ["table_v2", "opts"]


def test_retype() -> None:
    tools, _ = list_tools(make(drift("retype", "limit")))
    assert tools[0]["inputSchema"]["properties"]["limit"]["type"] == "string"


def test_retype_unknown_type_becomes_string() -> None:
    sch = {"type": "object", "properties": {"x": {"type": "null"}}}
    tools, _ = list_tools(make(drift("retype", "x")), sch)
    assert tools[0]["inputSchema"]["properties"]["x"]["type"] == "string"


def test_retype_non_dict_property_noop() -> None:
    sch = {"type": "object", "properties": {"x": True}}
    core = make(drift("retype", "x"))
    core.on_client_line(list_req())
    assert sends(core.on_server_line(tools_resp(1, sch))) == [tools_resp(1, sch)]


def test_nest_keeps_required_consistent() -> None:
    tools, _ = list_tools(make(drift("nest", "table")))
    sch = tools[0]["inputSchema"]
    assert sch["properties"]["table"] == {
        "type": "object",
        "properties": {"table": {"type": "string"}},
        "required": ["table"],
    }
    assert sch["required"] == ["table", "opts"]


def test_nest_optional_not_required() -> None:
    tools, _ = list_tools(make(drift("nest", "limit")))
    assert "required" not in tools[0]["inputSchema"]["properties"]["limit"]


def test_dotted_path() -> None:
    tools, _ = list_tools(make(drift("drop", "opts.deep")))
    opts = tools[0]["inputSchema"]["properties"]["opts"]
    assert opts["properties"] == {}
    assert opts["required"] == []
    assert tools[0]["inputSchema"]["required"] == ["table", "opts"]


@pytest.mark.parametrize("path", ["nope", "a.b", "table.x", "opts.nope"])
def test_missing_path_is_noop_passthrough(path: str) -> None:
    core = make(drift("drop", path))
    core.on_client_line(list_req())
    acts = core.on_server_line(tools_resp())
    assert sends(acts) == [tools_resp()]
    assert traces(acts)[0].fault is None


def test_rename_collision_noop() -> None:
    sch = {"type": "object", "properties": {"a": {}, "a_v2": {}}}
    core = make(drift("rename", "a"))
    core.on_client_line(list_req())
    assert sends(core.on_server_line(tools_resp(1, sch))) == [tools_resp(1, sch)]


def test_tool_without_schema_untouched() -> None:
    core = make(drift("drop", "a"))
    core.on_client_line(list_req())
    resp = b({"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "read_rows"}, 5]}})
    assert sends(core.on_server_line(resp)) == [resp]


def test_multiple_faults_first_traced_rest_in_more() -> None:
    tools, ev = list_tools(make(drift("drop", "table"), drift("retype", "limit")))
    assert "table" not in tools[0]["inputSchema"]["properties"]
    assert tools[0]["inputSchema"]["properties"]["limit"]["type"] == "string"
    assert ev.fault is not None
    assert ev.fault.fault_id == "d-drop"
    more = ev.fault.detail["more"]
    assert isinstance(more, list)
    assert [m["fault_id"] for m in more] == ["d-retype"]  # type: ignore[index,call-overload]


def test_allowlist_first_then_faults() -> None:
    tools, _ = list_tools(make(drift("drop", "table", tool="*"), allow=frozenset({"echo"})))
    assert [t["name"] for t in tools] == ["echo"]
    assert "table" not in tools[0]["inputSchema"]["properties"]


def test_no_catalog_specs_byte_identical() -> None:
    core = make(injection_spec(tool="never-*"))
    core.on_client_line(list_req())
    resp = tools_resp() + b"\r"
    assert sends(core.on_server_line(resp)) == [resp]


def test_catalog_deterministic() -> None:
    def run() -> list[str]:
        core = make(drift("rename", "table"), vanish(1))
        out = [repr(list_tools(core)[0])]
        core.on_client_line(call_req(2, "read_rows"))
        out += [repr(a) for a in core.on_server_line(call_resp(2))]
        out.append(repr(list_tools(core)[0]))
        return out

    assert run() == run()


def test_engine_catalog_rng_deterministic() -> None:
    e1 = FaultEngine(plan(vanish(), seed=3), "srv")
    e2 = FaultEngine(plan(vanish(), seed=3), "srv")
    assert e1.catalog_rng("t").random() == e2.catalog_rng("t").random()
    assert e1.catalog_rng("t").random() != e1.catalog_rng("u").random()


def test_vanish_threshold_and_single_notification() -> None:
    core = make(vanish(2))
    names, _ = list_tools(core)
    assert [t["name"] for t in names] == ["read_rows", "echo"]

    core.on_client_line(call_req(2, "read_rows"))
    a1 = core.on_server_line(call_resp(2))
    assert len(sends(a1)) == 1  # below threshold: no notification
    assert [t["name"] for t in list_tools(core)[0]] == ["read_rows", "echo"]

    core.on_client_line(call_req(3, "read_rows"))
    a2 = core.on_server_line(call_resp(3))
    out = sends(a2)
    assert len(out) == 2
    assert json.loads(out[1]) == {
        "jsonrpc": "2.0",
        "method": "notifications/tools/list_changed",
    }
    assert isinstance(a2[-1], SendToClient)
    note = traces(a2)[-1]
    assert note.method == "notifications/tools/list_changed"
    assert note.fault is not None
    assert note.fault.mode == "tool_vanish"

    # exactly once: a later call to the vanished tool is rejected, no second notification
    tools, ev = list_tools(core)
    assert [t["name"] for t in tools] == ["echo"]
    assert ev.fault is not None
    assert ev.fault.detail["tool"] == "read_rows"


def test_vanished_call_rejected_not_forwarded_and_traced() -> None:
    core = make(vanish(1))
    core.on_client_line(call_req(1, "read_rows"))
    core.on_server_line(call_resp(1))
    acts = core.on_client_line(call_req(2, "read_rows"))
    assert not [a for a in acts if isinstance(a, SendToServer)]
    (reply,) = sends(acts)
    msg = json.loads(reply)
    assert msg["id"] == 2
    assert msg["error"] == {"code": -32602, "message": "Unknown tool: read_rows"}
    (ev,) = traces(acts)
    assert ev.is_error is True
    assert ev.fault is not None
    assert ev.fault.detail["rejected"] is True
    # other tools still forwarded
    assert sends(core.on_client_line(call_req(3, "echo"))) == [call_req(3, "echo")]


def test_vanished_call_without_id_has_no_reply() -> None:
    core = make(vanish(1))
    core.on_client_line(call_req(1, "read_rows"))
    core.on_server_line(call_resp(1))
    note = b({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "read_rows"}})
    acts = core.on_client_line(note)
    assert sends(acts) == []
    assert len(traces(acts)) == 1


def test_vanish_after_zero_hidden_at_first_list_no_notification() -> None:
    core = make(vanish(0))
    assert [t["name"] for t in list_tools(core)[0]] == ["echo"]
    acts = core.on_client_line(call_req(1, "read_rows"))
    assert json.loads(sends(acts)[0])["error"]["code"] == -32602


def test_error_responses_still_count() -> None:
    core = make(vanish(1))
    core.on_client_line(call_req(1, "read_rows"))
    err = b({"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "x"}})
    assert len(sends(core.on_server_line(err))) == 2


def test_vanish_does_not_affect_other_tools_calls() -> None:
    core = make(vanish(1))
    core.on_client_line(call_req(1, "echo"))
    assert len(sends(core.on_server_line(call_resp(1)))) == 1


def test_per_call_fault_still_applies_with_vanish_in_plan() -> None:
    core = make(vanish(5), injection_spec(tool="echo"))
    core.on_client_line(call_req(1, "echo"))
    ev = traces(core.on_server_line(call_resp(1)))[0]
    assert ev.fault is not None
    assert ev.fault.mode == "prompt_injection"
