import json
from typing import Any, Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_chaos.proxy.core import ProxyCore, SendToClient
from mcp_chaos.proxy.engine import FaultEngine
from mcp_chaos.schemas import (
    FaultMode,
    FaultSpec,
    SchemaDriftParams,
    ToolVanishParams,
    Trigger,
)
from tests.unit.proxy_helpers import plan


class Clock:
    def now_ms(self) -> int:
        return 0


names = st.sampled_from(["a", "b", "c", "d"])
schema_st = st.fixed_dictionaries(
    {"type": st.just("object"), "properties": st.dictionaries(names, st.just({"type": "string"}))},
    optional={"required": st.lists(names, max_size=3)},
)
tool_st = st.fixed_dictionaries(
    {"name": st.sampled_from(["t1", "t2", "t3", "other"]), "inputSchema": schema_st},
    optional={"description": st.text(max_size=5)},
)
op_st = st.sampled_from(["rename", "drop", "retype", "nest"])


def drift_spec(op: Literal["rename", "drop", "retype", "nest"], path: str) -> FaultSpec:
    return FaultSpec(
        id="d",
        mode=FaultMode.SCHEMA_DRIFT,
        params=SchemaDriftParams(op=op, path=path, layer="catalog"),
        trigger=Trigger(tool="t1"),
    )


def vanish_spec(after: int) -> FaultSpec:
    return FaultSpec(
        id="v",
        mode=FaultMode.TOOL_VANISH,
        params=ToolVanishParams(after_calls=after),
        trigger=Trigger(tool="t2"),
    )


@settings(max_examples=150, deadline=None)
@given(
    tools=st.lists(tool_st, max_size=6),
    op=op_st,
    path=names,
    after=st.integers(0, 2),
    calls=st.lists(st.sampled_from(["t1", "t2", "t3"]), max_size=5),
)
def test_catalog_faults_never_alter_non_matching_tools(
    tools: list[dict[str, Any]], op: Any, path: str, after: int, calls: list[str]
) -> None:
    core = ProxyCore(
        FaultEngine(plan(drift_spec(op, path), vanish_spec(after)), "srv"),
        run_id="r",
        agent="a",
        server="srv",
        clock=Clock(),
    )
    for i, tool in enumerate(calls):
        core.on_client_line(
            json.dumps(
                {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": tool}}
            ).encode()
        )
        core.on_server_line(
            json.dumps({"jsonrpc": "2.0", "id": i, "result": {"content": []}}).encode()
        )
    core.on_client_line(b'{"jsonrpc":"2.0","id":99,"method":"tools/list"}')
    resp = {"jsonrpc": "2.0", "id": 99, "result": {"tools": tools}}
    acts = core.on_server_line(json.dumps(resp).encode())
    (out,) = [a.line for a in acts if isinstance(a, SendToClient)]
    got = json.loads(out)["result"]["tools"]
    assert [t for t in got if t["name"] not in ("t1", "t2")] == [
        t for t in tools if t["name"] not in ("t1", "t2")
    ]
    # matching tools keep their relative order and appear at most as often as before
    assert [t["name"] for t in got if t["name"] in ("t1", "t2")] == [
        t["name"] for t in tools if t["name"] in ("t1", "t2") and not _vanished(t, calls, after)
    ]


def _vanished(tool: dict[str, Any], calls: list[str], after: int) -> bool:
    return tool["name"] == "t2" and calls.count("t2") >= after
