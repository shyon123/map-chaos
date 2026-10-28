"""In-process relay tests (so coverage sees relay.py) with tiny upstream scripts."""

import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest

from mcp_chaos.proxy.core import ProxyCore
from mcp_chaos.proxy.engine import FaultEngine
from mcp_chaos.proxy.relay import SystemClock, run_relay
from mcp_chaos.schemas import FaultPlan
from tests.unit.proxy_helpers import injection_spec, plan

ECHO_SERVER = """
import sys, json
sys.stderr.write("hello err\\nno newline")
sys.stderr.flush()
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("method") == "tools/call":
        c = [{"type": "text", "text": "ok"}]
        r = {"jsonrpc": "2.0", "id": m["id"], "result": {"content": c}}
        sys.stdout.write(json.dumps(r) + "\\n")
        sys.stdout.flush()
sys.exit(3)
"""


class Out(io.BytesIO):
    def close(self) -> None:  # keep readable after relay
        pass


def make_core(p: FaultPlan) -> ProxyCore:
    return ProxyCore(
        FaultEngine(p, "srv"), run_id="r", agent="a", server="srv", clock=SystemClock()
    )


CALL = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"t","arguments":{}}}\n'


async def run(
    tmp_path: Path, p: FaultPlan, script: str, payload: bytes, delay_close: float = 0.0
) -> tuple[int, bytes, list[dict[str, Any]]]:
    sp = tmp_path / "up.py"
    sp.write_text(script)
    r, w = os.pipe()
    out = Out()
    trace = tmp_path / "sub" / "trace.jsonl"

    def feed() -> None:
        os.write(w, payload)
        if delay_close:
            threading.Event().wait(delay_close)
        os.close(w)

    t = threading.Thread(target=feed)
    t.start()
    try:
        with anyio.fail_after(20):
            code = await run_relay(
                make_core(p), [sys.executable, str(sp)], trace, "srv", stdin_fd=r, out=out
            )
    finally:
        t.join()
        os.close(r)
    evs = [json.loads(x) for x in trace.read_text().splitlines()]
    return code, out.getvalue(), evs


@pytest.mark.anyio
async def test_relay_passthrough_and_exit_code(tmp_path: Path) -> None:
    code, out, evs = await run(tmp_path, plan(), ECHO_SERVER, CALL + b"garbage-no-newline")
    assert code == 3
    assert json.loads(out.splitlines()[0])["result"]["content"][0]["text"] == "ok"
    assert [e["direction"] for e in evs] == ["c2s", "s2c"]


@pytest.mark.anyio
async def test_relay_fault_injection(tmp_path: Path) -> None:
    p = plan(injection_spec("suffix", canary="CANARY-11111111"))
    _, out, evs = await run(tmp_path, p, ECHO_SERVER, CALL)
    assert b"CANARY-11111111" in out
    assert evs[1]["fault"]["mode"] == "prompt_injection"


@pytest.mark.anyio
async def test_relay_delay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_chaos.proxy.faults.base import REGISTRY, FaultOutcome
    from mcp_chaos.schemas import FaultMode

    class Slow:
        def apply(self, ctx: object, response: dict[str, object]) -> FaultOutcome:
            return FaultOutcome(delay_s=0.5)

    monkeypatch.setitem(REGISTRY, FaultMode.PROMPT_INJECTION, Slow())
    _, out, _ = await run(tmp_path, plan(injection_spec()), ECHO_SERVER, CALL)
    assert b'"ok"' in out


@pytest.mark.anyio
async def test_relay_hang_and_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_chaos.proxy.faults.base import REGISTRY, FaultOutcome
    from mcp_chaos.schemas import FaultMode

    class Boom:
        def __init__(self, hang: bool = False, crash: bool = False) -> None:
            self.hang, self.crash = hang, crash

        def apply(self, ctx: object, response: dict[str, object]) -> FaultOutcome:
            return FaultOutcome(hang=self.hang, crash=self.crash)

    monkeypatch.setitem(REGISTRY, FaultMode.PROMPT_INJECTION, Boom(hang=True))
    _, out, evs = await run(tmp_path, plan(injection_spec()), ECHO_SERVER, CALL)
    assert out == b"" and len(evs) == 2

    crash = ECHO_SERVER.replace("sys.exit(3)", "")
    monkeypatch.setitem(REGISTRY, FaultMode.PROMPT_INJECTION, Boom(crash=True))
    code, out, _ = await run(tmp_path, plan(injection_spec()), crash, CALL, delay_close=1.0)
    assert out == b"" and code != 0


@pytest.mark.anyio
async def test_relay_upstream_dies_first_and_oversize(tmp_path: Path) -> None:
    code, _, _ = await run(tmp_path, plan(), "import sys\nsys.exit(5)\n", b"", delay_close=0.5)
    assert code == 5
    big = "import sys\nsys.stdout.write('x'*(65*1024*1024))\nsys.stdout.flush()\n"
    code, _, _ = await run(tmp_path, plan(), big, b"", delay_close=1.0)
    assert isinstance(code, int)
