"""Real processes: SDK client -> guard -> fault proxy -> fake server (ADR 0007)."""

import json
import statistics
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import pytest
from mcp import Client, StdioServerParameters

from mcp_chaos.harden.config import READ_ONLY_TOOLS, GuardConfig
from mcp_chaos.proxy.faults import discover
from mcp_chaos.proxy.faults.base import REGISTRY
from mcp_chaos.schemas import FaultMode, FaultPlan, FaultSpec
from mcp_chaos.schemas.runs import HardeningConfig
from mcp_chaos.targets.workers import WorkerSpec, proxy_cmd

from .conftest import fake_server_cmd

discover.load_all()
CANARY = "CANARY-0badf00d"
FAKE_WORKER = WorkerSpec(
    name="files", server_id="fake", allow=frozenset({"read_rows", "echo", "get_doc", "flaky"})
)


def fault(mode: str, params: dict[str, Any], tool: str, calls: list[int] | None = None) -> Any:
    return FaultSpec.model_validate(
        {
            "id": f"F-{mode}",
            "mode": mode,
            "params": {"mode": mode, **params},
            "trigger": {"tool": tool, "calls": calls},
        }
    )


def needs(mode: FaultMode) -> pytest.MarkDecorator:
    return pytest.mark.skipif(mode not in REGISTRY, reason=f"{mode.value} not implemented yet")


def guard_cfg(**kw: Any) -> GuardConfig:
    h = {k: kw.pop(k) for k in list(kw) if k in HardeningConfig.model_fields}
    kw.setdefault("retry_tools", frozenset({"read_rows", "get_doc", "flaky"}) | READ_ONLY_TOOLS)
    return GuardConfig(hardening=HardeningConfig(**h), **kw)


class Chain:
    def __init__(self, tmp: Path, script: Path, specs: list[Any], cfg: GuardConfig | None) -> None:
        plan = FaultPlan(plan_id="p", seed=7, faults=specs)
        self.plan_path = tmp / "plan.json"
        self.plan_path.write_text(plan.model_dump_json())
        self.tmp = tmp
        self.script = script
        self.cfg = cfg
        self.trace_dir = tmp / "traces"

    @property
    def guard_trace(self) -> Path:
        return self.trace_dir / "files-fake.guard.jsonl"

    @property
    def fault_trace(self) -> Path:
        return self.trace_dir / "files-fake.jsonl"

    def guard_events(self) -> list[dict[str, Any]]:
        return [json.loads(x) for x in self.guard_trace.read_text().splitlines()]

    def fault_events(self) -> list[dict[str, Any]]:
        return [json.loads(x) for x in self.fault_trace.read_text().splitlines()]

    def actions(self) -> list[str]:
        return [e["detail"]["guard_action"] for e in self.guard_events()]

    def command(self) -> list[str]:
        cmd = proxy_cmd(
            FAKE_WORKER,
            run_id="r",
            plan_path=self.plan_path,
            trace_dir=self.trace_dir,
            upstream=fake_server_cmd(self.script),
            hardening=self.cfg.hardening if self.cfg else None,
            seed=3,
        )
        if self.cfg is None:
            return cmd
        # same chain proxy_cmd builds, but with this test's timing parameters
        i = cmd.index("--guard-config")
        cmd[i + 1] = self.cfg.model_dump_json()
        return cmd

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Client]:
        cmd = self.command()
        async with Client(StdioServerParameters(command=cmd[0], args=cmd[1:])) as c:
            yield c


def texts(res: types.CallToolResult) -> list[str]:
    return [b.text for b in res.content if isinstance(b, types.TextContent)]


@pytest.fixture
def script(fake_script_path: Path) -> Path:
    return fake_script_path


@pytest.mark.anyio
async def test_clean_calls_unchanged_except_spotlight(script: Path, tmp_path: Path) -> None:
    ch = Chain(tmp_path, script, [], guard_cfg(retry=True, schema_validation=True))
    with anyio.fail_after(30):
        async with ch.client() as c:
            tools = await c.list_tools()
            res = await c.call_tool("read_rows", {})
            doc = await c.call_tool("get_doc", {"id": "1"})
    assert {t.name for t in tools.tools} >= {"read_rows", "echo"}
    assert res.structured_content == {
        "rows": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        "count": 2,
    }
    assert "quick brown fox" in texts(doc)[0] and not ch.guard_trace.read_text().strip()


@pytest.mark.anyio
@needs(FaultMode.TIMEOUT)
async def test_f01_hang_recovered_by_retry(script: Path, tmp_path: Path) -> None:
    specs = [fault("timeout", {"delay_s": 2.0, "hang": False}, "read_rows", [0])]
    ch = Chain(tmp_path, script, specs, guard_cfg(retry=True, deadline_s=0.7))
    with anyio.fail_after(60):
        async with ch.client() as c:
            res = await c.call_tool("read_rows", {})
    assert not res.is_error and res.structured_content is not None
    assert ch.actions() == ["retried"]
    assert [e["detail"]["kind"] for e in ch.guard_events()] == ["deadline"]
    assert sum(1 for e in ch.fault_events() if e["fault"]) == 1  # retry (call 1) was clean


@pytest.mark.anyio
async def test_f04_frame_recovered(script: Path, tmp_path: Path) -> None:
    params = {"variant": "truncate", "layer": "frame"}
    specs = [fault("malformed_json", params, "read_rows", [0])]
    ch = Chain(tmp_path, script, specs, guard_cfg(retry=True, schema_validation=True))
    with anyio.fail_after(60):
        async with ch.client() as c:
            res = await c.call_tool("read_rows", {})
    assert not res.is_error and res.structured_content is not None
    assert "retried" in ch.actions()


@pytest.mark.anyio
async def test_f04_content_recovered_by_validation(script: Path, tmp_path: Path) -> None:
    params = {"variant": "truncate", "layer": "content"}
    specs = [fault("malformed_json", params, "read_rows", [0])]
    ch = Chain(tmp_path, script, specs, guard_cfg(retry=True, schema_validation=True))
    with anyio.fail_after(60):
        async with ch.client() as c:
            res = await c.call_tool("read_rows", {})
    assert not res.is_error and res.structured_content is not None
    assert ch.actions()[:2] == ["validated_fail", "retried"] or "retried" in ch.actions()


@pytest.mark.anyio
@needs(FaultMode.SERVER_ERROR)
async def test_f03_server_error_retried(script: Path, tmp_path: Path) -> None:
    specs = [fault("server_error", {"code": -32000, "crash": False}, "read_rows", [0])]
    ch = Chain(tmp_path, script, specs, guard_cfg(retry=True))
    with anyio.fail_after(60):
        async with ch.client() as c:
            res = await c.call_tool("read_rows", {})
    assert not res.is_error and res.structured_content is not None
    assert [e["detail"]["kind"] for e in ch.guard_events()] == ["rpc_error"]


@pytest.mark.anyio
async def test_f05_drift_caught_by_validation(script: Path, tmp_path: Path) -> None:
    specs = [
        fault(
            "schema_drift", {"op": "rename", "path": "rows", "layer": "content"}, "read_rows", [0]
        )
    ]
    # validation + retry: the retry (call 1) is clean
    ch = Chain(tmp_path, script, specs, guard_cfg(retry=True, schema_validation=True))
    with anyio.fail_after(60):
        async with ch.client() as c:
            await c.list_tools()  # the guard learns outputSchema from tools/list
            res = await c.call_tool("read_rows", {})
    assert not res.is_error and res.structured_content is not None
    assert "rows" in res.structured_content
    assert "validated_fail" in ch.actions()
    # validation alone: the drifted result becomes an explicit error
    d2 = tmp_path / "v"
    d2.mkdir()
    ch2 = Chain(d2, script, specs, guard_cfg(schema_validation=True))
    with anyio.fail_after(60):
        async with ch2.client() as c:
            await c.list_tools()
            res2 = await c.call_tool("read_rows", {})
    assert res2.is_error and "validation_failed" in texts(res2)[0]


@pytest.mark.anyio
async def test_f08_canary_quarantined(script: Path, tmp_path: Path) -> None:
    params = {"template_id": "echo_in_answer", "position": "suffix", "canary": CANARY}
    specs = [fault("prompt_injection", params, "get_doc", [0])]
    ch = Chain(tmp_path, script, specs, guard_cfg(injection_filter=True))
    with anyio.fail_after(60):
        async with ch.client() as c:
            res = await c.call_tool("get_doc", {"id": "1"})
    joined = "\n".join(texts(res))
    assert CANARY not in joined
    assert "[mcp-chaos guard: removed" in joined
    assert joined.count('<tool_output trust="untrusted">') == 2
    assert "quick brown fox" in joined  # legitimate content survives
    assert ch.actions() == ["injection_flagged"]
    assert any(CANARY in json.dumps(e["fault"]) for e in ch.fault_events() if e["fault"])


@pytest.mark.anyio
async def test_f07_truncated(script: Path, tmp_path: Path) -> None:
    specs = [fault("oversized_output", {"bytes": 300_000, "filler": "lorem"}, "get_doc", [0])]
    ch = Chain(tmp_path, script, specs, guard_cfg(size_cap=True, size_cap_bytes=2000))
    with anyio.fail_after(60):
        async with ch.client() as c:
            res = await c.call_tool("get_doc", {"id": "1"})
    joined = "\n".join(texts(res))
    assert len(joined.encode()) < 4000
    assert "[mcp-chaos guard: truncated" in joined
    assert ch.actions() == ["truncated"]


@pytest.mark.anyio
async def test_breaker_opens_after_repeated_failures(script: Path, tmp_path: Path) -> None:
    params = {"variant": "truncate", "layer": "frame"}
    specs = [fault("malformed_json", params, "read_rows")]  # every call fails
    cfg = guard_cfg(retry=True, circuit_breaker=True, max_retries=5, breaker_threshold=3)
    ch = Chain(tmp_path, script, specs, cfg)
    with anyio.fail_after(60):
        async with ch.client() as c:
            first = await c.call_tool("read_rows", {})
            second = await c.call_tool("read_rows", {})
            other = await c.call_tool("get_doc", {"id": "1"})
    for r in (first, second):
        assert r.is_error and texts(r) == ["tool temporarily unavailable (circuit open)"]
    assert not other.is_error  # breakers are per tool
    assert ch.actions().count("breaker_open") == 2
    assert sum(1 for e in ch.fault_events() if e["fault"]) == 3  # no 4th upstream attempt


@pytest.mark.anyio
async def test_guard_latency_overhead(script: Path, tmp_path: Path) -> None:
    """Measured, not gated: p50 per-call overhead of the guard on clean calls."""
    n = 40

    async def run(ch: Chain) -> float:
        async with ch.client() as c:
            for _ in range(5):
                await c.call_tool("echo", {"a": 1})
            samples: list[float] = []
            for _ in range(n):
                t0 = time.perf_counter()
                await c.call_tool("echo", {"a": 1})
                samples.append((time.perf_counter() - t0) * 1000)
        return statistics.median(samples)

    plain_dir, guarded_dir = tmp_path / "p", tmp_path / "g"
    plain_dir.mkdir()
    guarded_dir.mkdir()
    full = guard_cfg(
        schema_validation=True,
        retry=True,
        circuit_breaker=True,
        injection_filter=True,
        size_cap=True,
    )
    with anyio.fail_after(120):
        base = await run(Chain(plain_dir, script, [], None))
        hard = await run(Chain(guarded_dir, script, [], full))
    sys.stderr.write(
        f"\nguard overhead p50: base={base:.2f}ms guarded={hard:.2f}ms delta={hard - base:.2f}ms\n"
    )
    assert base > 0 and hard > 0


def test_proxy_cmd_builds_guard_chain(tmp_path: Path) -> None:
    up = ["server", "--x"]
    kw: dict[str, Any] = {
        "run_id": "r1",
        "plan_path": tmp_path / "plan.json",
        "trace_dir": tmp_path,
        "upstream": up,
    }
    plain = proxy_cmd(FAKE_WORKER, **kw)
    assert "--mode" not in plain
    # prompt addendum alone needs no guard
    assert proxy_cmd(FAKE_WORKER, hardening=HardeningConfig(prompt_addendum=True), **kw) == plain
    assert proxy_cmd(FAKE_WORKER, hardening=None, **kw) == plain
    h = HardeningConfig(retry=True, prompt_addendum=True)
    cmd = proxy_cmd(FAKE_WORKER, hardening=h, seed=9, **kw)
    sep = cmd.index("--")
    assert cmd[sep + 1 :] == plain
    assert cmd[cmd.index("--mode") + 1] == "guard"
    parsed = GuardConfig.model_validate_json(cmd[cmd.index("--guard-config") + 1])
    assert parsed.hardening == h and parsed.seed == 9
    assert cmd[cmd.index("--trace-out") + 1].endswith("files-fake.guard.jsonl")
