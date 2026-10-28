"""Docker sandbox checks. Opt-in: `uv run pytest -m docker -q` (needs Docker; builds the image)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from mcp_chaos.bench import sandbox
from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.bench.runner import run_one
from mcp_chaos.llm.gateway import GatewayConfig
from mcp_chaos.schemas import FaultPlan, RunConfig, derive_run_id
from mcp_chaos.targets.fixtures import build_fixture

pytestmark = pytest.mark.docker

TAG = "mcpchaos-target:test"
# Colima only shares $HOME with the VM, so bind-mount sources live under the project's .cache.
WORK = Path(__file__).resolve().parents[2] / ".cache" / "docker-tests"


@pytest.fixture(scope="module")
def gw(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GatewayHost]:
    base = tmp_path_factory.mktemp("gw")
    cfg = GatewayConfig(
        mode="replay",
        cache_dir=base / "cache",
        cassette_dir=base / "cassettes",
        global_budget_usd=1.0,
    )
    sandbox.build_image(TAG)
    sandbox.ensure_network()
    with GatewayHost(cfg) as host:
        sandbox.ensure_gw_relay(host.port)
        yield host


@pytest.fixture
def dirs() -> Iterator[tuple[Path, Path, Path]]:
    import shutil
    import tempfile

    WORK.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=WORK))
    scratch, in_dir, out = root / "scratch", root / "in", root / "out"
    for d in (scratch, in_dir, out):
        d.mkdir()
        d.chmod(0o777)
    yield scratch, in_dir, out
    shutil.rmtree(root, ignore_errors=True)


def probe(
    dirs: tuple[Path, Path, Path],
    script: str,
    entrypoint: str = "sh",
    args: list[str] | None = None,
) -> sandbox.ContainerResult:
    scratch, in_dir, out = dirs
    return sandbox.run_container(
        image=TAG,
        scratch=scratch,
        in_dir=in_dir,
        out_dir=out,
        env={
            "ANTHROPIC_BASE_URL": "http://gw:8080/r/probe",
            "ANTHROPIC_API_KEY": "dummy-not-a-key",
        },
        timeout_s=60,
        entrypoint=entrypoint,
        entry_args=args if args is not None else ["-c", script],
    )


def test_container_is_locked_down(gw: GatewayHost, dirs: tuple[Path, Path, Path]) -> None:
    script = (
        "echo uid=$(id -u); "
        "touch /usr/rofs-probe 2>/dev/null; echo rootfs_rc=$?; "
        "touch /tmp/ok; echo tmp_rc=$?; "
        "touch /scratch/probe; echo scratch_rc=$?; "
        "touch /run/in/x 2>/dev/null; echo in_rc=$?; "
        "grep CapEff /proc/self/status; grep NoNewPrivs /proc/self/status"
    )
    r = probe(dirs, script)
    kv = dict(line.split("=", 1) for line in r.stdout_tail.splitlines() if "=" in line)
    assert kv["uid"] == "10001"
    assert kv["rootfs_rc"] != "0"
    assert kv["tmp_rc"] == "0" and kv["scratch_rc"] == "0" and kv["in_rc"] != "0"
    assert "CapEff:\t0000000000000000" in r.stdout_tail
    assert "NoNewPrivs:\t1" in r.stdout_tail
    assert (dirs[0] / "probe").is_file()  # bind-mounted /scratch is visible to the host


def test_network_reaches_gateway_but_not_internet(
    gw: GatewayHost, dirs: tuple[Path, Path, Path]
) -> None:
    code = (
        "import json, urllib.request, urllib.error\n"
        "def hit(url, data=None):\n"
        "    try:\n"
        "        r = urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=8)\n"
        "        return r.status\n"
        "    except urllib.error.HTTPError as e:\n"
        "        return e.code\n"
        "    except Exception as e:\n"
        "        return type(e).__name__\n"
        "print(json.dumps({'gw': hit('http://gw:8080/r/nope/v1/messages', b'{}'),\n"
        "                  'internet': hit('https://example.com'),\n"
        "                  'ip': hit('http://1.1.1.1')}))\n"
    )
    r = probe(dirs, "", entrypoint="python", args=["-c", code])
    res = json.loads(r.stdout_tail.strip().splitlines()[-1])
    assert res["gw"] == 404  # gateway answered ("unknown run")
    assert res["internet"] not in (200, 301, 302, 404)
    assert isinstance(res["internet"], str) and isinstance(res["ip"], str)


def test_self_test_starts_all_servers_and_reaches_gateway(
    gw: GatewayHost, dirs: tuple[Path, Path, Path]
) -> None:
    scratch, _, out = dirs
    build_fixture("T1", scratch)
    for p in [scratch, *scratch.rglob("*")]:
        p.chmod(0o777 if p.is_dir() else 0o666)
    r = probe(
        dirs,
        "",
        entrypoint="python",
        args=["-m", "mcp_chaos.targets.entry", "--self-test", "--out", "/run/out"],
    )
    assert r.exit_code == 0, r.stderr_tail
    report = json.loads((out / "selftest.json").read_text())
    assert report["ok"] is True
    assert report["gateway_status"] == 404
    assert set(report["workers"]) == {"files", "data", "vcs"}
    assert all(w == {"missing": []} for w in report["workers"].values())


def test_full_pipeline_in_docker_without_llm(gw: GatewayHost, tmp_path: Path) -> None:
    """Replay mode with no cassette: the agent's model call is refused by the gateway, so the
    run ends ``agent_error`` -- proving config -> container -> gateway -> result -> check."""
    p = FaultPlan(plan_id="clean", seed=0, faults=[])
    cfg = RunConfig(
        run_id=derive_run_id("langgraph", "T5", p.plan_id, 0, None),
        framework="langgraph",
        task_id="T5",
        fault_plan=p,
        seed=0,
        max_turns=5,
        timeout_s=120,
        budget_usd=0.5,
    )
    r = run_one(
        cfg,
        sandbox="docker",
        gw=gw,
        results_dir=tmp_path / "results",
        image_tag=TAG,
        work_root=WORK,
    )
    assert r.status == "agent_error"
    assert r.check is not None and not r.check.success
    assert r.versions["sandbox"] == "docker" and r.versions["image_id"].startswith("sha256:")
    assert r.side_effects.fs_added == [] and r.side_effects.fs_modified == []


def test_relay_follows_gateway_port(gw: GatewayHost, tmp_path: Path) -> None:
    cfg = GatewayConfig(
        mode="replay",
        cache_dir=tmp_path / "c",
        cassette_dir=tmp_path / "k",
        global_budget_usd=1.0,
    )
    with GatewayHost(cfg) as other:
        sandbox.ensure_gw_relay(other.port)
        assert sandbox.docker(
            "inspect", "-f", '{{index .Config.Labels "mcpchaos.host_target"}}', sandbox.RELAY_NAME
        ).stdout.strip() == str(other.port)
    sandbox.ensure_gw_relay(gw.port)


def test_timeout_kills_container(
    gw: GatewayHost, dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "KILL_GRACE_S", 1.0)
    scratch, in_dir, out = dirs
    r = sandbox.run_container(
        image=TAG,
        scratch=scratch,
        in_dir=in_dir,
        out_dir=out,
        env={},
        timeout_s=2,
        entrypoint="sleep",
        entry_args=["300"],
    )
    assert r.timed_out and r.exit_code is None
