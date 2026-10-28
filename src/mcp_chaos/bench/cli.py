"""Command-line entrypoint for mcp-chaos."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from pydantic import ValidationError

from mcp_chaos.log import configure_logging, get_logger
from mcp_chaos.schemas.faults import FaultPlan
from mcp_chaos.schemas.runs import Framework, RunConfig, derive_run_id

if TYPE_CHECKING:
    from mcp_chaos.schemas.results import RunResult

BASELINE_PER_RUN_USD = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-chaos")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("proxy", help="fault-injecting stdio proxy (append: -- <upstream cmd...>)")
    p.add_argument("--agent", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", choices=["fault", "guard"], default="fault")
    p.add_argument("--allow", default=None, help="comma-separated tool allowlist")
    p.add_argument("--trace-out", required=True, type=Path)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--framework", choices=["langgraph", "agentsdk"], default="langgraph")
        sp.add_argument("--sandbox", choices=["docker", "local"], default="docker")
        sp.add_argument("--mode", choices=["live", "record", "replay"], required=True)
        sp.add_argument("--budget-usd", type=float, required=True)
        sp.add_argument("--bench-id", required=True)

    r = sub.add_parser("run-one", help="run one benchmark run")
    r.add_argument("--task", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--plan", type=Path, default=None)
    common(r)
    b = sub.add_parser("baseline", help="clean (no-fault) baseline over tasks x seeds")
    b.add_argument("--tasks", default="all")
    b.add_argument("--seeds", default="0,1,2")
    common(b)
    return parser


def _run_proxy(args: argparse.Namespace, upstream: list[str]) -> int:
    from mcp_chaos.proxy.core import ProxyCore
    from mcp_chaos.proxy.engine import FaultEngine, UnsupportedFaultError
    from mcp_chaos.proxy.relay import SystemClock, run_relay

    if args.mode == "guard":
        sys.stderr.write("mcp-chaos: --mode guard is not implemented yet\n")
        return 2
    if not upstream:
        sys.stderr.write("mcp-chaos: missing upstream command after --\n")
        return 2
    configure_logging()
    log = get_logger(__name__)
    try:
        plan = FaultPlan.model_validate_json(Path(args.plan).read_text(encoding="utf-8"))
        engine = FaultEngine(plan, args.server)
    except (OSError, ValidationError, UnsupportedFaultError) as e:
        sys.stderr.write(f"mcp-chaos: {e}\n")
        return 2
    allow = (
        frozenset(t for t in str(args.allow).split(",") if t) if args.allow is not None else None
    )
    core = ProxyCore(
        engine,
        run_id=args.run_id,
        agent=args.agent,
        server=args.server,
        clock=SystemClock(),
        allow=allow,
    )
    log.info("proxy_start", server=args.server, plan_id=plan.plan_id, run_id=args.run_id)
    return anyio.run(run_relay, core, upstream, Path(args.trace_out), args.server)


def _load_env_file(path: Path) -> None:
    """Tiny .env parser: KEY=VALUE lines; existing environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.removeprefix("export ").strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)


def _make_cfg(
    framework: Framework, task: str, seed: int, plan: FaultPlan, per_run_usd: float
) -> RunConfig:
    return RunConfig(
        run_id=derive_run_id(framework, task, plan.plan_id, seed, None),
        framework=framework,
        task_id=task,
        fault_plan=plan,
        seed=seed,
        budget_usd=per_run_usd,
    )


def _execute(args: argparse.Namespace, cfgs: list[RunConfig], global_usd: float) -> int:
    """Start gateway (+ docker plumbing), run every cfg in order, print a summary."""
    from mcp_chaos.bench import sandbox
    from mcp_chaos.bench.gateway_host import GatewayHost
    from mcp_chaos.bench.runner import RunnerError, run_one
    from mcp_chaos.llm.gateway import GatewayConfig, GatewayConfigError

    _load_env_file(Path(".env"))
    if args.mode != "replay" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write("mcp-chaos: ANTHROPIC_API_KEY is required for live/record mode\n")
        return 2
    configure_logging()
    results_dir = Path("results") / args.bench_id
    gcfg = GatewayConfig(
        mode=args.mode,
        cache_dir=Path(".cache") / "llm",
        cassette_dir=Path("cassettes") / args.bench_id,
        global_budget_usd=global_usd,
    )
    try:
        host = GatewayHost(gcfg)
    except GatewayConfigError as e:
        sys.stderr.write(f"mcp-chaos: {e}\n")
        return 2
    results: list[RunResult] = []
    code = 0
    with host:
        try:
            if args.sandbox == "docker":
                sandbox.build_image(sandbox.DEFAULT_IMAGE)
                sandbox.ensure_gw_relay(host.port)
            for cfg in cfgs:
                r = run_one(cfg, sandbox=args.sandbox, gw=host, results_dir=results_dir)
                results.append(r)
                if r.status == "budget_abort":
                    sys.stderr.write("mcp-chaos: budget exceeded, stopping\n")
                    code = 3
                    break
        except (RunnerError, sandbox.SandboxError) as e:
            sys.stderr.write(f"mcp-chaos: {e}\n")
            code = 1
    _print_table(results)
    total = sum(r.usage.cost_usd for r in results)
    print(f"total cost: ${total:.4f}  results: {results_dir / 'runs.jsonl'}")
    return code


def _print_table(results: list[RunResult]) -> None:
    tasks_seen = sorted({r.config.task_id for r in results})
    seeds = sorted({r.config.seed for r in results})
    by = {(r.config.task_id, r.config.seed): r for r in results}
    print("task  " + "  ".join(f"seed{s:<4}" for s in seeds))
    for t in tasks_seen:
        cells: list[str] = []
        for s in seeds:
            r = by.get((t, s))
            if r is None:
                cell = "-"
            elif r.status in ("completed", "agent_error") and r.check is not None:
                cell = "PASS" if r.check.success else "FAIL"
                if r.status == "agent_error":
                    cell += "*"
            else:
                cell = r.status
            cells.append(f"{cell:<8}")
        print(f"{t:<6}" + "  ".join(cells))


def _cmd_run_one(args: argparse.Namespace) -> int:
    if args.plan is not None:
        try:
            plan = FaultPlan.model_validate_json(Path(args.plan).read_text(encoding="utf-8"))
        except (OSError, ValidationError) as e:
            sys.stderr.write(f"mcp-chaos: {e}\n")
            return 2
    else:
        plan = FaultPlan(plan_id="clean", seed=args.seed, faults=[])
    cfg = _make_cfg(args.framework, args.task, args.seed, plan, args.budget_usd)
    return _execute(args, [cfg], args.budget_usd)


def _cmd_baseline(args: argparse.Namespace) -> int:
    from mcp_chaos.targets.fixtures import TASK_IDS

    tasks = list(TASK_IDS) if args.tasks == "all" else [t for t in args.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in TASK_IDS]
    if unknown:
        sys.stderr.write(f"mcp-chaos: unknown task(s): {','.join(unknown)}\n")
        return 2
    seeds = [int(s) for s in args.seeds.split(",") if s]
    cfgs = [
        _make_cfg(
            args.framework,
            t,
            s,
            FaultPlan(plan_id="clean", seed=s, faults=[]),
            min(BASELINE_PER_RUN_USD, args.budget_usd),
        )
        for t in tasks
        for s in seeds
    ]
    return _execute(args, cfgs, args.budget_usd)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    upstream: list[str] = []
    if "--" in raw:
        i = raw.index("--")
        raw, upstream = raw[:i], raw[i + 1 :]
    parser = build_parser()
    args = parser.parse_args(raw)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "proxy":
        return _run_proxy(args, upstream)
    if args.command == "run-one":
        return _cmd_run_one(args)
    if args.command == "baseline":
        return _cmd_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
