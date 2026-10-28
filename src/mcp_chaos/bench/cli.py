"""Command-line entrypoint for mcp-chaos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import anyio
from pydantic import ValidationError

from mcp_chaos.log import configure_logging, get_logger


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
    return parser


def _run_proxy(args: argparse.Namespace, upstream: list[str]) -> int:
    from mcp_chaos.proxy.core import ProxyCore
    from mcp_chaos.proxy.engine import FaultEngine, UnsupportedFaultError
    from mcp_chaos.proxy.relay import SystemClock, run_relay
    from mcp_chaos.schemas import FaultPlan

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
