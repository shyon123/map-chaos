"""``mcp-chaos judge | label-sample | kappa`` subcommands."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mcp_chaos.judge.kappa import (
    compute_kappas,
    read_human_csv,
    stratified_sample,
    write_sample_csv,
)
from mcp_chaos.judge.labels import LABELS_FILE, JudgeMode, load_records, save_records
from mcp_chaos.llm.prices import UnknownModelError
from mcp_chaos.schemas.results import RunResult
from mcp_chaos.schemas.trace import TraceEvent

DUMMY_KEY = "dummy-not-a-key"


def register(sub: Any) -> None:
    j = sub.add_parser("judge", help="label a bench's runs (deterministic + LLM judge)")
    j.add_argument("--bench-id", required=True)
    j.add_argument("--mode", choices=["offline", "sync", "batch"], default="offline")
    j.add_argument("--budget-usd", type=float, default=5.0)
    j.add_argument("--base-url", default=None, help="Anthropic API base URL (e.g. a fake)")
    j.add_argument("--model", default="claude-sonnet-5-5")
    j.set_defaults(handler=cmd_judge)
    s = sub.add_parser("label-sample", help="write a stratified CSV for human labeling")
    s.add_argument("--bench-id", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(handler=cmd_label_sample)
    k = sub.add_parser("kappa", help="Cohen's kappa (judge vs human) with bootstrap 95%% CI")
    k.add_argument("--bench-id", required=True)
    k.add_argument("--human", required=True, type=Path)
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(handler=cmd_kappa)


def _results_dir(bench_id: str) -> Path:
    return Path("results") / bench_id


def load_traces(results_dir: Path, run_id: str) -> list[TraceEvent]:
    files = sorted((results_dir / "traces" / run_id).glob("*.jsonl"))
    single = results_dir / "traces" / f"{run_id}.jsonl"
    if single.is_file():
        files.append(single)
    events: list[TraceEvent] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                with contextlib.suppress(ValidationError):
                    events.append(TraceEvent.model_validate_json(line))
    events.sort(key=lambda e: (e.t_ms, e.agent, e.seq))
    return events


def _load_runs(bench_id: str) -> dict[str, RunResult]:
    from mcp_chaos.bench.runner import RUNS_FILE, load_runs

    return load_runs(_results_dir(bench_id) / RUNS_FILE)


def cmd_judge(args: argparse.Namespace) -> int:
    from mcp_chaos.judge.llm_judge import Judge

    runs = _load_runs(args.bench_id)
    if not runs:
        sys.stderr.write(f"mcp-chaos: no runs found for bench {args.bench_id}\n")
        return 2
    mode: JudgeMode = args.mode
    client: Any = None
    if mode != "offline":
        from anthropic import AsyncAnthropic

        key = os.environ.get("ANTHROPIC_API_KEY") or (DUMMY_KEY if args.base_url else None)
        if key is None:
            sys.stderr.write("mcp-chaos: ANTHROPIC_API_KEY is required for sync/batch mode\n")
            return 2
        client = AsyncAnthropic(api_key=key, base_url=args.base_url, max_retries=2)
    rd = _results_dir(args.bench_id)
    inputs = [(r, load_traces(rd, r.run_id)) for r in runs.values()]
    try:
        judge = Judge(
            client,
            model=args.model,
            mode=mode,
            budget_usd=args.budget_usd,
            cache_dir=Path(".cache") / "judge",
        )
    except UnknownModelError as e:
        sys.stderr.write(f"mcp-chaos: {e}\n")
        return 2
    new = asyncio.run(judge.judge(inputs))
    path = rd / LABELS_FILE
    merged = load_records(path)
    merged.update(new)
    save_records(path, merged)
    counts: dict[str, int] = {}
    for r in new.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"judged={len(new)} status={counts} cost=${judge.spent_usd:.4f} -> {path}")
    if counts.get("budget_abort"):
        sys.stderr.write("mcp-chaos: judge budget exceeded, remaining runs left unjudged\n")
        return 3
    return 0


def cmd_label_sample(args: argparse.Namespace) -> int:
    runs = _load_runs(args.bench_id)
    if not runs:
        sys.stderr.write(f"mcp-chaos: no runs found for bench {args.bench_id}\n")
        return 2
    sample = stratified_sample(list(runs.values()), args.n, args.seed)
    out = args.out or _results_dir(args.bench_id) / "label_sample.csv"
    write_sample_csv(out, sample)
    print(f"wrote {len(sample)} rows -> {out}")
    return 0


def cmd_kappa(args: argparse.Namespace) -> int:
    records = load_records(_results_dir(args.bench_id) / LABELS_FILE)
    try:
        human = read_human_csv(args.human)
    except (OSError, KeyError) as e:
        sys.stderr.write(f"mcp-chaos: cannot read human labels: {e}\n")
        return 2
    results = compute_kappas(records, human, args.seed)
    if not results:
        sys.stderr.write("mcp-chaos: no overlapping labeled runs\n")
        return 2
    for r in results:
        print(r.line())
    return 0
