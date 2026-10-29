from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import anthropic
import pytest

from mcp_chaos.bench.cli import main
from mcp_chaos.judge.labels import LABELS_FILE, load_records
from mcp_chaos.schemas.results import RunResult
from tests.integration.test_llm_judge import client_for, fake
from tests.unit.judge_helpers import error_applied, make_run, trace

BENCH = "b1"


def write_bench(root: Path, runs: list[RunResult]) -> Path:
    d = root / "results" / BENCH
    (d / "traces").mkdir(parents=True)
    (d / "runs.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in runs))
    for r in runs:
        td = d / "traces" / r.run_id
        td.mkdir(exist_ok=True)
        (td / "files.jsonl").write_text(
            trace(r.run_id, 1, "files", fault=error_applied()).model_dump_json() + "\n\n"
        )
    return d


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    runs = [make_run(seed=s, plan_faults="error", success=s % 2 == 0) for s in range(6)]
    runs.append(make_run(seed=0))
    return write_bench(tmp_path, runs)


def test_offline_label_sample_and_missing_bench(
    bench: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["judge", "--bench-id", BENCH, "--mode", "offline"]) == 0
    recs = load_records(bench / LABELS_FILE)
    assert len(recs) == 7 and {r.status for r in recs.values()} == {"offline"}
    assert main(["label-sample", "--bench-id", BENCH, "--n", "4"]) == 0
    rows = list(csv.DictReader((bench / "label_sample.csv").open()))
    assert len(rows) == 4
    assert main(["judge", "--bench-id", "nope"]) == 2
    assert main(["label-sample", "--bench-id", "nope"]) == 2
    assert "wrote 4 rows" in capsys.readouterr().out


def test_sync_judge_then_kappa(
    bench: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = fake()

    def factory(**kw: Any) -> Any:
        assert kw["base_url"] == "http://fake" and kw["api_key"] == "dummy-not-a-key"
        return client_for(app)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["judge", "--bench-id", BENCH, "--mode", "sync", "--base-url", "http://fake"]
    assert main(args) == 0
    assert main(args) == 0  # second pass: cache hits, no new requests
    assert len(app.state.requests) == 7
    recs = load_records(bench / LABELS_FILE)
    assert {r.status for r in recs.values()} == {"ok"}

    assert main(["label-sample", "--bench-id", BENCH, "--n", "7"]) == 0
    rows = list(csv.DictReader((bench / "label_sample.csv").open()))
    for row in rows:
        row["human_failure_detected"] = "yes"  # judge said True everywhere
        row["human_false_success_claim"] = "yes" if row["task_success"] == "False" else "no"
        row["human_blast_radius"] = "none"
    human = bench / "human.csv"
    with human.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    capsys.readouterr()
    assert main(["kappa", "--bench-id", BENCH, "--human", str(human)]) == 0
    out = capsys.readouterr().out
    assert "failure_detected: kappa=1.000" in out and "false_success_claim" in out
    assert "95% CI" in out


def test_sync_needs_key_and_budget_exit_code(
    bench: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["judge", "--bench-id", BENCH, "--mode", "sync"]) == 2
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **_: client_for(fake()))
    batch = ["judge", "--bench-id", BENCH, "--mode", "batch", "--base-url", "http://f"]
    assert main([*batch, "--budget-usd", "0"]) == 3
    assert "budget" in capsys.readouterr().err
    assert main([*batch, "--model", "gpt-9"]) == 2


def test_kappa_errors(bench: Path) -> None:
    assert main(["kappa", "--bench-id", BENCH, "--human", str(bench / "missing.csv")]) == 2
    empty = bench / "e.csv"
    empty.write_text("run_id,human_failure_detected\n")
    assert main(["kappa", "--bench-id", BENCH, "--human", str(empty)]) == 2
