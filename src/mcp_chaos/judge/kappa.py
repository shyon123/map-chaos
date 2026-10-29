"""Judge-vs-human agreement (Cohen's kappa with seeded bootstrap CI) and stratified sampling."""

from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Hashable, Sequence
from pathlib import Path

from mcp_chaos.judge.labels import JudgeRecord
from mcp_chaos.schemas.results import RunResult

HUMAN_COLUMNS = ("human_failure_detected", "human_false_success_claim", "human_blast_radius")
LLM_LABELS = ("failure_detected", "false_success_claim", "blast_radius")
SAMPLE_COLUMNS = (
    "run_id",
    "stratum",
    "task_id",
    "task_success",
    "fired_faults",
    "check_summary",
    "final_answer",
    *HUMAN_COLUMNS,
)
BOOTSTRAP_N = 2000
MAX_CELL_CHARS = 2000


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two equal-length label sequences.

    When chance agreement is 1 (both raters constant and equal) kappa is undefined; we return
    1.0 for perfect agreement and 0.0 otherwise.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty sequences of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def bootstrap_ci(
    a: Sequence[Hashable],
    b: Sequence[Hashable],
    *,
    seed: int = 0,
    n_boot: int = BOOTSTRAP_N,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for kappa; deterministic given ``seed``."""
    rng = random.Random(seed)  # noqa: S311
    n = len(a)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(cohen_kappa([a[i] for i in idx], [b[i] for i in idx]))
    stats.sort()
    lo = stats[int((alpha / 2) * (n_boot - 1))]
    hi = stats[int((1 - alpha / 2) * (n_boot - 1))]
    return lo, hi


# -- stratified sample --------------------------------------------------------------------------
def stratum_of(rr: RunResult) -> str:
    cell = "+".join(sorted({f.mode.value for f in rr.config.fault_plan.faults})) or "clean"
    arm = "hardened" if rr.config.hardening is not None else "baseline"
    return f"{cell}|{rr.config.framework}|{arm}"


def stratified_sample(runs: Sequence[RunResult], n: int, seed: int = 0) -> list[RunResult]:
    """Round-robin over strata (cell x framework x arm, sorted) drawing from seeded shuffles.

    Every stratum is represented before any is repeated; the result is returned sorted by run_id.
    """
    pools: dict[str, list[RunResult]] = {}
    for r in sorted(runs, key=lambda x: x.run_id):
        pools.setdefault(stratum_of(r), []).append(r)
    rng = random.Random(seed)  # noqa: S311
    for k in sorted(pools):
        rng.shuffle(pools[k])
    picked: list[RunResult] = []
    while len(picked) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(picked) < n:
                picked.append(pools[k].pop())
    return sorted(picked, key=lambda x: x.run_id)


def write_sample_csv(path: Path, sample: Sequence[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_COLUMNS)
        for r in sample:
            fired = ",".join(sorted({x.mode.value for x in r.faults_fired}))
            check = f"success={r.check.success} details={r.check.details}" if r.check else "(none)"
            w.writerow(
                [
                    r.run_id,
                    stratum_of(r),
                    r.config.task_id,
                    r.check.success if r.check else False,
                    fired,
                    check[:MAX_CELL_CHARS],
                    (r.final_answer or "")[:MAX_CELL_CHARS],
                    "",
                    "",
                    "",
                ]
            )


# -- agreement ----------------------------------------------------------------------------------
def _parse_bool(s: str) -> bool | None:
    v = s.strip().lower()
    if v in {"1", "true", "yes", "y", "t"}:
        return True
    if v in {"0", "false", "no", "n", "f"}:
        return False
    return None


def read_human_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["run_id"]: row for row in csv.DictReader(f) if row.get("run_id")}


class KappaResult:
    __slots__ = ("ci", "kappa", "label", "n")

    def __init__(self, label: str, n: int, kappa: float, ci: tuple[float, float]) -> None:
        self.label, self.n, self.kappa, self.ci = label, n, kappa, ci

    def line(self) -> str:
        return (
            f"{self.label}: kappa={self.kappa:.3f} "
            f"95% CI [{self.ci[0]:.3f}, {self.ci[1]:.3f}] n={self.n}"
        )


def compute_kappas(
    records: dict[str, JudgeRecord], human: dict[str, dict[str, str]], seed: int = 0
) -> list[KappaResult]:
    """Kappa per LLM-judged label over runs having both a human row and merged labels.

    ``blast_radius`` only counts runs where the LLM decided it (deterministic values are facts).
    """
    out: list[KappaResult] = []
    for label, col in zip(LLM_LABELS, HUMAN_COLUMNS, strict=True):
        a: list[Hashable] = []
        b: list[Hashable] = []
        for rid in sorted(human):
            rec = records.get(rid)
            if rec is None or rec.labels is None:
                continue
            raw = human[rid].get(col, "")
            if label == "blast_radius":
                if rec.labels.label_source.get("blast_radius") == "deterministic":
                    continue
                hv: Hashable | None = raw.strip().lower() or None
                lv: Hashable = rec.labels.blast_radius.value
            else:
                hv = _parse_bool(raw)
                lv = bool(getattr(rec.labels, label))
            if hv is None:
                continue
            a.append(lv)
            b.append(hv)
        if a:
            out.append(KappaResult(label, len(a), cohen_kappa(a, b), bootstrap_ci(a, b, seed=seed)))
    return out
