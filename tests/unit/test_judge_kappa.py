from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mcp_chaos.judge.kappa import (
    SAMPLE_COLUMNS,
    bootstrap_ci,
    cohen_kappa,
    compute_kappas,
    read_human_csv,
    stratified_sample,
    stratum_of,
    write_sample_csv,
)
from mcp_chaos.judge.labels import JudgeRecord, LlmLabels, PartialLabels, merge

from .judge_helpers import make_run


def expand(tp: int, fp: int, fn: int, tn: int) -> tuple[list[bool], list[bool]]:
    a = [True] * tp + [True] * fp + [False] * fn + [False] * tn
    b = [True] * tp + [False] * fp + [True] * fn + [False] * tn
    return a, b


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ((20, 0, 0, 30), 1.0),
        ((0, 25, 25, 0), -1.0),
        ((25, 25, 25, 25), 0.0),
        ((45, 5, 15, 35), 0.6),  # classic worked example
        ((10, 0, 0, 0), 1.0),  # constant raters: defined as 1
    ],
)
def test_kappa_known(matrix: tuple[int, int, int, int], expected: float) -> None:
    a, b = expand(*matrix)
    assert cohen_kappa(a, b) == pytest.approx(expected)


def test_kappa_constant_disagreement_and_multiclass() -> None:
    assert cohen_kappa([True, True], [False, False]) == 0.0
    a = ["none", "self", "peer", "peer"]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ["none", "self", "peer", "self"]) == pytest.approx(
        (0.75 - 0.3125) / (1 - 0.3125)
    )


def test_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])


def test_bootstrap_ci_deterministic_and_brackets() -> None:
    a, b = expand(45, 5, 15, 35)
    lo, hi = bootstrap_ci(a, b, seed=7, n_boot=300)
    assert (lo, hi) == bootstrap_ci(a, b, seed=7, n_boot=300)
    assert lo < cohen_kappa(a, b) < hi
    assert (lo, hi) != bootstrap_ci(a, b, seed=8, n_boot=300)


def _pop() -> list:
    out = []
    for seed in range(6):
        for plan in ("none", "inject", "error"):
            for hardened in (False, True):
                for fw in ("langgraph", "agentsdk"):
                    out.append(
                        make_run(seed=seed, plan_faults=plan, hardened=hardened, framework=fw)
                    )
    return out


def test_stratified_sampling_deterministic_and_covers_strata() -> None:
    runs = _pop()
    s1 = stratified_sample(runs, 12, seed=3)
    assert s1 == stratified_sample(list(reversed(runs)), 12, seed=3)
    assert len(s1) == 12
    assert len({stratum_of(r) for r in s1}) == 12  # 12 strata exist, one each
    assert s1 != stratified_sample(runs, 12, seed=4)
    assert len(stratified_sample(runs, 10_000)) == len(runs)
    assert len({r.run_id for r in stratified_sample(runs, 30)}) == 30


def test_sample_csv_round_trip(tmp_path: Path) -> None:
    sample = stratified_sample(_pop(), 5)
    path = tmp_path / "s.csv"
    write_sample_csv(path, sample)
    rows = list(csv.DictReader(path.open()))
    assert tuple(rows[0]) == SAMPLE_COLUMNS
    assert [r["run_id"] for r in rows] == [r.run_id for r in sample]
    assert read_human_csv(path).keys() == {r.run_id for r in sample}


def _rec(rid: str, fd: bool, claims: bool, blast: str, det_blast=None) -> JudgeRecord:
    det = PartialLabels(task_success=False, blast_radius=det_blast)
    llm = LlmLabels(failure_detected=fd, claims_success=claims, blast_radius=blast, rationale="r")  # type: ignore[arg-type]
    return JudgeRecord(
        run_id=rid, mode="sync", status="ok", deterministic=det, llm=llm, labels=merge(det, llm)
    )


def test_compute_kappas() -> None:
    recs = {
        "a": _rec("a", True, True, "peer"),
        "b": _rec("b", False, False, "self"),
        "c": _rec("c", True, False, "supervisor"),
        "d": _rec("d", False, True, "none"),
        "e": JudgeRecord(run_id="e", mode="offline", status="offline"),
    }
    human = {
        "a": {
            "human_failure_detected": "yes",
            "human_false_success_claim": "1",
            "human_blast_radius": "peer",
        },
        "b": {
            "human_failure_detected": "no",
            "human_false_success_claim": "0",
            "human_blast_radius": "self",
        },
        "c": {
            "human_failure_detected": "true",
            "human_false_success_claim": "false",
            "human_blast_radius": "peer",
        },
        "d": {
            "human_failure_detected": "false",
            "human_false_success_claim": "true",
            "human_blast_radius": "",
        },
        "e": {
            "human_failure_detected": "true",
            "human_false_success_claim": "true",
            "human_blast_radius": "none",
        },
        "zz": {"human_failure_detected": "true"},
        "q": {"human_failure_detected": "maybe"},
    }
    res = {r.label: r for r in compute_kappas(recs, human, seed=1)}
    assert res["failure_detected"].kappa == 1.0 and res["failure_detected"].n == 4
    assert res["false_success_claim"].kappa == 1.0
    assert res["blast_radius"].n == 3  # d has no human blast label
    assert res["blast_radius"].kappa == pytest.approx(0.5)
    assert "kappa=1.000" in res["failure_detected"].line()


def test_deterministic_blast_excluded_from_blast_kappa() -> None:
    from mcp_chaos.schemas.results import BlastRadius

    recs = {"a": _rec("a", True, True, "peer", det_blast=BlastRadius.PERSISTENT)}
    human = {
        "a": {
            "human_failure_detected": "1",
            "human_false_success_claim": "1",
            "human_blast_radius": "peer",
        }
    }
    assert {r.label for r in compute_kappas(recs, human)} == {
        "failure_detected",
        "false_success_claim",
    }
