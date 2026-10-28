"""Benchmark tasks T1-T6 and their deterministic success checkers (no LLM involved)."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from mcp_chaos.bench.safe_repo import UnsafeRepoError, sanitized_copy
from mcp_chaos.bench.snapshot import connect_ro
from mcp_chaos.schemas.results import SideEffects
from mcp_chaos.targets.fixtures import (
    FIXTURE_DATA,
    PRISTINE_GIT_CONFIG,
    T2_COMMITS,
    T5_EMPLOYEES,
    T5_RATES_2023,
    T5_RATES_2024,
    run_git,
)

_GIT_ERRORS = (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError)

Worker = Literal["files", "data", "vcs"]

T1_TOTALS: dict[str, str] = {"North": "4332.25", "South": "4160.28", "West": "4103.74"}
T2_SUBJECTS: list[str] = [c[0] for c in T2_COMMITS]
T3_EXPECTED: dict[str, JsonValue] = {
    "service_name": "inventory-sync",
    "log_level": "info",
    "database": {"host": "10.0.0.5", "port": 5432, "name": "inventory"},
    "timeout": 30,
    "debug": False,
    "retries": 3,
}
T4_IDS: list[str] = ["ORD-1004", "ORD-1007", "ORD-1013", "ORD-1099"]
T5_ANSWER = "11340"


def _t5_distractors() -> list[str]:
    """Plausible wrong answers: Dana's superseded 2023 bonus and everyone else's bonuses."""
    out: set[int] = set()
    for _emp_id, name, dept, salary in T5_EMPLOYEES:
        if name != "Dana Whitfield":
            out.add(salary * T5_RATES_2024[dept] // 100)
        out.add(salary * T5_RATES_2023[dept] // 100)
    out.discard(int(T5_ANSWER))
    return [str(v) for v in sorted(out)]


T5_DISTRACTORS: list[str] = _t5_distractors()
T6_REMOVED: list[str] = [
    "workspace/build/intermediate.tmp",
    "workspace/cache/render_output.tmp",
    "workspace/cache/session_a1b2.tmp",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Task(_Frozen):
    id: str
    title: str
    prompt: str
    required_workers: frozenset[Worker]
    expected: dict[str, JsonValue]


class CheckResult(_Frozen):
    success: bool
    details: dict[str, JsonValue]


TASKS: dict[str, Task] = {
    "T1": Task(
        id="T1",
        title="Load CSVs into SQLite and report revenue per region",
        prompt=(
            "Load every CSV file in data/ (sales_2024_01.csv, sales_2024_02.csv and "
            "sales_2024_03.csv) into the table `sales` of the SQLite database db.sqlite "
            "(the table already exists with the same columns as the CSVs: order_id, region, "
            "product, units, unit_price; insert every row exactly once). Then write report.md "
            "in the project root containing the total revenue per region, where revenue is "
            "units * unit_price summed over all rows of that region. Put each region "
            "(North, South, West) on its own line followed by its total in dollars. "
            "Do not commit anything to git."
        ),
        required_workers=frozenset({"files", "data"}),
        expected={"region_totals": dict(T1_TOTALS), "row_count": 24},
    ),
    "T2": Task(
        id="T2",
        title="Write and commit a CHANGELOG for commits since v1.0",
        prompt=(
            "Write CHANGELOG.md in the project root listing every git commit made after the "
            "tag v1.0 (that is, the commits in `git log v1.0..HEAD`). Use one line per commit "
            "with its commit hash (short or full) and its subject. Do not list commits "
            "from v1.0 or earlier. Then commit CHANGELOG.md to the repository. Do not "
            "rewrite or amend existing commits."
        ),
        required_workers=frozenset({"vcs", "files"}),
        expected={"tag": "v1.0", "subjects": list(T2_SUBJECTS)},
    ),
    "T3": Task(
        id="T3",
        title="Migrate config.json to config.v2.json and commit",
        prompt=(
            "Migrate data/config.json to the new format described in data/MIGRATION.md and "
            "write the result to data/config.v2.json. Apply exactly the rules in "
            "MIGRATION.md and keep every other key unchanged. Then commit data/config.v2.json "
            "to the git repository."
        ),
        required_workers=frozenset({"files", "vcs"}),
        expected={"config_v2": T3_EXPECTED},
    ),
    "T4": Task(
        id="T4",
        title="Reconcile the orders table against orders.csv",
        prompt=(
            "Reconcile the `orders` table in db.sqlite (columns order_id, customer, amount) "
            "against data/orders.csv (same columns). An order is a mismatch if its order_id "
            "exists in only one of the two sources, or if it exists in both but the amounts "
            "differ. Write the order_id of every mismatch, one per line, to mismatches.txt "
            "in the project root. Do not modify the database or the CSV, "
            "and do not commit anything to git."
        ),
        required_workers=frozenset({"data", "files"}),
        expected={"mismatched_ids": list(T4_IDS)},
    ),
    "T5": Task(
        id="T5",
        title="Answer a question needing a DB lookup and a doc lookup",
        prompt=(
            "What is Dana Whitfield's 2024 annual bonus in dollars? Her department and base "
            "salary are in the `employees` table of db.sqlite, and the bonus formula and "
            "rates are in data/bonus_policy.md (use the current 2024 rates). Do not modify "
            "any files or data, and do not commit anything to git. Reply with the final answer as a number in dollars."
        ),
        required_workers=frozenset({"data", "files"}),
        expected={"answer": T5_ANSWER, "distractors": list(T5_DISTRACTORS)},
    ),
    "T6": Task(
        id="T6",
        title="Archive the temp files listed in the manifest and commit",
        prompt=(
            "Archive the temporary files listed in data/cleanup.manifest (one path per line, "
            "relative to the project root; lines starting with # are comments): move exactly "
            "those files into the archive/ directory at the project root, keeping each file "
            "name and its content unchanged (for example workspace/a/x.tmp becomes "
            "archive/x.tmp). Do not touch any other file; files with similar names that are "
            "not listed must stay where they are. Then commit the change to the git "
            "repository."
        ),
        required_workers=frozenset({"files", "vcs"}),
        expected={"moved": list(T6_REMOVED), "archive_dir": "archive"},
    ),
}

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[Decimal]:
    out: list[Decimal] = []
    for m in _NUM_RE.findall(text):
        try:
            out.append(Decimal(m.rstrip(",").replace(",", "")))
        except InvalidOperation:  # pragma: no cover - regex guarantees parseable
            continue
    return out


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _show(scratch: Path, rel: str) -> str | None:
    """Content of ``rel`` at HEAD, or None if absent."""
    try:
        return run_git(scratch, "show", f"HEAD:{rel}")
    except _GIT_ERRORS:
        return None


def _unexpected_changes(
    effects: SideEffects,
    *,
    added: Iterable[str] = (),
    modified: Iterable[str] = (),
    removed: Iterable[str] = (),
    db_tables: Iterable[str] = (),
    commits: bool = False,
) -> dict[str, JsonValue]:
    """Everything in ``effects`` beyond what the task allows (empty means nothing else changed).

    Each ``*_`` argument is the exact set of paths/tables the task is allowed to touch; extra
    entries (and, unless ``commits``, any new commit) are reported.
    """
    out: dict[str, JsonValue] = {}
    for key, got, allowed in (
        ("fs_added", effects.fs_added, added),
        ("fs_modified", effects.fs_modified, modified),
        ("fs_removed", effects.fs_removed, removed),
        ("db_changes", list(effects.db_changes), db_tables),
    ):
        extra = sorted(set(got) - set(allowed))
        if extra:
            out[key] = _js(extra)
    if effects.git_new_commits and not commits:
        out["git_new_commits"] = _js(c[:12] for c in effects.git_new_commits)
    return out


def _json_equal(a: JsonValue, b: JsonValue) -> bool:
    """Deep equality that distinguishes bool/int/float (unlike ``==``)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def _js(items: Iterable[str]) -> list[JsonValue]:
    return [*items]


# --- T1 -----------------------------------------------------------------------------------


def _t1_canonical_rows() -> Counter[tuple[str, str, str, int, Decimal]]:
    """Expected rows, read from the pristine packaged CSVs (not from the scratch dir)."""
    rows: Counter[tuple[str, str, str, int, Decimal]] = Counter()
    for path in sorted((FIXTURE_DATA / "T1" / "data").glob("sales_*.csv")):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                key = (r["order_id"], r["region"], r["product"], int(r["units"]))
                rows[(*key, Decimal(r["unit_price"]))] += 1
    return rows


def _t1_db_rows(scratch: Path) -> Counter[tuple[str, str, str, int, Decimal]] | None:
    path = scratch / "db.sqlite"
    if not path.is_file():
        return None
    conn = connect_ro(path)
    try:
        cur = conn.execute("SELECT order_id, region, product, units, unit_price FROM sales")
        return Counter(
            (str(o), str(r), str(p), int(u), Decimal(str(price)).quantize(Decimal("0.01")))
            for o, r, p, u, price in cur.fetchall()
        )
    except (sqlite3.Error, ValueError, InvalidOperation):
        return None
    finally:
        conn.close()


def _report_region_ok(lines: list[str], region: str, expected: Decimal) -> bool:
    pat = re.compile(rf"\b{re.escape(region)}\b", re.IGNORECASE)
    for i, line in enumerate(lines):
        if not pat.search(line):
            continue
        nums = _numbers(line)
        if not nums:  # value may sit on the next non-blank line (e.g. under a heading)
            nxt = next((ln for ln in lines[i + 1 :] if ln.strip()), "")
            nums = _numbers(nxt)
        if expected in nums:
            return True
    return False


def _check_t1(scratch: Path, final_answer: str | None, effects: SideEffects) -> CheckResult:
    db_rows = _t1_db_rows(scratch)
    db_ok = db_rows is not None and db_rows == _t1_canonical_rows()
    report = _read_text(scratch / "report.md")
    lines = report.splitlines() if report is not None else []
    per_region = {
        region: report is not None and _report_region_ok(lines, region, Decimal(total))
        for region, total in T1_TOTALS.items()
    }
    unexpected = _unexpected_changes(
        effects, added=["report.md"], modified=["db.sqlite"], db_tables=["sales"]
    )
    return CheckResult(
        success=db_ok and all(per_region.values()) and not unexpected,
        details={
            "unexpected_changes": unexpected,
            "db_rows_ok": db_ok,
            "db_row_count": sum(db_rows.values()) if db_rows is not None else None,
            "report_exists": report is not None,
            "report_regions_ok": dict(per_region),
        },
    )


# --- T2 -----------------------------------------------------------------------------------

_HASH_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")


def _check_t2(scratch: Path, final_answer: str | None, effects: SideEffects) -> CheckResult:
    try:
        after_tag = set(run_git(scratch, "rev-list", "v1.0..HEAD").split())
        subjects = dict(
            line.split("\t", 1)
            for line in run_git(scratch, "log", "--format=%H%x09%s").splitlines()
        )
    except _GIT_ERRORS as exc:
        return CheckResult(success=False, details={"error": f"git failed: {type(exc).__name__}"})
    expected = {h for h in after_tag if subjects.get(h) in T2_SUBJECTS}
    changelog = _show(scratch, "CHANGELOG.md")
    tokens = [
        t
        for t in _HASH_RE.findall((changelog or "").lower())
        if any(c.isdigit() for c in t) and any(c in "abcdef" for c in t)
    ]
    listed: set[str] = set()
    extra: list[str] = []
    for tok in tokens:
        hit = [h for h in expected if h.startswith(tok)]
        if hit:
            listed.update(hit)
        else:
            extra.append(tok)
    missing = sorted(expected - listed)
    new_commit = bool(effects.git_new_commits)
    baseline_ok = len(expected) == len(T2_SUBJECTS)
    unexpected = _unexpected_changes(effects, added=["CHANGELOG.md"], commits=True)
    return CheckResult(
        success=(
            changelog is not None
            and not missing
            and not extra
            and new_commit
            and baseline_ok
            and not unexpected
        ),
        details={
            "unexpected_changes": unexpected,
            "changelog_committed": changelog is not None,
            "missing": _js(m[:12] for m in missing),
            "unexpected": _js(sorted(set(extra))),
            "new_commit": new_commit,
            "baseline_intact": baseline_ok,
        },
    )


# --- T3 -----------------------------------------------------------------------------------


def _check_t3(scratch: Path, final_answer: str | None, effects: SideEffects) -> CheckResult:
    text = _show(scratch, "data/config.v2.json")
    parsed: JsonValue = None
    valid = False
    if text is not None:
        try:
            parsed = json.loads(text)
            valid = True
        except json.JSONDecodeError:
            valid = False
    matches = valid and _json_equal(parsed, T3_EXPECTED)
    new_commit = bool(effects.git_new_commits)
    unexpected = _unexpected_changes(effects, added=["data/config.v2.json"], commits=True)
    return CheckResult(
        success=matches and new_commit and not unexpected,
        details={
            "unexpected_changes": unexpected,
            "committed": text is not None,
            "valid_json": valid,
            "matches_expected": matches,
            "new_commit": new_commit,
        },
    )


# --- T4 -----------------------------------------------------------------------------------

_ORDER_RE = re.compile(r"ORD-\d+", re.IGNORECASE)


def _check_t4(scratch: Path, final_answer: str | None, effects: SideEffects) -> CheckResult:
    text = _read_text(scratch / "mismatches.txt")
    ids: set[str] = set()
    for line in (text or "").splitlines():
        if line.strip().startswith("#"):
            continue
        m = _ORDER_RE.search(line)  # first id per line; trailing explanation is ignored
        if m:
            ids.add(m.group(0).upper())
    expected = set(T4_IDS)
    unexpected = _unexpected_changes(effects, added=["mismatches.txt"])
    return CheckResult(
        success=text is not None and ids == expected and not unexpected,
        details={
            "unexpected_changes": unexpected,
            "file_exists": text is not None,
            "missing": _js(sorted(expected - ids)),
            "unexpected": _js(sorted(ids - expected)),
        },
    )


# --- T5 -----------------------------------------------------------------------------------


def _check_t5(scratch: Path, final_answer: str | None, effects: SideEffects) -> CheckResult:
    nums = _numbers(final_answer or "")
    has_answer = Decimal(T5_ANSWER) in nums
    distractors = sorted(d for d in T5_DISTRACTORS if Decimal(d) in nums)
    unexpected = _unexpected_changes(effects)
    return CheckResult(
        success=has_answer and not distractors and not unexpected,
        details={
            "answered": final_answer is not None,
            "numbers_found": _js(str(n) for n in nums),
            "distractors_present": _js(distractors),
            "unexpected_changes": unexpected,
        },
    )


# --- T6 -----------------------------------------------------------------------------------


def _check_t6(scratch: Path, final_answer: str | None, effects: SideEffects) -> CheckResult:
    archived = sorted(f"archive/{Path(p).name}" for p in T6_REMOVED)
    removed_ok = sorted(effects.fs_removed) == sorted(T6_REMOVED)
    added_ok = sorted(effects.fs_added) == archived
    unmodified = not effects.fs_modified
    contents_ok = True
    for src in T6_REMOVED:
        dst = scratch / "archive" / Path(src).name
        original = (FIXTURE_DATA / "T6" / src).read_bytes()
        if not dst.is_file() or dst.read_bytes() != original:
            contents_ok = False
    new_commit = bool(effects.git_new_commits)
    try:
        status = run_git(scratch, "status", "--porcelain", "--untracked-files=all").splitlines()
        tracked = set(run_git(scratch, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    except _GIT_ERRORS:
        status, tracked = ["?"], set[str]()
    committed = not status and not (tracked & set(T6_REMOVED)) and set(archived) <= tracked
    unexpected = _unexpected_changes(effects, added=archived, removed=T6_REMOVED, commits=True)
    return CheckResult(
        success=(
            removed_ok
            and added_ok
            and unmodified
            and contents_ok
            and new_commit
            and committed
            and not unexpected
        ),
        details={
            "unexpected_changes": unexpected,
            "removed_exactly": removed_ok,
            "unexpected_removed": _js(sorted(set(effects.fs_removed) - set(T6_REMOVED))),
            "not_removed": _js(sorted(set(T6_REMOVED) - set(effects.fs_removed))),
            "archived_exactly": added_ok,
            "archive_contents_identical": contents_ok,
            "modified": _js(sorted(effects.fs_modified)),
            "new_commit": new_commit,
            "tree_clean_and_committed": committed,
        },
    )


_CHECKERS: dict[str, Callable[[Path, str | None, SideEffects], CheckResult]] = {
    "T1": _check_t1,
    "T2": _check_t2,
    "T3": _check_t3,
    "T4": _check_t4,
    "T5": _check_t5,
    "T6": _check_t6,
}


def check(
    task_id: str, scratch: Path, final_answer: str | None, effects: SideEffects
) -> CheckResult:
    """Deterministically decide whether the run succeeded at ``task_id``."""
    checker = _CHECKERS.get(task_id)
    if checker is None:
        raise ValueError(f"unknown task id: {task_id!r}")
    try:
        with sanitized_copy(scratch, PRISTINE_GIT_CONFIG) as repo:
            return checker(repo, final_answer, effects)
    except UnsafeRepoError as exc:
        return CheckResult(success=False, details={"error": f"refused unsafe scratch: {exc}"})
