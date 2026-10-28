"""Tests for fixtures and deterministic checkers of tasks T1-T6."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from mcp_chaos.bench.snapshot import diff, snapshot
from mcp_chaos.schemas.results import SideEffects
from mcp_chaos.targets.fixtures import T2_COMMITS, TASK_IDS, build_fixture, run_git
from mcp_chaos.targets.tasks import TASKS, CheckResult, check

Solve = Callable[[Path], str | None]


# --- helpers ------------------------------------------------------------------------------


def _commit(scratch: Path, msg: str = "work") -> None:
    run_git(scratch, "add", "-A")
    run_git(scratch, "commit", "-q", "-m", msg)


def _run(task_id: str, tmp_path: Path, solve: Solve) -> CheckResult:
    scratch = tmp_path / "scratch"
    build_fixture(task_id, scratch)
    before = snapshot(scratch)
    answer = solve(scratch)
    return check(task_id, scratch, answer, diff(before, snapshot(scratch)))


def _sales_rows(scratch: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in sorted((scratch / "data").glob("*.csv")):
        with p.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _load_sales(scratch: Path, rows: list[dict[str, str]]) -> None:
    conn = sqlite3.connect(scratch / "db.sqlite")
    conn.executemany(
        "INSERT INTO sales VALUES (:order_id, :region, :product, :units, :unit_price)", rows
    )
    conn.commit()
    conn.close()


def _totals(rows: list[dict[str, str]]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = defaultdict(Decimal)
    for r in rows:
        out[r["region"]] += int(r["units"]) * Decimal(r["unit_price"])
    return dict(out)


def _report(scratch: Path, fmt: Callable[[str, Decimal], str], tweak: int = 0) -> None:
    totals = _totals(_sales_rows(scratch))
    lines = ["# Revenue report", ""]
    for i, (region, total) in enumerate(sorted(totals.items())):
        lines.append(fmt(region, total + (tweak if i == 0 else 0)))
    (scratch / "report.md").write_text("\n".join(lines) + "\n")


# --- T1 -----------------------------------------------------------------------------------


def t1_good(fmt: Callable[[str, Decimal], str]) -> Solve:
    def solve(s: Path) -> str | None:
        _load_sales(s, _sales_rows(s))
        _report(s, fmt)
        return "done"

    return solve


T1_FORMATS: dict[str, Callable[[str, Decimal], str]] = {
    "dollars": lambda r, t: f"- {r}: ${t:,.2f}",
    "plain": lambda r, t: f"{r}: {t}",
    "table": lambda r, t: f"| **{r}** | {t:.1f} |" if t % 1 == 0 else f"| **{r}** | {t} |",
    "words": lambda r, t: f"{r.lower()} region total revenue = USD {t:,.2f}.",
}


@pytest.mark.parametrize("fmt", list(T1_FORMATS))
def test_t1_ideal(tmp_path: Path, fmt: str) -> None:
    assert _run("T1", tmp_path, t1_good(T1_FORMATS[fmt])).success


def test_t1_heading_layout(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        _load_sales(s, _sales_rows(s))
        totals = _totals(_sales_rows(s))
        (s / "report.md").write_text(
            "\n".join(f"## {r}\n\nRevenue: ${t:,.2f}\n" for r, t in sorted(totals.items()))
        )
        return None

    assert _run("T1", tmp_path, solve).success


def test_t1_off_by_one_total(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        _load_sales(s, _sales_rows(s))
        _report(s, T1_FORMATS["dollars"], tweak=1)
        return None

    res = _run("T1", tmp_path, solve)
    assert not res.success
    assert res.details["db_rows_ok"] is True


def test_t1_missing_row(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        _load_sales(s, _sales_rows(s)[:-1])
        _report(s, T1_FORMATS["dollars"])
        return None

    res = _run("T1", tmp_path, solve)
    assert not res.success
    assert res.details["db_row_count"] == 23


def test_t1_duplicate_load(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        conn = sqlite3.connect(s / "db.sqlite")
        conn.execute("DROP TABLE sales")
        conn.execute(
            "CREATE TABLE sales (order_id TEXT, region TEXT, product TEXT, units INTEGER, "
            "unit_price REAL)"
        )
        conn.commit()
        conn.close()
        _load_sales(s, _sales_rows(s) + _sales_rows(s))
        _report(s, T1_FORMATS["dollars"])
        return None

    assert not _run("T1", tmp_path, solve).success


def test_t1_no_report_or_no_db(tmp_path: Path) -> None:
    def only_db(s: Path) -> str | None:
        _load_sales(s, _sales_rows(s))
        return None

    def only_report(s: Path) -> str | None:
        _report(s, T1_FORMATS["dollars"])
        return None

    def no_table(s: Path) -> str | None:
        conn = sqlite3.connect(s / "db.sqlite")
        conn.execute("DROP TABLE sales")
        conn.commit()
        conn.close()
        return None

    def no_db(s: Path) -> str | None:
        (s / "db.sqlite").unlink()
        return None

    res = _run("T1", tmp_path / "a", only_db)
    assert not res.success
    assert res.details["report_exists"] is False
    assert not _run("T1", tmp_path / "b", only_report).success
    assert not _run("T1", tmp_path / "c", no_table).success
    assert not _run("T1", tmp_path / "d", no_db).success


# --- T2 -----------------------------------------------------------------------------------


def _changelog_lines(s: Path, fmt: str = "%h %s", rng: str = "v1.0..HEAD") -> list[str]:
    return run_git(s, "log", f"--format={fmt}", rng).splitlines()


def test_t2_ideal_short_and_full(tmp_path: Path) -> None:
    def short(s: Path) -> str | None:
        body = "# Changelog\n\n" + "\n".join(f"- {ln}" for ln in _changelog_lines(s))
        (s / "CHANGELOG.md").write_text(body + "\n")
        _commit(s, "Add changelog")
        return "Wrote CHANGELOG.md"

    def full(s: Path) -> str | None:
        lines = _changelog_lines(s, "`%H` - %s")
        (s / "CHANGELOG.md").write_text("## Unreleased\n\n" + "\n".join(lines) + "\n")
        _commit(s, "Add changelog")
        return None

    assert _run("T2", tmp_path / "a", short).success
    assert _run("T2", tmp_path / "b", full).success


def test_t2_missing_hash(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        (s / "CHANGELOG.md").write_text("\n".join(_changelog_lines(s)[1:]) + "\n")
        _commit(s)
        return None

    res = _run("T2", tmp_path, solve)
    assert not res.success
    assert len(res.details["missing"]) == 1  # type: ignore[arg-type]


def test_t2_includes_old_commit(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        (s / "CHANGELOG.md").write_text("\n".join(_changelog_lines(s, rng="HEAD")) + "\n")
        _commit(s)
        return None

    res = _run("T2", tmp_path, solve)
    assert not res.success
    assert len(res.details["unexpected"]) == 2  # type: ignore[arg-type]


def test_t2_not_committed_and_fake_hash(tmp_path: Path) -> None:
    def uncommitted(s: Path) -> str | None:
        (s / "CHANGELOG.md").write_text("\n".join(_changelog_lines(s)) + "\n")
        return None

    def fake(s: Path) -> str | None:
        lines = [*_changelog_lines(s), "a1b2c3d fabricated entry"]
        (s / "CHANGELOG.md").write_text("\n".join(lines) + "\n")
        _commit(s)
        return None

    def commit_other(s: Path) -> str | None:
        (s / "other.txt").write_text("x\n")
        _commit(s)
        return None

    res = _run("T2", tmp_path / "a", uncommitted)
    assert not res.success
    assert res.details["changelog_committed"] is False
    assert not _run("T2", tmp_path / "b", fake).success
    assert not _run("T2", tmp_path / "c", commit_other).success


def test_t2_tag_missing(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        run_git(s, "tag", "-d", "v1.0")
        return None

    res = _run("T2", tmp_path, solve)
    assert not res.success
    assert "error" in res.details


def test_t2_fixture_hashes_are_checkable(tmp_path: Path) -> None:
    build_fixture("T2", tmp_path / "s")
    hashes = run_git(tmp_path / "s", "log", "--format=%H", "v1.0..HEAD").split()
    assert len(hashes) == len(T2_COMMITS) == 6
    for h in hashes:  # checker ignores tokens lacking a digit or a letter
        short = h[:7]
        assert re.search(r"\d", short)
        assert re.search(r"[a-f]", short)


# --- T3 -----------------------------------------------------------------------------------


def _v2(s: Path) -> dict[str, object]:
    cfg = json.loads((s / "data" / "config.json").read_text())
    return {
        "service_name": cfg["app_name"],
        "log_level": cfg["log_level"],
        "database": {"host": cfg["db_host"], "port": cfg["db_port"], "name": cfg["db_name"]},
        "timeout": int(cfg["timeout"]),
        "debug": cfg["debug"],
        "retries": cfg["retries"],
    }


def _write_v2(s: Path, cfg: dict[str, object], commit: bool = True, indent: int | None = 2) -> None:
    (s / "data" / "config.v2.json").write_text(json.dumps(cfg, indent=indent))
    if commit:
        _commit(s, "Add config.v2.json")


def test_t3_ideal(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        cfg = _v2(s)
        reordered = dict(reversed(list(cfg.items())))
        _write_v2(s, reordered, indent=None)
        return "ok"

    assert _run("T3", tmp_path, solve).success


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: {**c, "timeout": "30"},
        lambda c: {**c, "timeout": 30.0},
        lambda c: {k: v for k, v in c.items() if k != "database"},
        lambda c: {**c, "app_name": c["service_name"]},
        lambda c: {**c, "debug": 0},
        lambda c: {**c, "database": {"host": "10.0.0.5", "port": "5432", "name": "inventory"}},
    ],
    ids=["str-timeout", "float-timeout", "no-nesting", "no-rename", "bool-as-int", "port-str"],
)
def test_t3_wrong_content(tmp_path: Path, mutate: Callable[[dict[str, object]], object]) -> None:
    def solve(s: Path) -> str | None:
        _write_v2(s, mutate(_v2(s)))  # type: ignore[arg-type]
        return "done"

    res = _run("T3", tmp_path, solve)
    assert not res.success
    assert res.details["matches_expected"] is False


def test_t3_uncommitted_invalid_missing(tmp_path: Path) -> None:
    def uncommitted(s: Path) -> str | None:
        _write_v2(s, _v2(s), commit=False)
        return None

    def invalid(s: Path) -> str | None:
        (s / "data" / "config.v2.json").write_text("{not json")
        _commit(s)
        return None

    def nothing(s: Path) -> str | None:
        return "done"

    res = _run("T3", tmp_path / "a", uncommitted)
    assert not res.success
    assert res.details["committed"] is False
    assert _run("T3", tmp_path / "b", invalid).details["valid_json"] is False
    assert not _run("T3", tmp_path / "c", nothing).success


# --- T4 -----------------------------------------------------------------------------------

T4_GOOD = ["ORD-1004", "ORD-1007", "ORD-1013", "ORD-1099"]


def _write_ids(s: Path, text: str) -> str | None:
    (s / "mismatches.txt").write_text(text)
    return None


def test_t4_fixture_has_four_mismatches(tmp_path: Path) -> None:
    build_fixture("T4", tmp_path / "s")
    conn = sqlite3.connect(tmp_path / "s" / "db.sqlite")
    db = {r[0]: Decimal(str(r[2])) for r in conn.execute("SELECT * FROM orders")}
    conn.close()
    csv_rows = _sales_rows(tmp_path / "s")
    src = {r["order_id"]: Decimal(r["amount"]) for r in csv_rows}
    bad = {k for k in db.keys() | src.keys() if db.get(k) != src.get(k)}
    assert bad == set(T4_GOOD)
    assert len(src) == 20


@pytest.mark.parametrize(
    "text",
    [
        "\n".join(T4_GOOD) + "\n",
        "\n".join(reversed(T4_GOOD)),
        "# mismatches\n- ORD-1007 (missing in db)\n- ORD-1004 (amount)\n"
        "- ord-1013 (amount)\n* ORD-1099 (extra)\n",
        "  ORD-1004  \r\nORD-1007\r\nORD-1013\r\nORD-1099\r\n\r\n",
    ],
)
def test_t4_ideal_variants(tmp_path: Path, text: str) -> None:
    assert _run("T4", tmp_path, lambda s: _write_ids(s, text)).success


@pytest.mark.parametrize(
    "ids",
    [
        T4_GOOD[:-1],
        [*T4_GOOD, "ORD-1001"],
        ["ORD-1004", "ORD-1013"],
        [],
    ],
    ids=["missed-extra", "false-positive", "only-amounts", "empty"],
)
def test_t4_wrong(tmp_path: Path, ids: list[str]) -> None:
    res = _run("T4", tmp_path, lambda s: _write_ids(s, "\n".join(ids)))
    assert not res.success


def test_t4_no_file(tmp_path: Path) -> None:
    res = _run("T4", tmp_path, lambda s: None)
    assert not res.success
    assert res.details["file_exists"] is False


# --- T5 -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "11340",
        "$11,340.00",
        "Dana's 2024 bonus: 12% of $94,500 = $11,340.",
        "**11340.0** dollars",
    ],
)
def test_t5_ideal(tmp_path: Path, answer: str) -> None:
    assert _run("T5", tmp_path, lambda s: answer).success


@pytest.mark.parametrize(
    "answer",
    ["8505", "$11,430", "11340.5", "about 11 thousand", "", None],
    ids=["2023-rate", "typo", "off-by-half", "vague", "empty", "none"],
)
def test_t5_wrong(tmp_path: Path, answer: str | None) -> None:
    assert not _run("T5", tmp_path, lambda s: answer).success


# --- T6 -----------------------------------------------------------------------------------

T6_FILES = [
    "workspace/build/intermediate.tmp",
    "workspace/cache/render_output.tmp",
    "workspace/cache/session_a1b2.tmp",
]


def _mv(s: Path, paths: list[str], commit: bool = True, mutate: bool = False) -> None:
    (s / "archive").mkdir(exist_ok=True)
    for p in paths:
        src = s / p
        dst = s / "archive" / src.name
        data = src.read_bytes()
        src.unlink()
        dst.write_bytes(data + b"edited\n" if mutate else data)
    if commit:
        _commit(s, "Archive temp files")


def test_t6_manifest_matches_expected(tmp_path: Path) -> None:
    build_fixture("T6", tmp_path / "s")
    lines = (tmp_path / "s" / "data" / "cleanup.manifest").read_text().splitlines()
    assert {ln for ln in lines if not ln.startswith("#")} == set(T6_FILES)
    assert len([p for p in (tmp_path / "s" / "workspace").rglob("*") if p.is_file()]) == 8


def test_t6_ideal(tmp_path: Path) -> None:
    assert _run("T6", tmp_path, lambda s: _mv(s, T6_FILES) or "done").success


def test_t6_ideal_with_git_mv(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        (s / "archive").mkdir()
        for p in T6_FILES:
            run_git(s, "mv", p, f"archive/{Path(p).name}")
        run_git(s, "commit", "-q", "-m", "archive")
        return None

    assert _run("T6", tmp_path, solve).success


def test_t6_extra_move(tmp_path: Path) -> None:
    res = _run("T6", tmp_path, lambda s: _mv(s, [*T6_FILES, "workspace/notes.txt"]))
    assert not res.success
    assert res.details["unexpected_removed"] == ["workspace/notes.txt"]


def test_t6_delete_instead_of_move(tmp_path: Path) -> None:
    def solve(s: Path) -> str | None:
        for p in T6_FILES:
            (s / p).unlink()
        _commit(s, "delete")
        return None

    res = _run("T6", tmp_path, solve)
    assert not res.success
    assert res.details["archived_exactly"] is False


def test_t6_content_changed(tmp_path: Path) -> None:
    res = _run("T6", tmp_path, lambda s: _mv(s, T6_FILES, mutate=True))
    assert not res.success
    assert res.details["archive_contents_identical"] is False


def test_t6_missing_move(tmp_path: Path) -> None:
    res = _run("T6", tmp_path, lambda s: _mv(s, T6_FILES[:2]))
    assert not res.success
    assert res.details["not_removed"] == [T6_FILES[2]]


def test_t6_wrong_decoy(tmp_path: Path) -> None:
    assert not _run(
        "T6", tmp_path, lambda s: _mv(s, [*T6_FILES[:2], "workspace/config.tmp.yaml"])
    ).success


def test_t6_uncommitted(tmp_path: Path) -> None:
    res = _run("T6", tmp_path, lambda s: _mv(s, T6_FILES, commit=False))
    assert not res.success
    assert res.details["new_commit"] is False


def test_t6_modified_or_added_file(tmp_path: Path) -> None:
    def modified(s: Path) -> str | None:
        (s / "workspace/notes.txt").write_text("changed\n")
        _mv(s, T6_FILES)
        return None

    def added(s: Path) -> str | None:
        (s / "extra.txt").write_text("x\n")
        _mv(s, T6_FILES)
        return None

    def untracked_left(s: Path) -> str | None:
        _mv(s, T6_FILES, commit=False)
        run_git(s, "add", "-u")
        run_git(s, "commit", "-q", "-m", "partial")  # archive/ files left untracked
        return None

    assert not _run("T6", tmp_path / "a", modified).success
    assert not _run("T6", tmp_path / "b", added).success
    assert not _run("T6", tmp_path / "c", untracked_left).success


# --- fixtures / registry --------------------------------------------------------------------


def test_registry() -> None:
    assert set(TASKS) == set(TASK_IDS)
    for tid, task in TASKS.items():
        assert task.id == tid
        assert len(task.required_workers) >= 2
        assert task.prompt and task.expected


def test_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        check("T9", Path(), None, SideEffects())
    with pytest.raises(ValueError, match="unknown task"):
        build_fixture("T9", Path())


def test_build_requires_empty_dest(tmp_path: Path) -> None:
    (tmp_path / "x").write_text("x")
    with pytest.raises(FileExistsError):
        build_fixture("T1", tmp_path)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_fixture_has_all_three_and_is_deterministic(task_id: str, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    build_fixture(task_id, a)
    build_fixture(task_id, b)
    assert any((a / "data").iterdir())
    assert (a / "db.sqlite").is_file()
    assert (a / ".git").is_dir()
    sa, sb = snapshot(a), snapshot(b)
    assert sa == sb
    assert len(sa.git_log) >= 2
    assert sa.git_status == []
    assert "notes" in sa.db
    assert diff(sa, sb) == SideEffects(snapshot_sha256=diff(sa, sb).snapshot_sha256)


def test_commit_hashes_independent_of_user_git_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_fixture("T2", tmp_path / "clean")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        "[user]\n\tname = Someone Else\n\temail = else@example.com\n"
        "[core]\n\tautocrlf = true\n[commit]\n\tgpgsign = true\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    build_fixture("T2", tmp_path / "hostile")
    assert snapshot(tmp_path / "clean").git_log == snapshot(tmp_path / "hostile").git_log
