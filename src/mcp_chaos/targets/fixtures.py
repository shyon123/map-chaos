"""Deterministic fixture builders for benchmark tasks T1-T6.

``build_fixture(task_id, dest)`` materializes a scratch directory containing static inputs
(``data/`` and friends), ``db.sqlite`` and a git repository with a fully deterministic history.
Building the same task twice yields identical file trees and identical commit hashes.
"""

from __future__ import annotations

import csv
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from mcp_chaos.bench.safe_repo import run_git as _run_git

FIXTURE_DATA = Path(__file__).parent / "fixture_data"
TASK_IDS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T6")

AUTHOR_NAME = "Bench Author"
AUTHOR_EMAIL = "bench@example.invalid"
BASE_DATE = "2024-03-01T12:00:00+0000"
GITIGNORE = "db.sqlite\n"
_IDENTITY: dict[str, str] = {
    "GIT_AUTHOR_NAME": AUTHOR_NAME,
    "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
    "GIT_COMMITTER_NAME": AUTHOR_NAME,
    "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
}

# The exact ``.git/config`` of every built fixture. ``build_fixture`` writes it right after
# ``git init`` (so it is platform independent); the host restores it into every sanitized copy.
PRISTINE_GIT_CONFIG: bytes = (
    b"[core]\n"
    b"\trepositoryformatversion = 0\n"
    b"\tfilemode = true\n"
    b"\tbare = false\n"
    b"\tlogallrefupdates = true\n"
)

# T2: commits made after tag v1.0, oldest first: (subject, relative path, file content).
T2_COMMITS: tuple[tuple[str, str, str], ...] = (
    (
        "Fix off-by-one in pagination",
        "src/pagination.py",
        "def pages(total: int, size: int) -> int:\n    return (total + size - 1) // size\n",
    ),
    (
        "Add CSV export endpoint",
        "src/export.py",
        "def to_csv(rows: list[list[str]]) -> str:\n"
        '    return "\\n".join(",".join(r) for r in rows)\n',
    ),
    ("Bump dependency pins", "requirements.txt", "requests==2.31.0\nflask==3.0.2\n"),
    (
        "Refactor config loader",
        "src/config.py",
        "import json\n\n\ndef load(path: str) -> dict:\n    with open(path) as f:\n"
        "        return json.load(f)\n",
    ),
    (
        "Document retry behaviour",
        "docs/retries.md",
        "# Retries\n\nRequests are retried three times with exponential backoff.\n",
    ),
    (
        "Handle empty input in parser",
        "src/parser.py",
        "def parse(text: str) -> list[str]:\n    if not text:\n        return []\n"
        "    return text.split()\n",
    ),
)
_T2_PARSER_V1 = "def parse(text: str) -> list[str]:\n    return text.split()\n"

_NOTES_DDL = "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)"

T5_EMPLOYEES: tuple[tuple[int, str, str, int], ...] = (
    (1, "Dana Whitfield", "Platform", 94500),
    (2, "Dana Whitman", "Design", 71000),
    (3, "Elliot Park", "Data", 88000),
    (4, "Farah Nasser", "Support", 52000),
    (5, "Gus Lindqvist", "Platform", 101000),
)
# Bonus rates (percent of base salary) per department. The other employees' bonuses and every
# 2023 bonus are the distractor values for T5.
T5_RATES_2024: dict[str, int] = {"Platform": 12, "Data": 10, "Design": 8, "Support": 6}
T5_RATES_2023: dict[str, int] = {"Platform": 9, "Data": 9, "Design": 7, "Support": 5}


def run_git(repo: Path, *args: str, date: str | None = None, check: bool = True) -> str:
    """Run ``git`` in ``repo`` hermetically with the bench identity; return stdout."""
    return _run_git(repo, *args, date=date, check=check, extra_env=_IDENTITY)


def _copy_static(task_id: str, dest: Path) -> None:
    shutil.copytree(FIXTURE_DATA / task_id, dest, dirs_exist_ok=True, copy_function=shutil.copyfile)


def _commit_date(n: int) -> str:
    return f"2024-03-{1 + n:02d}T12:00:00+0000"


def _commit(repo: Path, n: int, message: str) -> None:
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message, date=_commit_date(n))


def _write(dest: Path, rel: str, content: str) -> None:
    path = dest / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, newline="\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _create_db(dest: Path, task_id: str) -> None:
    ddl: list[str] = [_NOTES_DDL]
    inserts: list[tuple[str, Sequence[Sequence[object]]]] = []
    if task_id == "T1":
        ddl.append(
            "CREATE TABLE sales (order_id TEXT PRIMARY KEY, region TEXT NOT NULL, "
            "product TEXT NOT NULL, units INTEGER NOT NULL, unit_price REAL NOT NULL)"
        )
    elif task_id == "T2":
        ddl.append("CREATE TABLE releases (tag TEXT PRIMARY KEY, released_on TEXT NOT NULL)")
        inserts.append(("releases", [("v0.9", "2024-02-01"), ("v1.0", "2024-03-02")]))
    elif task_id == "T3":
        ddl.append("CREATE TABLE services (name TEXT PRIMARY KEY, port INTEGER NOT NULL)")
        inserts.append(("services", [("inventory-sync", 8080), ("billing", 8081)]))
    elif task_id == "T4":
        ddl.append(
            "CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer TEXT NOT NULL, "
            "amount REAL NOT NULL)"
        )
        rows = _read_csv(dest / "data" / "orders.csv")
        db_rows: list[tuple[str, str, float]] = []
        for r in rows:
            oid, amount = r["order_id"], float(r["amount"])
            if oid == "ORD-1007":  # missing from the database
                continue
            if oid == "ORD-1004":  # amount differs
                amount = 54.0
            if oid == "ORD-1013":  # amount differs
                amount = 199.89
            db_rows.append((oid, r["customer"], amount))
        db_rows.append(("ORD-1099", "Zephyr Ltd", 10.0))  # extra in the database
        inserts.append(("orders", db_rows))
    elif task_id == "T5":
        ddl.append(
            "CREATE TABLE employees (emp_id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "dept TEXT NOT NULL, base_salary INTEGER NOT NULL)"
        )
        inserts.append(("employees", list(T5_EMPLOYEES)))
    elif task_id == "T6":
        ddl.append("CREATE TABLE jobs (id INTEGER PRIMARY KEY, status TEXT NOT NULL)")
        inserts.append(("jobs", [(1, "done"), (2, "done"), (3, "pending")]))
    conn = sqlite3.connect(dest / "db.sqlite")
    try:
        for stmt in ddl:
            conn.execute(stmt)
        for table, rows_ in inserts:
            marks = ",".join("?" * len(rows_[0]))
            conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows_)  # noqa: S608
        conn.commit()
    finally:
        conn.close()


def _build_history(dest: Path, task_id: str) -> None:
    run_git(dest, "init", "-q", "--template=")
    (dest / ".git" / "config").write_bytes(PRISTINE_GIT_CONFIG)
    if task_id == "T2":
        _write(dest, ".gitignore", GITIGNORE)
        _write(dest, "README.md", "# Sample service\n")
        _commit(dest, 0, "Initial commit")
        _write(dest, "src/parser.py", _T2_PARSER_V1)
        _commit(dest, 1, "Add parser module")
        run_git(dest, "tag", "v1.0")
        for i, (subject, rel, content) in enumerate(T2_COMMITS):
            _write(dest, rel, content)
            _commit(dest, 2 + i, subject)
        return
    _write(dest, ".gitignore", GITIGNORE)
    _write(dest, "README.md", f"# Scratch project for {task_id}\n")
    # Commit the skeleton first, then the task inputs.
    run_git(dest, "add", ".gitignore", "README.md")
    run_git(dest, "commit", "-q", "-m", "Initial commit", date=_commit_date(0))
    _commit(dest, 1, "Add task inputs")


def build_fixture(task_id: str, dest: Path) -> None:
    """Create a fresh scratch directory for ``task_id`` at ``dest`` (must be absent or empty)."""
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown task id: {task_id!r}")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"fixture destination is not empty: {dest}")
    _copy_static(task_id, dest)
    _create_db(dest, task_id)
    _build_history(dest, task_id)
