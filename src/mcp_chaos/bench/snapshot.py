"""Sandbox snapshot/diff used to compute side effects and detect canaries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mcp_chaos.schemas.results import SideEffects
from mcp_chaos.targets.fixtures import run_git

DB_NAME = "db.sqlite"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TableSnap(_Frozen):
    rows: int
    sha256: str


class Snapshot(_Frozen):
    files: dict[str, str]  # relative path -> sha256 of contents (excludes .git/)
    git_head: str | None
    git_log: list[str]  # commit hashes, newest first
    git_status: list[str]  # `git status --porcelain` lines
    db: dict[str, TableSnap]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scratch_files(scratch: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(scratch.rglob("*")):
        rel = p.relative_to(scratch)
        if ".git" in rel.parts or not p.is_file():
            continue
        out[rel.as_posix()] = p
    return out


def _git_state(scratch: Path) -> tuple[str | None, list[str], list[str]]:
    if not (scratch / ".git").exists():
        return None, [], []
    head = run_git(scratch, "rev-parse", "--verify", "-q", "HEAD", check=False).strip() or None
    log = run_git(scratch, "log", "--format=%H", check=False).split() if head else []
    status = run_git(scratch, "status", "--porcelain", "--untracked-files=all").splitlines()
    return head, log, status


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [str(r[0]) for r in cur.fetchall()]


def connect_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)


def _db_state(scratch: Path) -> dict[str, TableSnap]:
    path = scratch / DB_NAME
    if not path.is_file():
        return {}
    out: dict[str, TableSnap] = {}
    conn = connect_ro(path)
    try:
        for table in _tables(conn):
            q = _quote(table)
            ncols = len(conn.execute(f"SELECT * FROM {q} LIMIT 0").description)  # noqa: S608
            order = ",".join(str(i + 1) for i in range(ncols))
            h = hashlib.sha256()
            n = 0
            for row in conn.execute(f"SELECT * FROM {q} ORDER BY {order}"):  # noqa: S608
                h.update(repr(tuple(row)).encode())
                n += 1
            out[table] = TableSnap(rows=n, sha256=h.hexdigest())
    finally:
        conn.close()
    return out


def snapshot(scratch: Path) -> Snapshot:
    """Capture file hashes, git state and per-table DB digests of a scratch dir."""
    files = {rel: _sha256(p.read_bytes()) for rel, p in _scratch_files(scratch).items()}
    head, log, status = _git_state(scratch)
    return Snapshot(
        files=files, git_head=head, git_log=log, git_status=status, db=_db_state(scratch)
    )


def diff(before: Snapshot, after: Snapshot) -> SideEffects:
    """Compute the side effects between two snapshots."""
    b, a = before.files, after.files
    seen = set(before.git_log)
    tables = sorted(set(before.db) | set(after.db))
    db_changes: dict[str, int] = {}
    for t in tables:
        delta = (after.db[t].rows if t in after.db else 0) - (
            before.db[t].rows if t in before.db else 0
        )
        if delta:
            db_changes[t] = delta
    canonical = json.dumps(after.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return SideEffects(
        fs_added=sorted(set(a) - set(b)),
        fs_removed=sorted(set(b) - set(a)),
        fs_modified=sorted(p for p in set(a) & set(b) if a[p] != b[p]),
        git_new_commits=[c for c in after.git_log if c not in seen],
        db_changes=db_changes,
        snapshot_sha256=_sha256(canonical.encode()),
    )


def find_canary(scratch: Path, snap: Snapshot, final_answer: str | None, canary: str) -> list[str]:
    """Locations of ``canary``: ``file:<path>``, ``db:<table>.<col>`` or ``answer``."""
    needle = canary.encode()
    hits: list[str] = []
    for rel in sorted(snap.files):
        path = scratch / rel
        if path.is_file() and (needle in path.read_bytes() or canary in rel):
            hits.append(f"file:{rel}")
    db_path = scratch / DB_NAME
    if snap.db and db_path.is_file():
        conn = connect_ro(db_path)
        try:
            for table in sorted(snap.db):
                cur = conn.execute(f"SELECT * FROM {_quote(table)}")  # noqa: S608
                cols = [d[0] for d in cur.description]
                found: set[str] = set()
                for row in cur:
                    for col, val in zip(cols, row, strict=True):
                        text = val.decode(errors="replace") if isinstance(val, bytes) else str(val)
                        if canary in text:
                            found.add(f"db:{table}.{col}")
                hits.extend(sorted(found))
        finally:
            conn.close()
    if final_answer is not None and canary in final_answer:
        hits.append("answer")
    return hits
