"""Tests for sandbox snapshot, diff and canary search."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mcp_chaos.bench.snapshot import diff, find_canary, snapshot
from mcp_chaos.targets.fixtures import build_fixture, run_git

CANARY = "CANARY-7f3a9c1e"


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    path = tmp_path / "s"
    build_fixture("T6", path)
    return path


def test_snapshot_contents(scratch: Path) -> None:
    snap = snapshot(scratch)
    assert "db.sqlite" in snap.files
    assert "workspace/notes.txt" in snap.files
    assert not any(p.startswith(".git/") for p in snap.files)
    assert snap.git_head == snap.git_log[0]
    assert snap.git_status == []
    assert snap.db["jobs"].rows == 3
    assert snap.db["notes"].rows == 0


def test_snapshot_without_git_or_db(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    snap = snapshot(tmp_path)
    assert snap.git_head is None
    assert snap.git_log == []
    assert snap.db == {}
    assert list(snap.files) == ["a.txt"]


def test_snapshot_git_without_commits(tmp_path: Path) -> None:
    run_git(tmp_path, "init", "-q")
    snap = snapshot(tmp_path)
    assert snap.git_head is None
    assert snap.git_log == []


def test_diff_identical(scratch: Path) -> None:
    snap = snapshot(scratch)
    eff = diff(snap, snapshot(scratch))
    assert eff.fs_added == eff.fs_removed == eff.fs_modified == []
    assert eff.git_new_commits == []
    assert eff.db_changes == {}
    assert len(eff.snapshot_sha256) == 64


def test_diff_fs_git_db(scratch: Path) -> None:
    before = snapshot(scratch)
    (scratch / "workspace/notes.txt").write_text("changed\n")
    (scratch / "workspace/README.md").unlink()
    (scratch / "new_b.txt").write_text("b")
    (scratch / "new_a.txt").write_text("a")
    run_git(scratch, "add", "-A")
    run_git(scratch, "commit", "-q", "-m", "work")
    conn = sqlite3.connect(scratch / "db.sqlite")
    conn.execute("INSERT INTO notes (body) VALUES ('x'), ('y')")
    conn.execute("DELETE FROM jobs WHERE id = 3")
    conn.commit()
    conn.close()
    after = snapshot(scratch)
    eff = diff(before, after)
    assert eff.fs_added == ["new_a.txt", "new_b.txt"]
    assert eff.fs_removed == ["workspace/README.md"]
    assert eff.fs_modified == ["db.sqlite", "workspace/notes.txt"]
    assert eff.git_new_commits == [after.git_head]
    assert eff.db_changes == {"notes": 2, "jobs": -1}
    assert eff.snapshot_sha256 != diff(before, before).snapshot_sha256
    assert eff.snapshot_sha256 == diff(before, after).snapshot_sha256


def test_diff_detects_same_count_db_update(scratch: Path) -> None:
    before = snapshot(scratch)
    conn = sqlite3.connect(scratch / "db.sqlite")
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = 1")
    conn.commit()
    conn.close()
    after = snapshot(scratch)
    assert before.db["jobs"].rows == after.db["jobs"].rows
    assert before.db["jobs"].sha256 != after.db["jobs"].sha256
    assert diff(before, after).db_changes == {}


def test_diff_new_and_dropped_tables(scratch: Path) -> None:
    before = snapshot(scratch)
    conn = sqlite3.connect(scratch / "db.sqlite")
    conn.execute('CREATE TABLE "odd ""name" (a INTEGER, b TEXT)')
    conn.execute('INSERT INTO "odd ""name" VALUES (1, \'x\')')
    conn.execute("DROP TABLE jobs")
    conn.commit()
    conn.close()
    assert diff(before, snapshot(scratch)).db_changes == {'odd "name': 1, "jobs": -3}


def test_git_status_and_dirty_tree(scratch: Path) -> None:
    (scratch / "untracked.txt").write_text("u")
    (scratch / "workspace/notes.txt").write_text("m")
    status = snapshot(scratch).git_status
    assert "?? untracked.txt" in status
    assert " M workspace/notes.txt" in status


def test_find_canary_in_file(scratch: Path) -> None:
    (scratch / "out.txt").write_text(f"hello {CANARY}\n")
    (scratch / "workspace" / "other.txt").write_text("nothing")
    snap = snapshot(scratch)
    assert find_canary(scratch, snap, None, CANARY) == ["file:out.txt"]


def test_find_canary_in_filename(scratch: Path) -> None:
    (scratch / f"{CANARY}.txt").write_text("x")
    snap = snapshot(scratch)
    assert find_canary(scratch, snap, None, CANARY) == [f"file:{CANARY}.txt"]


def test_find_canary_in_db(scratch: Path) -> None:
    conn = sqlite3.connect(scratch / "db.sqlite")
    conn.execute("INSERT INTO notes (body) VALUES (?), (?)", (f"x {CANARY}", f"{CANARY}!"))
    conn.execute("INSERT INTO jobs VALUES (9, ?)", (CANARY,))
    conn.execute("CREATE TABLE blobs (data BLOB)")
    conn.execute("INSERT INTO blobs VALUES (?)", (CANARY.encode(),))
    conn.commit()
    conn.close()
    snap = snapshot(scratch)
    hits = find_canary(scratch, snap, None, CANARY)
    assert "db:notes.body" in hits
    assert "db:jobs.status" in hits
    assert "db:blobs.data" in hits
    assert hits.count("db:notes.body") == 1


def test_find_canary_in_answer_and_absent(scratch: Path) -> None:
    snap = snapshot(scratch)
    assert find_canary(scratch, snap, f"done. {CANARY}", CANARY) == ["answer"]
    assert find_canary(scratch, snap, "done", CANARY) == []
    assert find_canary(scratch, snap, None, CANARY) == []


def test_find_canary_without_db(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text(CANARY)
    snap = snapshot(tmp_path)
    assert find_canary(tmp_path, snap, CANARY, CANARY) == ["file:a.txt", "answer"]
