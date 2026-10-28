from __future__ import annotations

from pathlib import Path

import pytest

from mcp_chaos.llm.cache import DiskCache, StoredResponse
from mcp_chaos.llm.cassette import CassetteEntry, CassetteReader, CassetteWriter


def test_roundtrip_and_layout(tmp_path: Path) -> None:
    c = DiskCache(tmp_path)
    key = "ab" + "0" * 62
    assert c.get(key) is None
    c.put(key, StoredResponse.from_body(200, {"content-type": "application/json"}, b"\x00body"))
    assert (tmp_path / "ab" / f"{key}.json").exists()
    got = c.get(key)
    assert got is not None and got.body == b"\x00body" and got.chunks is None
    assert not list((tmp_path / "ab").glob(".tmp-*"))


def test_sse_roundtrip_and_overwrite(tmp_path: Path) -> None:
    c = DiskCache(tmp_path)
    c.put("cd1", StoredResponse.from_sse(200, {}, [b"a", b"bc"]))
    got = c.get("cd1")
    assert got is not None and got.chunks == [b"a", b"bc"]
    c.put("cd1", StoredResponse.from_body(200, {}, b"x"))
    got = c.get("cd1")
    assert got is not None and got.body == b"x"


def test_corrupt_file_is_miss(tmp_path: Path) -> None:
    c = DiskCache(tmp_path)
    p = c.path("ef1")
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    assert c.get("ef1") is None


def test_failed_write_leaves_no_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = DiskCache(tmp_path)

    def boom(self: Path, target: Path) -> Path:
        raise OSError("nope")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        c.put("aa1", StoredResponse.from_body(200, {}, b"x"))
    assert not list(tmp_path.rglob(".tmp-*"))


def test_cassette_key_order_and_seq_fallback(tmp_path: Path) -> None:
    w = CassetteWriter(tmp_path / "cas")
    r1 = StoredResponse.from_body(200, {}, b"one")
    r2 = StoredResponse.from_body(200, {}, b"two")
    w.append("run", CassetteEntry.from_response(1, "k", r1))
    w.append("run", CassetteEntry.from_response(2, "k", r2))
    w.append("run", CassetteEntry.from_response(3, "other", r1))
    rd = CassetteReader(tmp_path / "cas", "run")
    got = [rd.lookup("k", 9), rd.lookup("k", 9), rd.lookup("k", 9)]
    assert [g.body for g in got if g] == [b"one", b"two", b"two"]
    fb = rd.lookup("missing", 3)
    assert fb is not None and fb.body == b"one"
    assert rd.lookup("missing", 99) is None
    strict = CassetteReader(tmp_path / "cas", "run", seq_fallback=False)
    assert strict.lookup("missing", 3) is None
    assert CassetteReader(tmp_path / "cas", "nonexistent").lookup("k", 1) is None
