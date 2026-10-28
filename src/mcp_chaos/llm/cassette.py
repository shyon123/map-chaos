"""Per-run JSONL cassettes: record by appending, replay by key then by sequence number."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mcp_chaos.llm.cache import StoredResponse


class CassetteEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    key: str
    status: int
    headers: dict[str, str]
    body_b64: str | None = None
    sse_chunks_b64: list[str] | None = None

    @classmethod
    def from_response(cls, seq: int, key: str, r: StoredResponse) -> CassetteEntry:
        return cls(
            seq=seq,
            key=key,
            status=r.status,
            headers=r.headers,
            body_b64=r.body_b64,
            sse_chunks_b64=r.sse_chunks_b64,
        )

    def to_response(self) -> StoredResponse:
        return StoredResponse(
            status=self.status,
            headers=self.headers,
            body_b64=self.body_b64,
            sse_chunks_b64=self.sse_chunks_b64,
        )


class CassetteWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.jsonl"

    def append(self, run_id: str, entry: CassetteEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path(run_id).open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")


class CassetteReader:
    """Loads a run's cassette; repeated identical keys replay in recorded order."""

    def __init__(self, directory: Path, run_id: str, seq_fallback: bool = True) -> None:
        self.seq_fallback = seq_fallback
        self._by_key: dict[str, list[CassetteEntry]] = {}
        self._by_seq: dict[int, CassetteEntry] = {}
        self._used: dict[str, int] = {}
        p = directory / f"{run_id}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = CassetteEntry.model_validate_json(line)
                self._by_key.setdefault(e.key, []).append(e)
                self._by_seq[e.seq] = e

    def lookup(self, key: str, seq: int) -> StoredResponse | None:
        entries = self._by_key.get(key)
        if entries:
            i = self._used.get(key, 0)
            self._used[key] = i + 1
            return entries[min(i, len(entries) - 1)].to_response()
        if self.seq_fallback:
            e = self._by_seq.get(seq)
            if e is not None:
                return e.to_response()
        return None
