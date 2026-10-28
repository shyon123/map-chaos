"""On-disk response cache: ``cache_dir/<key[:2]>/<key>.json``, atomic writes."""

from __future__ import annotations

import base64
import contextlib
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class StoredResponse(BaseModel):
    """A recorded upstream response (also the cassette payload shape)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: int
    headers: dict[str, str]
    body_b64: str | None = None
    sse_chunks_b64: list[str] | None = None

    @classmethod
    def from_body(cls, status: int, headers: dict[str, str], body: bytes) -> StoredResponse:
        return cls(status=status, headers=headers, body_b64=base64.b64encode(body).decode())

    @classmethod
    def from_sse(cls, status: int, headers: dict[str, str], chunks: list[bytes]) -> StoredResponse:
        return cls(
            status=status,
            headers=headers,
            sse_chunks_b64=[base64.b64encode(c).decode() for c in chunks],
        )

    @property
    def body(self) -> bytes:
        return base64.b64decode(self.body_b64 or "")

    @property
    def chunks(self) -> list[bytes] | None:
        if self.sse_chunks_b64 is None:
            return None
        return [base64.b64decode(c) for c in self.sse_chunks_b64]


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> StoredResponse | None:
        p = self.path(key)
        try:
            return StoredResponse.model_validate_json(p.read_bytes())
        except (OSError, ValueError):
            return None

    def put(self, key: str, resp: StoredResponse) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(resp.model_dump_json().encode())
            Path(tmp).replace(p)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp).unlink()
            raise
