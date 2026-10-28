"""Sans-IO newline framing for the JSON-RPC stdio relay."""

from __future__ import annotations

DEFAULT_MAX_LINE_BYTES = 64 * 1024 * 1024


class FramingError(Exception):
    """A line exceeded the configured maximum size."""


class LineSplitter:
    """Split a byte stream into lines on ``\\n``.

    Lines exclude the trailing ``\\n``; a trailing ``\\r`` is preserved byte-exact.
    """

    def __init__(self, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES) -> None:
        self.max_line_bytes = max_line_bytes
        self._buf = bytearray()
        self._scan = 0  # bytes of _buf already searched for a newline

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf += chunk
        lines: list[bytes] = []
        start = 0
        scan = self._scan
        while True:
            nl = self._buf.find(b"\n", scan)
            if nl < 0:
                break
            if nl - start > self.max_line_bytes:
                raise FramingError(f"line exceeds {self.max_line_bytes} bytes")
            lines.append(bytes(self._buf[start:nl]))
            start = scan = nl + 1
        del self._buf[:start]
        self._scan = len(self._buf)
        if len(self._buf) > self.max_line_bytes:
            raise FramingError(f"line exceeds {self.max_line_bytes} bytes")
        return lines

    def flush(self) -> bytes | None:
        """Return the trailing partial line (no ``\\n`` seen), if any, and reset."""
        if not self._buf:
            return None
        rest = bytes(self._buf)
        self._buf.clear()
        self._scan = 0
        return rest
