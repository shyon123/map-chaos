import pytest

from mcp_chaos.proxy.framing import FramingError, LineSplitter


def test_basic_lines() -> None:
    s = LineSplitter()
    assert s.feed(b"a\nb\nc") == [b"a", b"b"]
    assert s.feed(b"d\n") == [b"cd"]
    assert s.flush() is None


def test_crlf_preserved() -> None:
    assert LineSplitter().feed(b"a\r\nb\n") == [b"a\r", b"b"]


def test_empty_lines() -> None:
    assert LineSplitter().feed(b"\n\n") == [b"", b""]


def test_flush_trailing() -> None:
    s = LineSplitter()
    s.feed(b"partial")
    assert s.flush() == b"partial"
    assert s.flush() is None


def test_max_line_partial() -> None:
    s = LineSplitter(max_line_bytes=4)
    with pytest.raises(FramingError):
        s.feed(b"12345")


def test_max_line_complete() -> None:
    s = LineSplitter(max_line_bytes=4)
    with pytest.raises(FramingError):
        s.feed(b"12345\n")
    assert LineSplitter(max_line_bytes=4).feed(b"1234\n") == [b"1234"]
