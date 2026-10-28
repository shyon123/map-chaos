# ruff: noqa: S105, S106
from __future__ import annotations

import json

import pytest

from mcp_chaos.log import configure_logging, get_logger, redact_processor


def test_redacts_secret_keys_case_insensitive() -> None:
    out = redact_processor(
        None,
        "info",
        {
            "event": "x",
            "API_KEY": "abc",
            "Authorization": "Bearer z",
            "headers": {"X-Api-Key": "k", "ok": "v"},
            "access_token": "t",
            "client_secret": "s",
            "Password": "p",
            "input_tokens": 5,
            "items": [{"token": "q"}],
        },
    )
    assert out["API_KEY"] == "[REDACTED]"
    assert out["Authorization"] == "[REDACTED]"
    assert out["headers"] == {"X-Api-Key": "[REDACTED]", "ok": "v"}
    assert out["access_token"] == out["client_secret"] == out["Password"] == "[REDACTED]"
    assert out["input_tokens"] == 5
    assert out["items"] == [{"token": "[REDACTED]"}]


def test_scrubs_sk_ant_anywhere() -> None:
    out = redact_processor(
        None, "info", {"event": "key sk-ant-api03-AbC_d-1 leaked", "t": ("sk-ant-x1", 3)}
    )
    assert "sk-ant" not in out["event"]
    assert out["event"] == "key [REDACTED] leaked"
    assert out["t"] == ("[REDACTED]", 3)


def test_logs_to_stderr_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=True)
    log = get_logger("t")
    log.debug("hidden")
    log.info("hello", api_key="sk-ant-abc", note="sk-ant-zzz", n=1)
    cap = capsys.readouterr()
    assert cap.out == ""
    rec = json.loads(cap.err.strip())
    assert rec["event"] == "hello"
    assert rec["api_key"] == "[REDACTED]"
    assert rec["note"] == "[REDACTED]"
    assert rec["level"] == "info"
    assert "hidden" not in cap.err


def test_console_mode_debug_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("debug")
    get_logger("t").debug("visible", password="pw")
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "visible" in cap.err
    assert "pw" not in cap.err
    assert "[REDACTED]" in cap.err


def test_logging_follows_replaced_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    from mcp_chaos.log import configure_logging, get_logger

    configure_logging()
    closed = io.StringIO()
    monkeypatch.setattr(sys, "stderr", closed)
    closed.close()
    fresh = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fresh)
    get_logger("t").info("still_works")
    assert "still_works" in fresh.getvalue()
