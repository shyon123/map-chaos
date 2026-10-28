"""Smoke test for package scaffold."""

import mcp_chaos
from mcp_chaos.bench.cli import main


def test_version() -> None:
    assert mcp_chaos.__version__ == "0.1.0"


def test_main_no_command_returns_zero() -> None:
    assert main([]) == 0
