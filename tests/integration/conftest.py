"""Integration fixtures."""

import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_script_path() -> Path:
    return FIXTURES / "fake_script.json"


def fake_server_cmd(script: Path) -> list[str]:
    return [sys.executable, "-m", "mcp_chaos.servers.fake_server", "--script", str(script)]
