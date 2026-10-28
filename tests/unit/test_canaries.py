# ruff: noqa: S311
import random
import re

import pytest

from mcp_chaos.proxy.canaries import TEMPLATES, derive_canary, render_template


def test_derive_canary_format_and_determinism() -> None:
    c = derive_canary(random.Random(1))
    assert re.fullmatch(r"CANARY-[0-9a-f]{8}", c)
    assert c == derive_canary(random.Random(1))
    assert c != derive_canary(random.Random(2))


@pytest.mark.parametrize("tid", sorted(TEMPLATES))
def test_templates_benign(tid: str) -> None:
    text = render_template(tid, "CANARY-deadbeef")
    assert "CANARY-deadbeef" in text
    assert "{" not in text
    assert "http" not in text.lower()


def test_four_templates() -> None:
    assert set(TEMPLATES) == {"write_file", "echo_in_answer", "call_tool", "delegate"}
