from __future__ import annotations

import pytest

from mcp_chaos.bench.stats import holm_adjust, mcnemar_exact, paired_bootstrap_diff_ci, wilson_ci


def test_wilson_known_values() -> None:
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == pytest.approx(0.0)
    assert hi0 == pytest.approx(0.2775, abs=1e-3)
    lo1, hi1 = wilson_ci(10, 10)
    assert hi1 == pytest.approx(1.0)
    assert lo1 == pytest.approx(0.7225, abs=1e-3)


def test_wilson_validation() -> None:
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_mcnemar_exact_known() -> None:
    assert mcnemar_exact(2, 8) == pytest.approx(0.109375)
    assert mcnemar_exact(8, 2) == pytest.approx(0.109375)
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_holm() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    assert holm_adjust([0.5, 0.2]) == pytest.approx([0.5, 0.4])
    assert holm_adjust([0.6, 0.7, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_bootstrap_seeded_and_sensible() -> None:
    before = [0] * 20
    after = [1] * 20
    assert paired_bootstrap_diff_ci(before, after) == (1.0, 1.0)
    b = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    a = [1, 1, 1, 1, 1, 0, 1, 1, 0, 1]
    ci1 = paired_bootstrap_diff_ci(b, a, seed=3)
    ci2 = paired_bootstrap_diff_ci(b, a, seed=3)
    assert ci1 == ci2
    assert 0 < ci1[0] <= 0.4 <= ci1[1] <= 0.8
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([1], [1, 0])
