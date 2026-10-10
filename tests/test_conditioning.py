import numpy as np
import pytest

from ringdown.conditioning import (TASK_EXPONENTS, conditioning_report,
                                   mode_gaps)

MODES = [(1, -2, 2, 2, 0), (1, -2, 2, 2, 1), (1, -2, 3, 2, 0)]


def test_mode_gaps_basic():
    gaps = mode_gaps(MODES, chi=0.7)
    assert len(gaps) == 3
    for gap in gaps.values():
        assert gap > 0
    # dimensionful version scales by 1 / (M T_MSUN)
    gaps_hz = mode_gaps(MODES, chi=0.7, m_msun=70.0)
    ratios = [gaps_hz[k] / gaps[k] for k in gaps]
    assert np.allclose(ratios, ratios[0])


def test_known_225_226_crossing_gap():
    """Anchor: the (2,2,5)-(2,2,6) avoided crossing near chi ~ 0.897."""
    gaps = mode_gaps([(1, -2, 2, 2, 5), (1, -2, 2, 2, 6)], chi=0.8969)
    gap = list(gaps.values())[0]
    assert gap == pytest.approx(0.0667, abs=5e-3)


def test_conditioning_report_structure():
    report = conditioning_report(MODES, chi=0.7)
    assert len(report) == 3
    assert report[0]["smallest_gap"] is True
    assert all(not r["smallest_gap"] for r in report[1:])
    gaps = [r["gap"] for r in report]
    assert gaps == sorted(gaps)
    for r in report:
        for task, p in TASK_EXPONENTS.items():
            assert r["amplification"][task] == pytest.approx(r["gap"] ** (-p))
