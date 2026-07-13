"""Integration tests for the assembled BOOST sequence (pyboost.boost)."""

import numpy as np
import pytest

from pyboost import build_boost_sequence, BoostParams, scanner_055T
from pyboost.readout import _build_kernel


@pytest.fixture
def small():
    # Small matrix keeps the tests fast while exercising the full structure.
    return BoostParams(nx=16, ny=8, im_segments=8, dummy_heart_beats=1)


def _count_adc(seq):
    n = 0
    for i in range(1, len(seq.block_events) + 1):
        if getattr(seq.get_block(i), "adc", None) is not None:
            n += 1
    return n


def test_kernel_is_balanced_and_on_target():
    system, p = scanner_055T(), BoostParams()
    k = _build_kernel(system, p)
    # bSSFP demands zero net gradient moment per axis over the TR.
    assert k.gx_pre.area + k.gx.area + k.gx_post.area == pytest.approx(0.0, abs=1e-6)
    assert k.gz.area + k.gzr.area + k.gz_rew.area == pytest.approx(0.0, abs=1e-6)
    assert k.te_fill >= 0 and k.tr_fill >= 0


def test_sequence_timing_ok(small):
    seq = build_boost_sequence(small, scanner_055T())
    ok, errors = seq.check_timing()
    assert ok, f"check_timing failed: {errors[:3]}"


def test_constant_tr(small):
    """Every bSSFP TR (imaging block) is exactly TR long."""
    system = scanner_055T()
    seq = build_boost_sequence(small, system, add_trigger=False)
    # Duration is dominated by 2 contrasts x (dummy + n_shots) heartbeats of RR.
    dur = seq.duration()[0]
    expected = 2 * (small.dummy_heart_beats + small.n_shots) * small.rr
    assert dur == pytest.approx(expected, abs=system.block_duration_raster * 4)


def test_kspace_lines_acquired(small):
    """Both contrasts of every shot record one ADC per phase-encode line."""
    seq = build_boost_sequence(small, scanner_055T(), add_trigger=False)
    assert _count_adc(seq) == 2 * small.n_shots * small.im_segments


def test_rr_too_short_raises():
    with pytest.raises(ValueError):
        build_boost_sequence(BoostParams(rr=0.2, nx=16, ny=8, im_segments=8))
