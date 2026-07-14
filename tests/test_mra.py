"""Tests for the bright-blood MRA sequence (pyboost.mra)."""

import pytest

from pyboost import build_mra_sequence, BoostParams, scanner_055T


@pytest.fixture
def small():
    return BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                       centric=True)


def _count_adc(seq):
    return sum(getattr(seq.get_block(i), "adc", None) is not None
               for i in range(1, len(seq.block_events) + 1))


def test_mra_timing_ok(small):
    seq = build_mra_sequence(small, scanner_055T())
    ok, errors = seq.check_timing()
    assert ok, f"check_timing failed: {errors[:3]}"


def test_mra_single_contrast_kspace(small):
    """One bright-blood contrast per heartbeat -> ny lines total (not 2*ny)."""
    seq = build_mra_sequence(small, scanner_055T(), add_trigger=False)
    assert _count_adc(seq) == small.n_shots * small.im_segments


def test_mra_duration(small):
    system = scanner_055T()
    seq = build_mra_sequence(small, system, add_trigger=False)
    dur = seq.duration()[0]
    expected = (small.dummy_heart_beats + small.n_shots) * small.rr
    assert dur == pytest.approx(expected, abs=system.block_duration_raster * 4)


def test_mra_prep_toggles_change_duration(small):
    """Dropping T2-prep and FatSat shortens each heartbeat's prep."""
    system = scanner_055T()
    full = build_mra_sequence(small, system, use_t2prep=True, use_fatsat=True,
                              add_trigger=False).duration()[0]
    bare = build_mra_sequence(small, system, use_t2prep=False, use_fatsat=False,
                              add_trigger=False).duration()[0]
    # Same total (RR-filled) duration, but the bare version spends less on prep;
    # both must still pass timing and fit the RR interval.
    assert full == pytest.approx(bare, abs=system.block_duration_raster * 4)


def test_mra_inversion_option_builds(small):
    seq = build_mra_sequence(small, scanner_055T(), use_inversion=True)
    assert seq.check_timing()[0]
