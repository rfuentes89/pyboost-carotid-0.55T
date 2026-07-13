"""Unit tests for the BOOST preparation modules (pyboost.prep)."""

import numpy as np
import pytest

from pyboost.system import scanner_055T, fat_frequency
from pyboost.prep import fat_sat, t2_prep, adiabatic_ir


def _flip_deg(rf):
    """Nominal flip angle [deg] from the RF waveform (signal in Hz)."""
    dt = np.diff(rf.t, prepend=0.0)
    return abs(np.sum(rf.signal * dt)) * 360.0


@pytest.fixture
def system():
    return scanner_055T()


def test_fat_sat_frequency_and_flip(system):
    blocks = fat_sat(system, flip_angle_deg=180.0)
    # spoiler, RF, spoiler
    assert len(blocks) == 3
    rf = blocks[1][0]
    # Centred on the fat resonance (~-80 Hz at 0.55T) and ~180 deg.
    assert rf.freq_offset == pytest.approx(fat_frequency(system), rel=1e-6)
    assert _flip_deg(rf) == pytest.approx(180.0, abs=1.0)
    # Opposed spoilers on z bracket the pulse.
    assert blocks[0][0].channel == "z" and blocks[2][0].channel == "z"
    assert np.sign(blocks[0][0].amplitude) != np.sign(blocks[2][0].amplitude)


def test_t2_prep_composite_timing(system):
    te, trf = 50e-3, 500e-6
    blocks = t2_prep(system, te=te, trf=trf)
    # 90x, delay, 180y, delay, -90x, spoiler
    assert len(blocks) == 6
    half = te / 2 - 1.5 * trf
    assert blocks[1][0].delay == pytest.approx(half)
    assert blocks[3][0].delay == pytest.approx(half)
    # 90 / 180 / 90 flip angles.
    assert _flip_deg(blocks[0][0]) == pytest.approx(90.0, abs=1.0)
    assert _flip_deg(blocks[2][0]) == pytest.approx(180.0, abs=1.0)
    assert _flip_deg(blocks[4][0]) == pytest.approx(90.0, abs=1.0)
    # 180 is phase-shifted (y) relative to the 90 (x).
    assert blocks[2][0].phase_offset == pytest.approx(np.pi / 2)


def test_t2_prep_rejects_too_short_te(system):
    with pytest.raises(ValueError):
        t2_prep(system, te=1e-3, trf=500e-6)


def test_adiabatic_ir_structure(system):
    blocks = adiabatic_ir(system, post_delay=40e-3)
    assert len(blocks) == 3  # inversion, spoiler, delay
    assert blocks[2][0].delay == pytest.approx(40e-3)
    inv = blocks[0][0]
    # Long, frequency-swept (adiabatic) pulse: ~10.24 ms with a sweeping RF
    # phase (the tanh frequency modulation shows up as a varying signal phase).
    assert inv.shape_dur == pytest.approx(10.24e-3, rel=1e-3)
    assert np.ptp(np.angle(inv.signal)) > 1.0  # radians of phase sweep


def test_adiabatic_ir_rejects_negative_delay(system):
    with pytest.raises(ValueError):
        adiabatic_ir(system, post_delay=-1e-3)
