"""Unit tests for the BOOST preparation modules (pyboost.prep)."""

import numpy as np
import pytest

from pyboost.system import scanner_055T, fat_frequency
from pyboost.prep import fat_sat, t2_prep, inversion


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


def test_inversion_block_default(system):
    blocks = inversion(system, post_delay=40e-3)  # kind="block" by default
    assert len(blocks) == 3  # inversion, spoiler, delay
    assert blocks[2][0].delay == pytest.approx(40e-3)
    assert _flip_deg(blocks[0][0]) == pytest.approx(180.0, abs=1.0)


def test_inversion_adiabatic_is_frequency_swept(system):
    blocks = inversion(system, post_delay=40e-3, kind="adiabatic")
    inv = blocks[0][0]
    # ~10.24 ms hyperbolic-secant with a sweeping RF phase (tanh FM).
    assert inv.shape_dur == pytest.approx(10.24e-3, rel=1e-3)
    assert np.ptp(np.angle(inv.signal)) > 1.0  # radians of phase sweep


def test_inversion_rejects_negative_delay(system):
    with pytest.raises(ValueError):
        inversion(system, post_delay=-1e-3)


def test_inversion_rejects_unknown_kind(system):
    with pytest.raises(ValueError):
        inversion(system, post_delay=40e-3, kind="spam")


# --- T2-prep with multiple refocusing pulses --------------------------------

import pypulseq as pp  # noqa: E402


def _propagate(blocks, off_hz=0.0, b1=1.0):
    """Ideal-pulse Bloch propagation of the blocks that t2_prep actually returns.

    No relaxation, so a perfect preparation leaves Mz = +1 and the shortfall
    isolates pulse imperfection. Reads each RF's flip, phase and duration from the
    pypulseq event, and each delay from its delay event; the trailing spoiler is
    skipped. This tests the artefact itself, not a re-implementation of it.
    """
    def rot(phase, angle, dur):
        w1 = b1 * angle / dur
        n = np.array([w1 * np.cos(phase), w1 * np.sin(phase), 2 * np.pi * off_hz])
        w = np.linalg.norm(n)
        k = n / w
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(w * dur) * K + (1 - np.cos(w * dur)) * (K @ K)

    def free(t):
        th = 2 * np.pi * off_hz * t
        return np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                         [0, 0, 1]])

    m = np.array([0.0, 0.0, 1.0])
    for blk in blocks[:-1]:                       # drop the spoiler
        ev = blk[0]
        if hasattr(ev, "delay") and not hasattr(ev, "signal"):
            m = free(ev.delay) @ m
        else:
            m = rot(ev.phase_offset, np.deg2rad(_flip_deg(ev)), ev.shape_dur) @ m
    return m[2]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_t2_prep_block_structure_for_n_refocus(system, n):
    blocks = t2_prep(system, te=50e-3, trf=500e-6, n_refocus=n)
    assert len(blocks) == 2 * n + 4
    flips = [_flip_deg(b[0]) for b in blocks if hasattr(b[0], "signal")]
    assert flips[0] == pytest.approx(90.0, abs=1.0)
    assert flips[-1] == pytest.approx(90.0, abs=1.0)
    assert all(f == pytest.approx(180.0, abs=1.0) for f in flips[1:-1])
    assert len(flips) == n + 2


def test_t2_prep_refocusing_follows_the_mlev_phase_pattern(system):
    blocks = t2_prep(system, te=50e-3, trf=500e-6, n_refocus=8)
    phases = [b[0].phase_offset for b in blocks
              if hasattr(b[0], "signal")][1:-1]
    signs = [1 if np.isclose(p, np.pi / 2) else -1 for p in phases]
    assert signs == [+1, +1, -1, -1, -1, +1, +1, -1]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_t2_prep_te_is_independent_of_n_refocus(system, n):
    """TE is centre-of-90 to centre-of-(-90); extra pulses must not move it.

    Computed from the events themselves (delays plus RF shape durations), not
    from block durations: those include the RF dead time and ringdown, which are
    not part of the echo time.
    """
    te, trf = 50e-3, 500e-6
    blocks = t2_prep(system, te=te, trf=trf, n_refocus=n)
    pulses = [b[0] for b in blocks if hasattr(b[0], "signal")]
    delays = [b[0].delay for b in blocks
              if hasattr(b[0], "delay") and not hasattr(b[0], "signal")]
    achieved = (0.5 * pulses[0].shape_dur + sum(p.shape_dur for p in pulses[1:-1])
                + sum(delays) + 0.5 * pulses[-1].shape_dur)
    assert achieved == pytest.approx(te, abs=n * system.block_duration_raster)


def test_default_t2_prep_is_unchanged_for_boost(system):
    """BOOST and MRA call t2_prep with defaults and must not notice this change."""
    assert len(t2_prep(system)) == 6


def test_ideal_pulses_restore_mz_for_every_n(system):
    for n in (1, 2, 4, 8, 16):
        assert _propagate(t2_prep(system, n_refocus=n)) == pytest.approx(1.0, abs=1e-6)


def test_mlev_cycling_is_robust_to_b1_error_where_a_single_180_is_not(system):
    """The point of the extra pulses (Bloch simulation of the returned blocks).

    At +-20% B1 a single refocusing leaves Mz near 0.83; four or eight cycled
    pulses recover it. The ideal-pulse MRzero run cannot show this, so it has to
    be pinned here.
    """
    one = _propagate(t2_prep(system, n_refocus=1), b1=0.8)
    four = _propagate(t2_prep(system, n_refocus=4), b1=0.8)
    eight = _propagate(t2_prep(system, n_refocus=8), b1=1.2)
    assert one < 0.9
    assert four > 0.99 and eight > 0.99


def test_phase_cycling_is_what_provides_the_robustness(system):
    """Control: 8 pulses all on +y are *worse* than the MLEV pattern at B1 -10%."""
    cycled = _propagate(t2_prep(system, n_refocus=8), b1=0.9)
    blocks = t2_prep(system, n_refocus=8)
    for b in blocks[1:-1]:                         # force every 180 onto +y
        if hasattr(b[0], "signal") and _flip_deg(b[0]) > 150:
            b[0].phase_offset = np.pi / 2
    uncycled = _propagate(blocks, b1=0.9)
    assert cycled > uncycled + 0.02


def test_off_resonance_is_better_tolerated_with_more_refocusing(system):
    worst = lambda n: min(_propagate(t2_prep(system, n_refocus=n), off_hz=f)
                          for f in np.linspace(-100, 100, 21))
    assert worst(4) > worst(1)


@pytest.mark.parametrize("n", [0, 3, 5, 32])
def test_t2_prep_rejects_invalid_refocus_count(system, n):
    with pytest.raises(ValueError, match="n_refocus"):
        t2_prep(system, n_refocus=n)


def test_t2_prep_rejects_te_too_short_for_the_pulse_train(system):
    with pytest.raises(ValueError):
        t2_prep(system, te=5e-3, n_refocus=16)


@pytest.mark.parametrize("n", [1, 4, 8, 16])
def test_t2_prep_timing_is_clean_on_the_raster(system, n):
    """n=8 gives a 2.375 ms delay, off the 10 us raster, unless snapped."""
    seq = pp.Sequence(system=system)
    for b in t2_prep(system, te=50e-3, n_refocus=n):
        seq.add_block(*b)
    ok, errors = seq.check_timing()
    assert ok, errors[:3]
