"""Tests for the two-point Dixon separation (pyboost.dixon).

Validated on **synthetic** signals, where water, fat and the field map are set
independently and the answer is known exactly. This is deliberate: MRzero has no
multi-species model, so ``phantom.py`` represents fat as a B0 offset, which is
mathematically indistinguishable from off-resonance water. A simulated phantom
cannot decide whether a species separation is right; only a forward model that
keeps water and fat as separate species can.
"""

import numpy as np
import pytest

from pyboost import scanner_055T, fat_frequency
from pyboost.dixon import (fat_phasors, conditioning, separate_water_fat,
                           synthesize_in_opposed)

TE1, TE2 = 4.18e-3, 9.85e-3


@pytest.fixture
def df():
    return fat_frequency(scanner_055T(max_grad=23.0, max_slew=25.0,
                                      rf_ringdown_time=20e-6))


def forward(w, f, psi, dfreq, te1=TE1, te2=TE2, phi0=0.7):
    """Noise-free two-echo signal for known water, fat and field map."""
    c1, c2 = fat_phasors(te1, te2, dfreq)
    s1 = (w + f * c1) * np.exp(2j * np.pi * psi * te1 + 1j * phi0)
    s2 = (w + f * c2) * np.exp(2j * np.pi * psi * te2 + 1j * phi0)
    return s1, s2


# --- Conditioning ----------------------------------------------------------

def test_conditioning_matches_the_design_numbers(df):
    c = conditioning(TE1, TE2, df)
    assert c.phase_deg == pytest.approx(-162.5, abs=0.5)
    assert c.det == pytest.approx(1.977, abs=0.005)
    assert c.noise_amplification == pytest.approx(1.012, abs=0.002)
    assert c.alias_period_hz == pytest.approx(176.4, abs=0.5)


def test_opposed_phase_is_the_noise_optimum(df):
    """|det A| = 2 sin(dphi/2) peaks at 180 deg; REACT pays ~1.2% for 162.6."""
    best = conditioning(TE1, TE1 + 0.5 / abs(df), df)
    ours = conditioning(TE1, TE2, df)
    assert best.det == pytest.approx(2.0, abs=1e-6)
    assert ours.det < best.det
    assert ours.noise_amplification < 1.02


# --- Per-voxel algebra -----------------------------------------------------

@pytest.mark.parametrize("w,f,psi", [
    (1.0, 0.2, 0.0),      # water-dominant
    (0.3, 0.8, 10.0),     # fat-dominant
    (0.9, 0.0, -25.0),    # pure water  (|r| = 1: a == 0, root A is pure fat)
    (0.0, 1.0, 40.0),     # pure fat    (|r| = 1)
    (2.0, 0.5, 5.0),
])
def test_one_candidate_recovers_ground_truth_exactly(w, f, psi, df):
    s1, s2 = forward(np.array([w]), np.array([f]), np.array([psi]), df)
    cand = separate_water_fat(s1, s2, TE1, TE2, df)
    alias = 1.0 / (TE2 - TE1)
    hits = [k for k in (0, 1)
            if abs(cand.water[k, 0] - w) < 1e-9 and abs(cand.fat[k, 0] - f) < 1e-9]
    assert hits, f"neither candidate recovered W={w}, F={f}"
    k = hits[0]
    dpsi = (cand.psi[k, 0] - psi + alias / 2) % alias - alias / 2
    assert abs(dpsi) < 1e-6, "amplitudes right but field map wrong"


def test_the_two_roots_are_reciprocal(df):
    """t and 1/t: the water/fat pair falls out of the algebra, not an assumption."""
    w, f = np.array([1.0, 0.4, 2.0]), np.array([0.25, 0.9, 0.5])
    cand = separate_water_fat(*forward(w, f, np.zeros(3), df), TE1, TE2, df)
    t0 = cand.fat[0] / cand.water[0]
    t1 = cand.fat[1] / cand.water[1]
    assert np.allclose(t0 * t1, 1.0, atol=1e-8)


def test_both_candidates_fit_the_data_equally(df):
    """A single voxel cannot choose; that is why resolution must be spatial."""
    cand = separate_water_fat(*forward(np.array([1.0]), np.array([0.3]),
                                       np.array([12.0]), df), TE1, TE2, df)
    assert cand.residual[0, 0] < 1e-20
    assert cand.residual[1, 0] < 1e-20


def test_180_degrees_is_not_a_special_degenerate_angle(df):
    """Regression guard for a claim that was made and then refuted by measurement.

    Both candidates must fit exactly at 180 deg *and* at REACT's 162.6 deg; the
    angle does not decide whether a voxel is separable.
    """
    for dte in (0.5 / abs(df), (162.6 / 360.0) / abs(df)):
        te2 = TE1 + dte
        s1, s2 = forward(np.array([0.6]), np.array([0.5]), np.array([8.0]),
                         df, te2=te2)
        cand = separate_water_fat(s1, s2, TE1, te2, df)
        assert cand.residual[0, 0] < 1e-20 and cand.residual[1, 0] < 1e-20


def test_water_dominant_candidate_is_first(df):
    cand = separate_water_fat(*forward(np.array([0.2]), np.array([1.5]),
                                       np.array([0.0]), df), TE1, TE2, df)
    assert cand.water[0, 0] >= cand.fat[0, 0]


def test_background_voxels_do_not_produce_nans(df):
    cand = separate_water_fat(np.zeros(3, dtype=complex),
                              np.zeros(3, dtype=complex), TE1, TE2, df)
    assert np.all(np.isfinite(cand.water)) and np.all(np.isfinite(cand.psi))
    assert np.all(cand.magnitude == 0)


def test_mismatched_shapes_rejected(df):
    with pytest.raises(ValueError, match="shapes differ"):
        separate_water_fat(np.zeros(4), np.zeros(5), TE1, TE2, df)


# --- Synthesized in/opposed phase -----------------------------------------

def test_in_and_opposed_phase_synthesis():
    w, f = np.array([1.0, 0.2]), np.array([0.3, 0.9])
    in_phase, opposed = synthesize_in_opposed(w, f)
    assert np.allclose(in_phase, w + f)
    assert np.allclose(opposed, np.abs(w - f))
