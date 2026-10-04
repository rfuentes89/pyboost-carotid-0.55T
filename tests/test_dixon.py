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


# --- Spatial resolution of the two candidates -----------------------------

from pyboost.dixon import resolve_field_map, select, separate  # noqa: E402


def _object(df, n=48, psi_fn=None, noise=0.0, seed=0):
    """Water disc + fat ring on a smooth field map, with optional noise."""
    y, x = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2
    r = np.hypot(y - cy, x - cx)
    water = np.where(r < n * 0.28, 1.0, 0.0)
    fat = np.where((r >= n * 0.32) & (r < n * 0.44), 1.0, 0.0)
    psi = (psi_fn(x, y, n) if psi_fn else
           30.0 * ((x - cx) / n + 0.5 * (y - cy) / n))
    s1, s2 = forward(water, fat, psi, df)
    if noise:
        rng = np.random.default_rng(seed)
        for s in (s1, s2):
            s += noise * (rng.standard_normal(s.shape)
                          + 1j * rng.standard_normal(s.shape))
    return water, fat, psi, s1, s2


def _swap_fraction(w, f, water, fat):
    fm, wm = fat > 0, water > 0
    wrong = np.sum(w[fm] > f[fm]) + np.sum(f[wm] > w[wm])
    return wrong / (fm.sum() + wm.sum())


def test_separation_recovers_a_synthetic_object(df):
    water, fat, psi, s1, s2 = _object(df)
    w, f, _ = separate(s1, s2, TE1, TE2, df)
    tissue = (water + fat) > 0
    assert np.allclose(w[tissue], water[tissue], atol=1e-6)
    assert np.allclose(f[tissue], fat[tissue], atol=1e-6)


def test_field_map_is_recovered_up_to_its_alias(df):
    water, fat, psi, s1, s2 = _object(df)
    _, _, est = separate(s1, s2, TE1, TE2, df)
    alias = 1.0 / (TE2 - TE1)
    tissue = (water + fat) > 0
    err = (est[tissue] - psi[tissue] + alias / 2) % alias - alias / 2
    assert np.abs(err).max() < 1e-6


def test_no_swaps_when_the_field_map_wraps(df):
    """psi is only defined mod 176 Hz; a localised excursion past +/-88 Hz must
    resolve, which needs the jump between neighbours measured modulo the alias."""
    def bump(x, y, n):
        # Near zero almost everywhere (what the prior assumes), with one smooth
        # excursion to ~130 Hz that crosses the alias boundary.
        return 4.0 + 130.0 * np.exp(-((x - 6) ** 2 + (y - n / 2) ** 2)
                                    / (2 * 5.0 ** 2))
    water, fat, psi, s1, s2 = _object(df, psi_fn=bump)
    assert psi[(water + fat) > 0].max() > 88.0
    w, f, _ = separate(s1, s2, TE1, TE2, df)
    assert _swap_fraction(w, f, water, fat) == 0


def test_result_does_not_depend_on_which_component_is_brightest(df):
    """Regression: a single lucky seed used to make a broken resolver look fine.

    With uniform amplitude the first voxel was the seed. Flipping the object
    moves the seed to the other component; both orientations must resolve.
    """
    water, fat, psi, s1, s2 = _object(df)
    for flip in (lambda a: a, lambda a: a[::-1, ::-1]):
        w, f, _ = separate(flip(s1), flip(s2), TE1, TE2, df)
        assert _swap_fraction(w, f, flip(water), flip(fat)) == 0


def test_background_does_not_disturb_the_resolution(df):
    """Voxels with no signal must not vote (they look like 'pure water')."""
    water, fat, psi, s1, s2 = _object(df)
    assert (np.abs(s1) == 0).any()
    cand = separate_water_fat(s1, s2, TE1, TE2, df)
    pick = resolve_field_map(cand)
    assert pick[np.abs(s1) == 0].sum() == 0      # background defaults to candidate 0


@pytest.mark.parametrize("noise", [0.01, 0.03, 0.06, 0.1])
def test_swaps_stay_rare_under_noise_across_seeds(df, noise):
    """The clinical failure mode is swaps, not blur (Pennig 2020: 10/35 at 3T).

    Many noise draws, not one: a fixed seed let a real failure pass. At sigma
    0.03 roughly 1 draw in 12 used to swap a whole region, because background
    noise passed the relative threshold and decided the orientation.
    """
    worst = 0.0
    for seed in range(12):
        water, fat, psi, s1, s2 = _object(df, noise=noise, seed=seed)
        w, f, _ = separate(s1, s2, TE1, TE2, df)
        worst = max(worst, _swap_fraction(w, f, water, fat))
    assert worst < 0.05, f"worst of 12 draws: {worst:.1%} swapped"


@pytest.mark.parametrize("noise", [0.01, 0.03, 0.1])
def test_noise_estimate_tracks_the_true_sigma(df, noise):
    from pyboost.dixon import estimate_noise
    ests = [estimate_noise(np.abs(_object(df, noise=noise, seed=k)[3]))
            for k in range(6)]
    assert np.mean(ests) == pytest.approx(noise, rel=0.1)


def test_noise_estimate_is_independent_of_the_background_fraction(df):
    """The first guess is biased by how much of the image is background; the
    iteration must remove that, not inherit it."""
    from pyboost.dixon import estimate_noise
    rng = np.random.default_rng(1)
    sigma = 0.03
    for tissue_fraction in (0.1, 0.4, 0.7):
        n = 64
        img = np.zeros((n, n))
        img.flat[: int(tissue_fraction * n * n)] = 1.0
        noisy = np.abs(img + sigma * (rng.standard_normal(img.shape)
                                      + 1j * rng.standard_normal(img.shape)))
        assert estimate_noise(noisy) == pytest.approx(sigma, rel=0.15), (
            f"tissue fraction {tissue_fraction}")


def test_noise_estimate_of_a_noise_free_image_is_zero():
    from pyboost.dixon import estimate_noise
    assert estimate_noise(np.zeros((8, 8))) == 0.0


def test_global_swap_is_a_documented_limit_of_the_prior(df):
    """Water sitting at psi = -(fat shift) is *identical* to fat at psi = 0.

    No smoothness argument can separate them, so the |psi|-near-zero prior calls
    it fat. That is not a bug to fix: it is the boundary of what the method can
    know, and it means an unshimmed offset near the fat-water shift will swap
    water and fat as a whole on real data.
    """
    n = 24
    water = np.ones((n, n))
    s1, s2 = forward(water, 0 * water, np.full((n, n), df), df)  # psi = df
    # The image is all tissue, so there is no background to estimate noise from;
    # the data are noise-free, so say so explicitly.
    w, f, _ = separate(s1, s2, TE1, TE2, df, noise_sigma=0.0)
    assert f.mean() > w.mean()     # called fat: the prior prefers psi near zero


# --- Water and fat of opposite sign ------------------------------------------
# REACT's inversion leaves water negative while fat, past its null, is positive.
# W and F are real but of either sign; only the relative sign is observable (the
# global sign is absorbed in the unknown phase phi0).

from pyboost.dixon import select_opposed  # noqa: E402

SIGNED = [(-1.0, 0.5), (-0.3, 1.0), (0.5, -1.0), (-1.0, 0.1), (-0.2, 1.5),
          (1.0, 0.5), (-1.0, -0.5)]


@pytest.mark.parametrize("w,f", SIGNED)
def test_opposite_sign_pairs_are_recovered_exactly(df, w, f):
    """Regression: the solver took abs() of its roots and returned, for
    W=-1, F=0.5, a water of 1.52 and a field-map error of -45 Hz."""
    psi = 5.0
    cand = separate_water_fat(*[np.array([x]) for x in forward(np.array([w]),
                                                               np.array([f]),
                                                               np.array([psi]), df)],
                              TE1, TE2, df, signed=True)
    assert np.allclose(cand.residual, 0.0, atol=1e-12)        # both candidates fit
    alias = cand.alias_hz
    hits = 0
    for k in (0, 1):
        ok_amp = (np.isclose(cand.water[k, 0], abs(w), atol=1e-9)
                  and np.isclose(cand.fat[k, 0], abs(f), atol=1e-9))
        err = (cand.psi[k, 0] - psi + alias / 2) % alias - alias / 2
        if ok_amp and abs(err) < 1e-6:
            hits += 1
            assert bool(cand.opposed[k, 0]) == (w * f < 0)
    assert hits == 1, "exactly one candidate is the truth"


def test_both_candidates_agree_on_the_relative_sign(df):
    """The roots t and 1/t have the same sign, so the data fixes it."""
    s = forward(np.array([-0.3]), np.array([1.0]), np.array([5.0]), df)
    cand = separate_water_fat(s[0], s[1], TE1, TE2, df, signed=True)
    assert cand.opposed[0, 0] == cand.opposed[1, 0] == True  # noqa: E712


def test_pure_species_is_not_flagged_as_opposed(df):
    for w, f in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
        s = forward(np.array([w]), np.array([f]), np.array([5.0]), df)
        cand = separate_water_fat(s[0], s[1], TE1, TE2, df, signed=True)
        assert not cand.opposed.any()


def test_w_equals_minus_f_is_degenerate_without_nans(df):
    """t = -1: the two roots coincide, like W == F."""
    s = forward(np.array([-1.0]), np.array([1.0]), np.array([5.0]), df)
    cand = separate_water_fat(s[0], s[1], TE1, TE2, df, signed=True)
    assert np.isfinite(cand.water).all() and np.isfinite(cand.psi).all()
    assert np.allclose(cand.water[0], cand.water[1], atol=1e-6)


def _signed_object(df, noise=0.0, seed=0, n=48, mixed_wf=(-0.9, 0.1)):
    """Negative water disc, positive fat ring, and a mixed ring between them."""
    y, x = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2
    r = np.hypot(y - cy, x - cx)
    water = np.where(r < n * 0.28, -1.0, 0.0)
    fat = np.where((r >= n * 0.32) & (r < n * 0.44), 1.0, 0.0)
    mixed = (r >= n * 0.28) & (r < n * 0.32)
    water = np.where(mixed, mixed_wf[0], water)    # partial volume, opposite signs
    fat = np.where(mixed, mixed_wf[1], fat)
    psi = 30.0 * ((x - cx) / n + 0.5 * (y - cy) / n)
    s1, s2 = forward(water, fat, psi, df)
    if noise:
        rng = np.random.default_rng(seed)
        for s in (s1, s2):
            s += noise * (rng.standard_normal(s.shape)
                          + 1j * rng.standard_normal(s.shape))
    return water, fat, psi, s1, s2


def test_spatial_resolution_with_opposite_signs(df):
    water, fat, psi, s1, s2 = _signed_object(df)
    cand = separate_water_fat(s1, s2, TE1, TE2, df, signed=True)
    pick = resolve_field_map(cand)
    w, f, est = select(cand, pick)
    opp = select_opposed(cand, pick)
    tissue = (np.abs(water) + np.abs(fat)) > 0
    assert np.allclose(w[tissue], np.abs(water[tissue]), atol=1e-6)
    assert np.allclose(f[tissue], np.abs(fat[tissue]), atol=1e-6)
    truth_opp = (water * fat < 0)
    assert (opp[tissue] == truth_opp[tissue]).all()
    alias = 1.0 / (TE2 - TE1)
    err = (est[tissue] - psi[tissue] + alias / 2) % alias - alias / 2
    assert np.abs(err).max() < 1e-6


@pytest.mark.parametrize("noise", [0.01, 0.03])
def test_opposite_sign_swaps_stay_rare_under_noise(df, noise):
    worst = 0.0
    for seed in range(8):
        # No mixed ring here: partial-volume bridging is a separate limit, see below.
        water, fat, psi, s1, s2 = _signed_object(df, noise=noise, seed=seed,
                                                 mixed_wf=(0.0, 0.0))
        w, f, _ = separate(s1, s2, TE1, TE2, df, signed=True)
        worst = max(worst, _swap_fraction(w, f, np.abs(water), np.abs(fat)))
    assert worst < 0.05, f"worst of 8 draws: {worst:.1%} swapped"


def test_synthesized_images_swap_roles_when_signs_oppose():
    w, f = np.array([1.0, 1.0]), np.array([0.3, 0.3])
    opp = np.array([False, True])
    in_phase, opposed = synthesize_in_opposed(w, f, opp)
    assert np.allclose(in_phase, [1.3, 0.7])       # species add / cancel
    assert np.allclose(opposed, [0.7, 1.3])
    assert np.allclose(synthesize_in_opposed(w, f)[0], [1.3, 1.3])   # old behaviour


@pytest.mark.xfail(strict=True, reason=(
    "Known limit of the region growing, independent of sign: a partial-volume "
    "ring between water and fat can bridge the two regions through voxels whose "
    "wrong candidate has a smooth field map, and a whole region comes out swapped. "
    "Measured on this object: mixture (0.5, 0.5) swaps 664 of 1396 voxels, "
    "(-0.7, 0.3) swaps 560; with noise 0.01 even (0.9, 0.1) and (-0.9, 0.1) swap "
    "about 11% in 8 of 8 draws, same-sign and opposite-sign alike. The same "
    "objects without the ring resolve exactly."))
@pytest.mark.parametrize("mixed,noise", [((0.5, 0.5), 0.0), ((-0.7, 0.3), 0.0),
                                         ((0.9, 0.1), 0.01), ((-0.9, 0.1), 0.01)])
def test_partial_volume_bridge_does_not_swap_a_region(df, mixed, noise):
    water, fat, psi, s1, s2 = _signed_object(df, mixed_wf=mixed, noise=noise)
    if mixed[0] > 0:                       # same-sign variant of the same object
        water = np.abs(water)
        s1, s2 = forward(water, fat, psi, df)
        if noise:
            rng = np.random.default_rng(0)
            for x in (s1, s2):
                x += noise * (rng.standard_normal(x.shape)
                              + 1j * rng.standard_normal(x.shape))
    w, f, _ = separate(s1, s2, TE1, TE2, df, signed=True)
    assert _swap_fraction(w, f, np.abs(water), np.abs(fat)) == 0


def test_default_keeps_the_same_sign_assumption(df):
    """signed=False is the previous behaviour: exact for same-sign pairs, and a
    wrong answer with a large residual for opposite-sign ones. That is why the
    opposite-sign path is an explicit option (see the docstring and O12)."""
    same = forward(np.array([1.0]), np.array([0.5]), np.array([5.0]), df)
    opp = forward(np.array([-1.0]), np.array([0.5]), np.array([5.0]), df)
    c_same = separate_water_fat(same[0], same[1], TE1, TE2, df)
    c_opp = separate_water_fat(opp[0], opp[1], TE1, TE2, df)
    assert np.allclose(c_same.residual, 0.0, atol=1e-12)
    assert c_opp.residual.min() > 0.1
    assert not c_opp.opposed.any()


def test_inter_echo_decay_makes_signed_less_forgiving_than_abs(df):
    """Documents why signed is not the default. A pure water voxel whose echo 2
    has decayed (T2*, not in the model) is explained by a small opposite-sign fat;
    abs() forces a same-sign one. Amplitudes are similar, the field-map candidates
    are not."""
    s1, s2 = forward(np.array([1.0]), np.array([0.0]), np.array([5.0]), df)
    s2 = s2 * 0.9
    a = separate_water_fat(s1, s2, TE1, TE2, df)
    b = separate_water_fat(s1, s2, TE1, TE2, df, signed=True)
    assert abs(a.fat[0, 0] - b.fat[0, 0]) < 0.05           # similar contamination
    assert bool(b.opposed[0, 0]) and not bool(a.opposed[0, 0])
    assert abs(a.psi[0, 0] - b.psi[0, 0]) > 5.0            # different field map
