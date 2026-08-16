"""Tests for the REACT angiography sequence (pyboost.react, pyboost.readout_dixon).

The tests that matter here are the ones that pin the *derived* 0.55T numbers:
the Dixon echo spacing and the inversion time have no published value at this
field, so they are computed, and a regression in either would silently produce
a sequence that still passes check_timing but images nothing useful.
"""

import math

import pytest

from pyboost import (build_react_sequence, ReactParams, scanner_055T,
                     dixon_echo_spacing, kernel_report, null_time_after_t2prep,
                     encode_order, shot_encodes, fat_frequency)
from pyboost.phantom import TISSUE_PROPERTIES


@pytest.fixture
def system():
    """The user's MAGNETOM Free.Max limits (derated below the 26/45 nameplate)."""
    return scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)


@pytest.fixture
def small():
    return ReactParams(nx=64, ny=32, tfe_factor=16, dummy_shots=1)


def _count_adc(seq):
    return sum(getattr(seq.get_block(i), "adc", None) is not None
               for i in range(1, len(seq.block_events) + 1))


# --- Derived physics -------------------------------------------------------

def test_fat_water_shift_at_055T(system):
    """~-80 Hz. Everything else in REACT at this field follows from this."""
    assert fat_frequency(system) == pytest.approx(-79.6, abs=0.5)


def test_dixon_spacing_matches_published_phase(system):
    """The 0.55T spacing must reproduce the 1.5T protocol's water-fat phase.

    Isaak 2021 (JCMR) uses TE1 1.72 / TE2 3.80 ms at 1.5T -> 2.08 ms spacing.
    At 1.5T the shift is 217 Hz, so that is 162.6 deg of relative phase. Holding
    the phase (not the time) is what transfers the protocol to 0.55T.
    """
    dte = dixon_echo_spacing(system)
    df = abs(fat_frequency(system))
    assert 360.0 * df * dte == pytest.approx(162.6, abs=0.5)
    # Sanity on the magnitude: ~2.7x the 1.5T spacing, and below opposed phase.
    assert dte == pytest.approx(5.67e-3, rel=0.02)
    assert dte < dixon_echo_spacing(system, 180.0)


def test_null_time_reduces_to_stir_without_t2prep():
    """With no T2-prep the formula must collapse to the textbook T1*ln2."""
    t1 = TISSUE_PROPERTIES["fat"]["T1"]
    assert null_time_after_t2prep(t1, TISSUE_PROPERTIES["fat"]["T2"], 0.0) == \
        pytest.approx(t1 * math.log(2))


def test_t2prep_shortens_the_null_time():
    """A T2-prep attenuates Mz before inversion, so the null comes earlier."""
    fat = TISSUE_PROPERTIES["fat"]
    plain = null_time_after_t2prep(fat["T1"], fat["T2"], 0.0)
    after = null_time_after_t2prep(fat["T1"], fat["T2"], 50e-3)
    assert after < plain


def test_default_ti_nulls_fat_and_leaves_blood_inverted():
    """The REACT mechanism: background at zero while blood is still inverted."""
    p = ReactParams()
    ti = p.resolved_ti()
    assert ti == pytest.approx(84.2e-3, rel=0.02)

    def mz(tissue):
        t = TISSUE_PROPERTIES[tissue]
        e2 = math.exp(-p.t2prep_duration / t["T2"])
        return 1.0 - (1.0 + e2) * math.exp(-ti / t["T1"])

    assert mz("fat") == pytest.approx(0.0, abs=1e-9)
    # Blood is strongly inverted -> large magnitude signal against a null
    # background. That contrast, not inflow, is what makes REACT work.
    assert mz("blood") < -0.5
    assert abs(mz("blood")) > 3 * abs(mz("muscle"))


# --- Kernel timing ---------------------------------------------------------

def test_kernel_hits_the_target_spacing(system, small):
    rep = kernel_report(system, small)
    assert rep["delta_te"] == pytest.approx(dixon_echo_spacing(system),
                                            abs=system.grad_raster_time)
    assert rep["te2"] == pytest.approx(rep["te1"] + rep["delta_te"])
    assert rep["te1"] > 0


def test_bandwidth_too_low_is_rejected_not_mistimed(system):
    """Below the floor the echoes cannot fit; that must raise, not silently slip."""
    p = ReactParams(readout_bandwidth=150.0)
    with pytest.raises(ValueError, match="Dixon spacing"):
        kernel_report(system, p)


def test_chemical_shift_displacement_is_subpixel(system, small):
    assert kernel_report(system, small)["chemical_shift_px"] < 0.5


# --- Sequence assembly -----------------------------------------------------

def test_react_timing_ok(system, small):
    seq = build_react_sequence(small, system)
    ok, errors = seq.check_timing()
    assert ok, f"check_timing failed: {errors[:3]}"


def test_two_adcs_per_tr(system, small):
    """The whole point of the Dixon readout: two echoes per excitation."""
    seq = build_react_sequence(small, system)
    assert _count_adc(seq) == small.n_shots * small.tfe_factor * 2


def test_both_echoes_sample_the_same_ky(system):
    """If the echoes sat on different ky lines, water/fat separation is invalid."""
    p = ReactParams(nx=16, ny=8, tfe_factor=8, dummy_shots=0, centric=False)
    seq = build_react_sequence(p, system)
    ka = seq.calculate_kspace()[0]
    n = p.nx
    for tr in range(p.ny):
        ky1 = ka[1, (2 * tr) * n]
        ky2 = ka[1, (2 * tr + 1) * n]
        assert ky1 == pytest.approx(ky2), f"TR {tr}: echoes on different ky"


def test_kspace_covers_the_requested_matrix(system):
    p = ReactParams(nx=16, ny=8, tfe_factor=8, dummy_shots=0, centric=False)
    seq = build_react_sequence(p, system)
    ka = seq.calculate_kspace()[0]
    delta_ky = 1.0 / p.fov
    ky = sorted({round(float(v), 6) for v in ka[1, ::p.nx]})
    assert len(ky) == p.ny
    assert ky[1] - ky[0] == pytest.approx(delta_ky)


def test_no_ecg_trigger(system, small):
    """REACT is untriggered -- that is the 'and Triggering' in the name."""
    seq = build_react_sequence(small, system)
    has_trigger = any(getattr(seq.get_block(i), "trig", None) is not None
                      for i in range(1, len(seq.block_events) + 1))
    assert not has_trigger


def test_definitions_carry_the_derived_values(system, small):
    """The .seq must be self-documenting: derived 0.55T values travel with it."""
    seq = build_react_sequence(small, system)
    d = seq.definitions
    assert d["Name"] == "REACT_carotid_055T"
    assert d["ReactT2prep"] == pytest.approx(50e-3)
    assert d["ReactTI"] == pytest.approx(84.2e-3, rel=0.05)
    assert d["ReactDeltaTE"] == pytest.approx(5.67e-3, rel=0.02)


def test_3d_slab_builds(system):
    p = ReactParams(nx=32, ny=16, nz=8, tfe_factor=16, dummy_shots=0,
                    slice_thickness=40e-3)
    seq = build_react_sequence(p, system)
    assert seq.check_timing()[0]
    assert p.n_encodes == 128


def test_3d_partition_encoding_is_correct(system):
    """kz must step by 1/slab and every (ky, kz) pair must be visited once.

    Regression guard: the partition encode rides on the same gradient as the
    slice rephaser, so the prephaser block has to fit their *combined* area.
    """
    p = ReactParams(nx=16, ny=4, nz=4, tfe_factor=16, dummy_shots=0,
                    centric=False, slice_thickness=40e-3)
    seq = build_react_sequence(p, system)
    ka = seq.calculate_kspace()[0]
    n = p.nx
    visited = [(round(float(ka[1, 2 * tr * n]), 3),
                round(float(ka[2, 2 * tr * n]), 3))
               for tr in range(p.ny * p.nz)]
    assert len(set(visited)) == p.n_encodes
    kz = sorted({z for _, z in visited})
    assert kz[1] - kz[0] == pytest.approx(1.0 / p.slice_thickness)


def test_3d_survives_a_tight_prephaser(system):
    """A short readout shrinks the prephaser block; the z gradient must still fit."""
    p = ReactParams(nx=32, ny=8, nz=16, tfe_factor=8, dummy_shots=0,
                    slice_thickness=30e-3, readout_bandwidth=400.0)
    assert build_react_sequence(p, system).check_timing()[0]


def test_centric_order_starts_at_kspace_centre():
    p = ReactParams(ny=32, nz=1, centric=True)
    first = encode_order(p)[0]
    assert abs(first[0] - p.ny / 2) <= 1


def test_shots_partition_all_encodes():
    p = ReactParams(ny=32, nz=2, tfe_factor=8)
    seen = [e for s in range(p.n_shots) for e in shot_encodes(s, p)]
    assert sorted(seen) == sorted(encode_order(p))


def test_shot_interval_too_short_is_rejected(system):
    p = ReactParams(nx=64, ny=32, tfe_factor=32, dummy_shots=0,
                    shot_interval=0.2)
    with pytest.raises(ValueError, match="shot_interval"):
        build_react_sequence(p, system)


def test_adiabatic_inversion_builds(system, small):
    """The scanner variant must assemble even though MRzero cannot model it."""
    p = ReactParams(nx=small.nx, ny=small.ny, tfe_factor=small.tfe_factor,
                    dummy_shots=1, inversion_kind="adiabatic")
    seq = build_react_sequence(p, system)
    assert seq.check_timing()[0]


def test_fatsat_fallback_builds(system, small):
    seq = build_react_sequence(small, system, use_fatsat=True)
    assert seq.check_timing()[0]


# --- Differentiable TI (pyboost.diffopt) -----------------------------------

@pytest.fixture
def tiny():
    """Smallest sequence that still has a full prep + shot, for MRzero."""
    return ReactParams(nx=16, ny=8, tfe_factor=8, dummy_shots=0)


def _voxel(name):
    import MRzeroCore as mr0
    t = TISSUE_PROPERTIES[name]
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=t["PD"], T1=t["T1"], T2=t["T2"],
        T2dash=0.03, D=0.0, voxel_size=0.005,
    )


def test_inversion_rep_is_found_and_is_not_the_t2prep_refocus(system, tiny):
    """A REACT shot has two 180 deg pulses; only one carries the TI."""
    from pyboost.diffopt import import_react_for_optimization
    seq0, inv_idx, info = import_react_for_optimization(tiny, system)
    assert len(inv_idx) == 1, "expected exactly one inversion per shot"
    # The one that carries TI is the long repetition, not the T2-prep refocus.
    durations = [float(rep.event_time.sum()) for rep in seq0]
    assert durations[inv_idx[0]] == pytest.approx(tiny.resolved_ti(), abs=5e-3)


def test_set_react_ti_is_exact(system, tiny):
    """The delay absorbs the whole change, so the requested TI is what plays."""
    import torch
    from pyboost.diffopt import import_react_for_optimization, set_react_ti
    seq0, inv_idx, info = import_react_for_optimization(tiny, system)
    for target in (0.06, 0.12):
        set_react_ti(seq0, inv_idx, info, torch.tensor(target))
        assert float(seq0[inv_idx[0]].event_time.sum()) == \
            pytest.approx(target, abs=1e-6)


def test_ti_is_differentiable(system, tiny):
    """Gradient must flow from the signal back to TI, or optimization is fake."""
    import torch
    from pyboost.diffopt import (import_react_for_optimization, set_react_ti,
                                 react_dc_signal)
    seq0, inv_idx, info = import_react_for_optimization(tiny, system)
    ti = torch.tensor(float(tiny.resolved_ti()), requires_grad=True)
    set_react_ti(seq0, inv_idx, info, ti)
    react_dc_signal(seq0, _voxel("blood"), tiny.nx).abs().backward()
    assert ti.grad is not None
    assert math.isfinite(float(ti.grad)) and abs(float(ti.grad)) > 0


def test_derived_ti_suppresses_fat_better_than_a_far_off_ti(system, tiny):
    """Sanity on the derivation: 84 ms must beat 180 ms at nulling fat.

    At TI ~180 ms fat has recovered past its null and outshines blood, which the
    TI sweep shows flipping the contrast negative. If the derived TI did not
    clearly win here, the null-time formula would be wrong.
    """
    import torch
    from pyboost.diffopt import (import_react_for_optimization, set_react_ti,
                                 react_dc_signal)
    seq0, inv_idx, info = import_react_for_optimization(tiny, system)
    fat = _voxel("fat")

    def fat_signal(ti_s):
        set_react_ti(seq0, inv_idx, info, torch.tensor(ti_s))
        return float(react_dc_signal(seq0, fat, tiny.nx).abs())

    assert fat_signal(tiny.resolved_ti()) < fat_signal(0.180)
