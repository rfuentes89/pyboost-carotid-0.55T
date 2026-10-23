"""Tests for the undersampling pattern and accelerated 3D REACT."""

import numpy as np
import pytest

from pyboost import (ReactParams, scanner_055T, build_react_sequence,
                     react_3d_208x208x88, encode_order, shot_encodes)
from pyboost.sampling import sampling_mask, acquired_encodes, normalised_radius


@pytest.fixture(scope="module")
def system():
    return scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)


@pytest.mark.parametrize("r", [2.0, 3.0, 5.0, 8.0])
def test_count_is_exactly_n_over_r(r):
    ny, nz = 64, 32
    assert sampling_mask(ny, nz, acceleration=r).sum() == round(ny * nz / r)


def test_r1_is_the_full_grid():
    assert sampling_mask(16, 8).all()
    assert sampling_mask(16, 8, acceleration=1.0).shape == (8, 16)


def test_centre_is_fully_sampled():
    ny, nz = 64, 32
    m = sampling_mask(ny, nz, acceleration=5.0, center_fraction=0.03)
    kz, ky = np.mgrid[0:nz, 0:ny]
    rho = normalised_radius(ky, kz, ny, nz)
    n_center = int(round(0.03 * ny * nz))
    inner = np.argsort(rho.ravel(), kind="stable")[:n_center]
    assert m.ravel()[inner].all()


def test_pattern_is_reproducible_and_seed_dependent():
    a = sampling_mask(64, 32, acceleration=5.0, seed=1)
    b = sampling_mask(64, 32, acceleration=5.0, seed=1)
    c = sampling_mask(64, 32, acceleration=5.0, seed=2)
    assert (a == b).all() and (a != c).any()


def test_density_falls_with_radius():
    """Variable density: the inner half of k-space is sampled more densely."""
    ny, nz = 128, 64
    m = sampling_mask(ny, nz, acceleration=5.0)
    kz, ky = np.mgrid[0:nz, 0:ny]
    rho = normalised_radius(ky, kz, ny, nz)
    inner, outer = rho < 0.5, rho >= 0.9
    assert m[inner].mean() > 3 * m[outer].mean()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        sampling_mask(8, 8, acceleration=0.5)
    with pytest.raises(ValueError):
        sampling_mask(8, 8, acceleration=2.0, center_fraction=1.5)
    with pytest.raises(ValueError):
        ReactParams(acceleration=0.9)


def test_n_encodes_and_shots_follow_the_acceleration():
    p = ReactParams(ny=64, nz=32, acceleration=4.0, tfe_factor=22)
    assert p.n_encodes == 512
    assert p.n_shots == -(-512 // 22)
    assert ReactParams(ny=64, nz=32).n_encodes == 2048


def test_acquired_set_is_partitioned_between_shots():
    p = ReactParams(ny=64, nz=32, acceleration=5.0, tfe_factor=22)
    seen = [e for s in range(p.n_shots) for e in shot_encodes(s, p)]
    assert len(seen) == p.n_encodes
    assert sorted(seen) == sorted(acquired_encodes(64, 32, acceleration=5.0))


def test_every_shot_starts_next_to_the_centre_in_3d():
    p = ReactParams(ny=64, nz=32, acceleration=5.0, tfe_factor=22)
    ordered = sorted(encode_order(p),
                     key=lambda e: float(normalised_radius(e[0], e[1], 64, 32)))
    nearest = set(ordered[:p.n_shots + 2])
    for s in range(p.n_shots):
        assert shot_encodes(s, p)[0] in nearest
        r = [float(normalised_radius(e[0], e[1], 64, 32)) for e in shot_encodes(s, p)]
        assert r == sorted(r)


def test_radius_is_normalised_so_anisotropic_grids_are_not_skewed():
    """With ny != nz an index distance would favour the longer axis."""
    p = ReactParams(ny=64, nz=8, centric=True)
    first_ten = encode_order(p)[:10]
    spread_ky = max(abs(e[0] - 32) for e in first_ten) / 32
    spread_kz = max(abs(e[1] - 4) for e in first_ten) / 4
    assert abs(spread_ky - spread_kz) < 0.35      # comparable in normalised terms


def test_accelerated_sequence_visits_exactly_the_pattern(system):
    p = ReactParams(nx=32, ny=16, nz=8, tfe_factor=16, dummy_shots=0,
                    acceleration=3.0, slice_thickness=30e-3,
                    readout_bandwidth=400.0)
    seq = build_react_sequence(p, system)
    assert seq.check_timing()[0]
    ka = seq.calculate_kspace()[0]
    n = p.nx
    dky, dkz = 1.0 / p.fov, 1.0 / p.slice_thickness
    visited = {(round(float(ka[1, 2 * tr * n]) / dky) + p.ny // 2,
                round(float(ka[2, 2 * tr * n]) / dkz) + p.nz // 2)
               for tr in range(p.n_encodes)}
    expected = set(acquired_encodes(p.ny, p.nz, acceleration=3.0,
                                    center_fraction=p.sampling_center_fraction,
                                    power=p.sampling_power, seed=p.sampling_seed))
    assert len(visited) == p.n_encodes
    assert visited == expected


def test_3d_protocol_preset_builds_and_fits_the_timing(system):
    p = react_3d_208x208x88(5.0)
    assert (p.nx, p.ny, p.nz) == (208, 208, 88)
    assert p.fov / p.nx == pytest.approx(0.96e-3, rel=0.01)
    assert p.slice_thickness / p.nz == pytest.approx(0.96e-3)
    assert p.n_encodes == round(208 * 88 / 5)
    seq = build_react_sequence(p, system)
    assert seq.check_timing()[0]


def test_default_bandwidth_does_not_fit_at_nx_208(system):
    """Why the preset uses 384.6 Hz/pixel (12.5 us dwell): 300 rounds to a 15 us dwell and is rejected, not silently mistimed."""
    p = react_3d_208x208x88(5.0, readout_bandwidth=300.0)
    with pytest.raises(ValueError, match="readout"):
        build_react_sequence(p, system)


def test_fov_definition_is_the_slab_not_slab_times_nz(system):
    """Regression: the z extent was slice_thickness * nz (7.4 m for 88 x 84.5 mm)."""
    p = ReactParams(nx=32, ny=16, nz=8, tfe_factor=16, dummy_shots=0,
                    slice_thickness=30e-3, readout_bandwidth=400.0)
    fov = build_react_sequence(p, system).definitions["FOV"]
    assert fov[2] == pytest.approx(30e-3)
    assert max(fov) < 1.0
