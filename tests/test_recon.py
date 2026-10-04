"""Tests for the REACT k-space reordering (pyboost.recon), independent of Dixon."""

import numpy as np
import pytest

from pyboost import scanner_055T, ReactParams, build_react_sequence
from pyboost.recon import kspace_from_signal, images_from_kspace


@pytest.fixture
def system():
    return scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)


def test_reordering_places_a_point_object_correctly(system, tmp_path):
    """Guards echo demultiplexing and the centric order, on their own."""
    mr0 = pytest.importorskip("MRzeroCore")
    p = ReactParams(nx=32, ny=32, tfe_factor=32, dummy_shots=0, fov=0.2)
    path = str(tmp_path / "recon.seq")
    build_react_sequence(p, system).write(path)
    seq0 = mr0.Sequence.import_file(path)
    offset = (0.05, -0.025)
    obj = mr0.CustomVoxelPhantom(pos=[[offset[0], offset[1], 0.0]], PD=1.0,
                                 T1=1.0, T2=0.2, T2dash=0.03, D=0.0,
                                 voxel_size=0.004)
    signal, _ = mr0.util.simulate(seq0, obj)

    k1, k2 = kspace_from_signal(signal, p)
    assert np.all(np.abs(k1).sum(axis=-1) > 0), "echo 1 has unfilled lines"
    assert np.all(np.abs(k2).sum(axis=-1) > 0), "echo 2 has unfilled lines"

    i1, _ = images_from_kspace(k1, k2)
    im = np.abs(i1[0])
    iy, ix = np.unravel_index(np.argmax(im), im.shape)
    dx = p.fov / p.nx
    assert (ix - p.nx / 2) * dx == pytest.approx(offset[0], abs=dx)
    assert (iy - p.ny / 2) * dx == pytest.approx(offset[1], abs=dx)


def test_short_signal_is_rejected():
    p = ReactParams(nx=16, ny=16, tfe_factor=16, dummy_shots=0)
    with pytest.raises(ValueError, match="samples"):
        kspace_from_signal(np.zeros(10, dtype=complex), p)


def test_partial_acquisition_leaves_the_rest_zero():
    """Unacquired encodes must stay zero, not wrap around or alias."""
    p = ReactParams(nx=16, ny=16, tfe_factor=4, dummy_shots=0)
    n_tr = 4
    sig = np.ones(n_tr * 2 * p.nx, dtype=complex)
    k1, k2 = kspace_from_signal(sig, p, acquired_shots=1)
    assert np.count_nonzero(np.abs(k1).sum(axis=-1)) == n_tr
    assert np.count_nonzero(np.abs(k2).sum(axis=-1)) == n_tr


def test_reconstruction_preserves_the_inter_echo_phase(system, tmp_path):
    """Phase is the whole Dixon measurement, so the reconstruction must keep it.

    A water-like voxel must show ~0 deg between the echoes and a voxel at the fat
    offset the designed water-fat angle. Tested on single voxels because on a
    combined object the weak tissue is contaminated by its neighbours' truncation
    ringing -- a point-spread effect, not a reconstruction error.
    """
    mr0 = pytest.importorskip("MRzeroCore")
    from pyboost import fat_frequency, kernel_report
    p = ReactParams(nx=16, ny=16, tfe_factor=16, dummy_shots=1, fov=0.2)
    rep = kernel_report(system, p)
    df = fat_frequency(system)
    designed = abs(360.0 * df * rep["delta_te"])
    path = str(tmp_path / "phase.seq")
    build_react_sequence(p, system).write(path)
    seq0 = mr0.Sequence.import_file(path)

    def phase(b0):
        obj = mr0.CustomVoxelPhantom(pos=[[0.02, 0.01, 0.0]], PD=1.0, T1=1.0,
                                     T2=0.2, T2dash=0.03, B0=b0, D=0.0,
                                     voxel_size=0.004)
        sig, _ = mr0.util.simulate(seq0, obj)
        i1, i2 = (a[0] for a in images_from_kspace(*kspace_from_signal(sig, p)))
        iy, ix = np.unravel_index(np.argmax(np.abs(i1)), i1.shape)
        return float(np.rad2deg(np.angle(i2[iy, ix] * np.conj(i1[iy, ix]))))

    assert abs(phase(0.0)) < 1.0
    assert abs(phase(df)) == pytest.approx(designed, abs=1.0)
