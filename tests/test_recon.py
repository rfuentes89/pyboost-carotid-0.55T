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
