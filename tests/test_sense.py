"""Tests for the synthetic array and the SENSE reconstruction (pyboost.coils/.sense).

The array is a test bench with invented geometry (see pyboost.coils): these tests
check that the reconstruction machinery is right, not anything about a real coil.
"""

import numpy as np
import pytest

from pyboost.coils import synthetic_array, noise_correlation
from pyboost.sampling import sampling_mask
from pyboost.sense import sense_reconstruct, estimate_maps, coil_combine_rss

SHAPE = (12, 24, 24)
VOX = (3e-3, 3e-3, 3e-3)


@pytest.fixture(scope="module")
def maps():
    return synthetic_array(SHAPE, VOX, n_azimuth=4, n_rings=2,
                           cylinder_radius=0.07, loop_radius=0.03,
                           ring_pitch=0.025)


def _object(phase=0.6):
    z, y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1], 0:SHAPE[2]]
    c = np.array([(SHAPE[0] - 1) / 2, (SHAPE[1] - 1) / 2, (SHAPE[2] - 1) / 2])
    d = np.sqrt(((z - c[0]) / 5.0) ** 2 + ((y - c[1]) / 9.0) ** 2
                + ((x - c[2]) / 9.0) ** 2)
    img = np.where(d < 1, 1.0, 0.0) + 0.5 * np.where((d < 0.45), 1.0, 0.0)
    return img.astype(complex) * np.exp(1j * phase)


def _ksp(img, maps):
    ax = (-3, -2, -1)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img[None] * maps, axes=ax),
                                       axes=ax), axes=ax)


def _nrmse(a, b, m):
    return float(np.linalg.norm((a - b)[m]) / np.linalg.norm(b[m]))


def test_array_shape_normalisation_and_diversity(maps):
    assert maps.shape == (8,) + SHAPE
    ctr = maps[:, SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2]
    assert np.sqrt((np.abs(ctr) ** 2).sum()) == pytest.approx(1.0)
    assert noise_correlation(maps) < 0.7                  # channels are not copies


def test_full_sampling_recovers_the_object_exactly(maps):
    """Same scaling as images_from_kspace: ifftn with 1/N, so the object comes back
    up to the encoding model (which here is exactly the one used to simulate)."""
    img = _object()
    ksp = _ksp(img, maps)
    rec = sense_reconstruct(ksp, np.ones(SHAPE[:2], dtype=bool), maps, lam=0.0,
                            max_iter=80)
    body = np.abs(img) > 0
    # forward model used fftn (no scaling); images_from_kspace's ifftn undoes it
    # exactly, so the unit-amplitude object is expected back.
    assert _nrmse(rec * img.size ** 0, img, body) < 0.02


def test_sense_beats_adjoint_at_r3_and_preserves_phase(maps):
    img = _object(phase=0.6)
    ksp = _ksp(img, maps)
    mask = sampling_mask(SHAPE[1], SHAPE[0], acceleration=3.0)
    ku = ksp * mask[None, :, :, None]
    sense = sense_reconstruct(ku, mask, maps, lam=0.0, max_iter=80)
    full = sense_reconstruct(ksp, np.ones(SHAPE[:2], bool), maps, lam=0.0, max_iter=80)
    body = np.abs(img) > 0
    ax = (-3, -2, -1)
    adj = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ku * 1.0, axes=ax), axes=ax),
                          axes=ax)
    adj = (np.conj(maps) * adj).sum(0)                   # zero-filled, model-combined
    assert _nrmse(sense, full, body) < 0.5 * _nrmse(adj / np.abs(adj).max()
                                                    * np.abs(full).max(), full, body)
    ang = np.angle(sense[np.abs(img) > 0.9] * np.exp(-1j * 0.6))
    assert np.abs(ang).mean() < 0.1


def test_rss_combination_discards_the_phase(maps):
    img = _object(phase=0.6)
    rss = coil_combine_rss(_ksp(img, maps))
    assert np.isrealobj(rss) and (rss >= 0).all()         # nothing left to separate with


def test_espirit_maps_match_the_truth_up_to_a_phase(maps):
    img = _object()
    ksp = _ksp(img, maps)
    mask = sampling_mask(SHAPE[1], SHAPE[0], acceleration=3.0, center_fraction=0.12)
    est = estimate_maps(ksp * mask[None, :, :, None], calib_width=8,
                        kernel_width=4)   # the default 6 does not fit 12 slices
    body = np.abs(img) > 0.9
    # compare the *images* the two maps produce, which removes the per-voxel phase
    a = np.abs(est)[:, body]
    b = np.abs(maps)[:, body]
    a = a / np.linalg.norm(a, axis=0)
    b = b / np.linalg.norm(b, axis=0)
    assert float((a * b).sum(0).mean()) > 0.9


def test_shape_checks(maps):
    ksp = _ksp(_object(), maps)
    with pytest.raises(ValueError):
        sense_reconstruct(ksp, np.ones((3, 3), bool), maps)
    with pytest.raises(ValueError):
        sense_reconstruct(ksp, np.ones(SHAPE[:2], bool), maps[:4])
    assert not sense_reconstruct(np.zeros_like(ksp), np.ones(SHAPE[:2], bool),
                                 maps).any()


# --- Classical SENSE unfolding of a regular undersampling ---------------------

from pyboost.sense import regular_mask, unfold_regular  # noqa: E402


@pytest.mark.parametrize("ry,rz", [(2, 1), (2, 2), (3, 2)])
def test_regular_unfolding_is_exact_without_noise(maps, ry, rz):
    img = _object(phase=0.6)
    ksp = _ksp(img, maps)
    mask = regular_mask(SHAPE[:2], ry, rz)
    rec, g = unfold_regular(ksp * mask[None, :, :, None], maps, ry, rz)
    body = np.abs(img) > 0
    assert _nrmse(rec, img, body) < 1e-6
    ang = np.angle(rec[np.abs(img) > 0.9] * np.exp(-1j * 0.6))
    assert np.abs(ang).max() < 1e-6                      # phase preserved exactly


def test_g_factor_is_at_least_one_and_grows_with_acceleration(maps):
    img = _object()
    ksp = _ksp(img, maps)
    gs = []
    for ry, rz in ((2, 1), (2, 2)):
        mask = regular_mask(SHAPE[:2], ry, rz)
        _, g = unfold_regular(ksp * mask[None, :, :, None], maps, ry, rz)
        body = np.abs(img) > 0
        assert g[body].min() >= 1 - 1e-6
        gs.append(float(g[body].mean()))
    assert gs[1] > gs[0]


def test_regular_mask_and_shape_checks(maps):
    assert regular_mask((12, 24), 2, 3).sum() == (12 // 3) * (24 // 2)
    with pytest.raises(ValueError):
        regular_mask((12, 24), 5, 1)
    with pytest.raises(ValueError):
        unfold_regular(_ksp(_object(), maps), maps[:4], 2, 1)
