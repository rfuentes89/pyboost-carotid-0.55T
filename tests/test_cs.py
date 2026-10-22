"""Tests for the single-coil L1-wavelet reconstruction (pyboost.cs)."""

import numpy as np
import pytest

from pyboost.cs import cs_reconstruct
from pyboost.recon import images_from_kspace
from pyboost.sampling import sampling_mask


def _object(n=24, phase=0.7):
    """Piecewise-constant 3D object with a constant complex phase."""
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    c = (n - 1) / 2
    ball = np.hypot(np.hypot(y - c, x - c), z - c) < n * 0.3
    small = np.hypot(np.hypot(y - c - 3, x - c + 2), z - c) < n * 0.12
    img = 0.4 * ball + 0.6 * small
    return img.astype(complex) * np.exp(1j * phase)


def _kspace(img):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))


def _nrmse(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_full_sampling_matches_the_plain_inverse_fft():
    """Scaling convention and centring agree with pyboost.recon at lam = 0."""
    img = _object()
    k = _kspace(img)
    mask = np.ones(k.shape[:2], dtype=bool)
    rec = cs_reconstruct(k, mask, lam=0.0, max_iter=30)
    ref = images_from_kspace(k, k)[0]
    assert np.allclose(rec, ref, atol=1e-4 * np.abs(ref).max())


def test_cs_beats_zero_filling_on_a_sparse_object():
    img = _object()
    k = _kspace(img)
    mask = sampling_mask(img.shape[1], img.shape[0], acceleration=3.0)
    ku = k * mask[:, :, None]
    zf = images_from_kspace(ku, ku)[0]
    cs = cs_reconstruct(ku, mask, lam=0.005, max_iter=100)
    ref = images_from_kspace(k, k)[0]
    assert _nrmse(cs, ref) < _nrmse(zf, ref)


def test_complex_phase_is_preserved():
    """The Dixon separation is in the phase; the reconstruction must keep it."""
    phase = 0.7
    img = _object(phase=phase)
    k = _kspace(img)
    mask = sampling_mask(img.shape[1], img.shape[0], acceleration=3.0)
    cs = cs_reconstruct(k * mask[:, :, None], mask, lam=0.005, max_iter=100)
    tissue = np.abs(img) > 0.3
    ang = np.angle(cs[tissue] * np.exp(-1j * phase))
    assert np.abs(ang).mean() < 0.05


def test_lambda_is_relative_to_the_signal_level():
    img = _object()
    k = _kspace(img)
    mask = sampling_mask(img.shape[1], img.shape[0], acceleration=3.0)
    a = cs_reconstruct(k * mask[:, :, None], mask, lam=0.005, max_iter=40)
    b = cs_reconstruct(1000.0 * k * mask[:, :, None], mask, lam=0.005, max_iter=40)
    assert np.allclose(b, 1000.0 * a, rtol=1e-3, atol=1e-3 * np.abs(b).max())


def test_empty_data_returns_zeros_and_bad_shapes_are_rejected():
    z = np.zeros((4, 4, 4), dtype=complex)
    assert not cs_reconstruct(z, np.ones((4, 4), bool)).any()
    with pytest.raises(ValueError):
        cs_reconstruct(z, np.ones((3, 4), bool))
    with pytest.raises(ValueError):
        cs_reconstruct(z[0], np.ones((4, 4), bool))
