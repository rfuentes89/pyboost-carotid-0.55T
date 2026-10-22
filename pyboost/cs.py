"""Compressed-sensing reconstruction of undersampled REACT data (single coil).

Provenance -- read this before citing
-------------------------------------
Published REACT protocols reconstruct with compressed SENSE (Philips), whose
sparsity transform, regularisation and solver are not described in the papers
read. This module is a **plain L1-wavelet reconstruction** (Lustig, Donoho &
Pauly, Magn Reson Med 2007, the model sigpy implements)::

    min_x  1/2 || P F x - y ||^2 + lambda || W x ||_1

with ``P`` the sampling mask, ``F`` the Fourier transform and ``W`` a 3D
wavelet. The choices of wavelet (``db4``, sigpy's default) and ``lambda`` are
**design choices of this code**; ``lambda`` is relative, see below, and
``scripts/validate_cs_retrospective.py`` shows how it was picked and how much
the answer depends on it.

What it does not do: **no coil sensitivities.** This is a one-coil
reconstruction; SENSE with the 20-channel head/neck coil is not implemented or
validated here.

The two echoes are reconstructed independently and the result is **complex**:
the Dixon separation lives in the phase, so a magnitude-only reconstruction (or
any step that discards phase) would defeat it.
"""

from __future__ import annotations

import numpy as np

from .recon import images_from_kspace


def cs_reconstruct(kspace: np.ndarray, mask: np.ndarray, lam: float = 0.01,
                   wave_name: str = "db4", max_iter: int = 100) -> np.ndarray:
    """L1-wavelet reconstruction of one echo.

    ``kspace`` is ``(nz, ny, nx)`` with zeros where nothing was acquired and
    ``mask`` the ``(nz, ny)`` boolean of acquired ``(kz, ky)`` lines. Returns the
    complex image in the **same scaling as** :func:`pyboost.recon.images_from_kspace`
    (inverse FFT with ``1/N``), so the two are directly comparable.

    ``lam`` is relative: the data are scaled so the zero-filled image has unit
    peak before solving, so ``lam = 0.01`` means 1% of that peak per wavelet
    coefficient whatever the absolute signal level. ``lam = 0`` is plain least
    squares on the sampled data, i.e. the zero-filled image for a fully sampled
    mask.
    """
    import sigpy as sp
    from sigpy.mri import app

    if kspace.ndim != 3:
        raise ValueError("kspace must be (nz, ny, nx)")
    nz, ny, nx = kspace.shape
    if mask.shape != (nz, ny):
        raise ValueError(f"mask {mask.shape} does not match kspace {(nz, ny)}")

    n = kspace.size
    zero_filled = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace), norm="ortho"))
    scale = float(np.abs(zero_filled).max())
    if scale == 0.0:
        return np.zeros_like(kspace, dtype=complex)

    # sigpy's FFT is centred and orthonormal; ours is centred with 1/N.
    y = (kspace / scale)[None]
    w = np.broadcast_to(mask[:, :, None], kspace.shape).astype(float)[None]
    mps = np.ones((1,) + kspace.shape, dtype=complex)
    img = app.L1WaveletRecon(y, mps, lam, weights=w, wave_name=wave_name,
                             max_iter=max_iter, show_pbar=False).run()
    return np.asarray(img) * scale / np.sqrt(n)


def cs_images(k1: np.ndarray, k2: np.ndarray, mask: np.ndarray, **kw):
    """Both echoes through :func:`cs_reconstruct`: ``(echo1, echo2)`` images."""
    return cs_reconstruct(k1, mask, **kw), cs_reconstruct(k2, mask, **kw)


def zero_filled_images(k1: np.ndarray, k2: np.ndarray):
    """The unreconstructed baseline: inverse FFT of the zero-filled data."""
    return images_from_kspace(k1, k2)
