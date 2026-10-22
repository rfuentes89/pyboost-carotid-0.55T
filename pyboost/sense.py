"""SENSE / SENSE+CS reconstruction of undersampled multi-channel REACT data.

Provenance -- read this before citing
-------------------------------------
Published REACT protocols use SENSE (Isaak 2021: 3.2) or Philips compressed SENSE
(3-10x). This module is a plain implementation of the standard models, via sigpy:

* SENSE (Pruessmann et al., Magn Reson Med 1999) as a least-squares problem
  ``min_x ||P F S x - y||^2 + lam ||x||^2``, solved by conjugate gradients, so it also
  works for the random (ky, kz) pattern of :mod:`pyboost.sampling`;
* SENSE + L1-wavelet (``lam`` as in :mod:`pyboost.cs`);
* sensitivity estimation by ESPIRiT (Uecker et al., Magn Reson Med 2014) from the
  fully sampled centre of the same undersampled data.

All regularisation values are design choices of this code. Phase is preserved: the
channels are combined by the model itself (``S^H``), never by root-sum-of-squares,
which would discard the phase the Dixon separation is built on.
"""

from __future__ import annotations

import numpy as np


def _normalise(ksp: np.ndarray):
    """Scale so the zero-filled root-sum-of-squares image has unit peak."""
    axes = (-3, -2, -1)
    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp, axes=axes),
                                       axes=axes, norm="ortho"), axes=axes)
    scale = float(np.sqrt((np.abs(img) ** 2).sum(0)).max())
    return scale


def estimate_maps(ksp: np.ndarray, calib_width: int = 8, thresh: float = 0.02,
                  kernel_width: int = 6, crop: float = 0.95) -> np.ndarray:
    """ESPIRiT sensitivity maps ``(ncoil, nz, ny, nx)`` from ``ksp`` (same shape).

    ``ksp`` is the undersampled multi-channel data with zeros where nothing was
    acquired; only its fully sampled central block is used (``calib_width`` lines
    per axis). The maps carry a global phase ambiguity per voxel absorbed in the
    image, as always with ESPIRiT.
    """
    from sigpy.mri import app
    return np.asarray(app.EspiritCalib(ksp, calib_width=calib_width, thresh=thresh,
                                       kernel_width=kernel_width, crop=crop,
                                       show_pbar=False).run())


def sense_reconstruct(ksp: np.ndarray, mask: np.ndarray, maps: np.ndarray,
                      lam: float = 0.0, l1: float = 0.0, wave_name: str = "db4",
                      max_iter: int = 50) -> np.ndarray:
    """SENSE (``l1 = 0``) or SENSE + L1-wavelet (``l1 > 0``) of one echo.

    ``ksp`` is ``(ncoil, nz, ny, nx)`` with zeros where nothing was acquired, ``mask``
    the ``(nz, ny)`` acquired lines, ``maps`` the sensitivities. Returns the complex
    image ``(nz, ny, nx)`` in the same scaling as
    :func:`pyboost.recon.images_from_kspace`. ``lam`` and ``l1`` are relative to the
    unit-peak normalisation of the data.

    **Convergence caveat, measured.** Both solvers are sigpy's, unpreconditioned. With the
    variable-density pattern the normal equations are badly conditioned and plain
    conjugate gradients converges very slowly: on the 20-channel synthetic bench at
    R = 5 the noise-free image error against the truth was 0.51 at 60 iterations, 0.31 at
    150, 0.21 at 300, 0.16 at 600 and 0.14 at 1200 (0.04 at R = 1 in 40). A
    sampling-density preconditioner tried here made it worse and was removed. So the
    iteration count decides the answer, and results at a few tens of iterations are
    not converged. :func:`unfold_regular` has no such issue.
    """
    from sigpy.mri import app
    nc, nz, ny, nx = ksp.shape
    if maps.shape != ksp.shape:
        raise ValueError(f"maps {maps.shape} do not match data {ksp.shape}")
    if mask.shape != (nz, ny):
        raise ValueError(f"mask {mask.shape} does not match data {(nz, ny)}")
    scale = _normalise(ksp)
    if scale == 0.0:
        return np.zeros((nz, ny, nx), dtype=complex)
    y = ksp / scale
    w = np.broadcast_to(mask[None, :, :, None], ksp.shape).astype(float)
    if l1 > 0:
        img = app.L1WaveletRecon(y, maps, l1, weights=w, wave_name=wave_name,
                                 max_iter=max_iter, show_pbar=False).run()
    else:
        img = app.SenseRecon(y, maps, lamda=lam, weights=w, max_iter=max_iter,
                             show_pbar=False).run()
    return np.asarray(img) * scale / np.sqrt(nz * ny * nx)


def regular_mask(shape_zy, ry: int, rz: int) -> np.ndarray:
    """``(nz, ny)`` mask of a regular (SENSE-style) undersampling by ``(ry, rz)``.

    Sampled where the *centred* index is a multiple of the factor, so the aliased
    image is exactly the sum of the ``ry * rz`` voxels one alias period apart.
    """
    nz, ny = shape_zy
    if ny % ry or nz % rz:
        raise ValueError("matrix must be divisible by the acceleration factors")
    kz, ky = np.meshgrid(np.arange(nz) - nz // 2, np.arange(ny) - ny // 2,
                         indexing="ij")
    return (kz % rz == 0) & (ky % ry == 0)


def unfold_regular(ksp: np.ndarray, maps: np.ndarray, ry: int, rz: int):
    """Classical SENSE unfolding (Pruessmann et al. 1999) of a regular undersampling.

    ``ksp`` is ``(ncoil, nz, ny, nx)`` holding only the lines of
    :func:`regular_mask` (zeros elsewhere). Returns ``(image, g)``: the complex
    image ``(nz, ny, nx)`` in the scaling of ``images_from_kspace`` and the
    geometry factor ``g`` (``SNR_acc = SNR_full / (g * sqrt(R))``).

    Per aliased voxel the ``ry * rz`` superposed voxels are recovered by least
    squares from the channels, ``x = pinv(S / R) y``. Channels are taken as
    uncorrelated with equal noise (identity noise covariance), which a real coil does
    not satisfy. This is a closed-form per-voxel solve, so unlike the iterative
    reconstructions it has no convergence question and gives the exact g.
    """
    nc, nz, ny, nx = ksp.shape
    if maps.shape != ksp.shape:
        raise ValueError(f"maps {maps.shape} do not match data {ksp.shape}")
    r = ry * rz
    ax = (-3, -2, -1)
    ali = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp, axes=ax), axes=ax),
                          axes=ax)                       # (nc, nz, ny, nx)
    z0, y0 = nz // rz, ny // ry
    # alias group of the aliased voxel (a, b): voxels (a + i*z0, b + j*y0)
    ii, jj = np.meshgrid(np.arange(rz), np.arange(ry), indexing="ij")
    zz = np.arange(z0)[:, None, None] + (ii.ravel() * z0)[None, None, :]   # (z0,1,r)
    yy = np.arange(y0)[None, :, None] + (jj.ravel() * y0)[None, None, :]   # (1,y0,r)
    zz = np.broadcast_to(zz, (z0, y0, r))
    yy = np.broadcast_to(yy, (z0, y0, r))
    y_al = ali[:, :z0, :y0, :]                           # (nc, z0, y0, nx)
    S = maps[:, zz, yy, :]                               # (nc, z0, y0, r, nx)
    A = np.moveaxis(S, (1, 2, 4), (0, 1, 2)) / r         # (z0, y0, nx, nc, r)
    yv = np.moveaxis(y_al, (1, 2, 3), (0, 1, 2))[..., None]   # (z0, y0, nx, nc, 1)
    AH = np.swapaxes(A.conj(), -1, -2)
    gram = AH @ A                                        # (z0, y0, nx, r, r)
    inv = np.linalg.inv(gram + 1e-12 * np.eye(r))
    x = inv @ (AH @ yv)                                  # (z0, y0, nx, r, 1)
    g = np.sqrt(np.real(np.einsum("...ii->...i", inv))
                * np.real(np.einsum("...ii->...i", gram)))   # (z0, y0, nx, r)

    img = np.zeros((nz, ny, nx), dtype=complex)
    gmap = np.zeros((nz, ny, nx))
    xi = np.arange(nx)[None, None, :]
    for p in range(r):
        zp, yp = zz[..., p][..., None], yy[..., p][..., None]
        img[zp, yp, xi] = x[..., p, 0]
        gmap[zp, yp, xi] = g[..., p]
    return img, gmap


def coil_combine_rss(ksp: np.ndarray) -> np.ndarray:
    """Root-sum-of-squares of the channel images. **Magnitude only: phase is lost.**

    Provided as the negative control for the Dixon separation, not for use.
    """
    axes = (-3, -2, -1)
    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp, axes=axes), axes=axes),
                          axes=axes)
    return np.sqrt((np.abs(img) ** 2).sum(0))
