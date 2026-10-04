"""Cartesian reconstruction for the REACT dual-echo acquisition.

The error-prone step is putting samples back where they belong. A REACT shot
plays ``tfe_factor`` TRs, each recording **two** ADCs (echo 1, then echo 2) at the
*same* phase-encode position, and shots walk k-space in centric order. Get any of
that wrong and the images look plausibly blurry rather than obviously broken.

The defence is not to restate the ordering at all: :func:`kspace_from_signal`
calls the functions that built the sequence (:func:`pyboost.react.shot_encodes`),
so the reconstruction cannot drift out of sync with the ``.seq``.

.. note::
   Multi-channel data must be coil-combined **with phase preserved** before
   :mod:`pyboost.dixon` sees it -- see the warning there.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .params import ReactParams
from .react import shot_encodes


def kspace_from_signal(signal, p: ReactParams,
                       acquired_shots: int | None = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Sort a flat ADC stream into one k-space array per echo.

    ``signal`` is the complex stream in acquisition order, ``2*nx`` samples per
    TR (a torch tensor is accepted). Dummy shots record no ADC, so they are not
    in the stream. ``acquired_shots`` defaults to ``p.n_shots``.

    Returns ``(k1, k2)``, each ``(nz, ny, nx)`` (``nz`` is 1 for 2D). Encodes
    that were not acquired stay zero.
    """
    if hasattr(signal, "detach"):
        signal = signal.detach().cpu().numpy()
    sig = np.asarray(signal).reshape(-1)
    n_shots = p.n_shots if acquired_shots is None else acquired_shots

    k1 = np.zeros((p.nz, p.ny, p.nx), dtype=complex)
    k2 = np.zeros_like(k1)
    pos = 0
    for shot in range(n_shots):
        for ky, kz in shot_encodes(shot, p):
            if pos + 2 * p.nx > sig.size:
                raise ValueError(
                    f"signal has {sig.size} samples but the encode list needs at "
                    f"least {pos + 2 * p.nx}. Check acquired_shots, and that "
                    f"dummy shots were excluded from the ADC stream.")
            k1[kz, ky, :] = sig[pos:pos + p.nx]
            k2[kz, ky, :] = sig[pos + p.nx:pos + 2 * p.nx]
            pos += 2 * p.nx
    return k1, k2


def images_from_kspace(k1: np.ndarray, k2: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Centred inverse FFT of both echoes (2D for one partition, else 3D)."""
    axes = (-2, -1) if k1.shape[0] == 1 else (-3, -2, -1)

    def _ifft(k):
        return np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(k, axes=axes), axes=axes), axes=axes)

    return _ifft(k1), _ifft(k2)
