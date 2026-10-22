"""Variable-density undersampling pattern for accelerated 3D REACT.

Provenance -- read this before citing
-------------------------------------
Every published REACT protocol is accelerated (compressed SENSE 3-10x, or SENSE
3.2 in Isaak 2021), but none describes the sampling pattern: the Philips
"Compressed SENSE" pattern is not documented in the papers read. Everything in
this module is therefore a **design choice of this code, not a published value**:
the density law, the width of the fully sampled centre and the seed.

What is fixed by the physics rather than by choice:

* the pattern lives on the ``(ky, kz)`` grid; the readout axis is always fully
  sampled, so each acquired line is one TR;
* the centre of k-space is sampled completely. It carries the contrast and gives
  the compressed-sensing reconstruction something to anchor on;
* the number of acquired lines is **exactly** ``round(n / R)``, so the quoted
  acceleration is the real one and scan time is predictable.

Law: after the central block, lines are drawn without replacement with
probability proportional to ``(1 - rho/rho_max) ** power``, where ``rho`` is the
*normalised* radius ``sqrt(((ky-ny/2)/(ny/2))**2 + ((kz-nz/2)/(nz/2))**2)``. For
isotropic voxels that equals the physical radius in k, which is what matters for
the decay of the prepared magnetization along a shot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np


def normalised_radius(ky, kz, ny: int, nz: int):
    """Radius in units of the k-space half-width on each axis (corner = sqrt(2))."""
    return np.sqrt(((np.asarray(ky) - ny / 2) / (ny / 2)) ** 2
                   + ((np.asarray(kz) - nz / 2) / (nz / 2)) ** 2)


@lru_cache(maxsize=16)
def _mask(ny: int, nz: int, acceleration: float, center_fraction: float,
          power: float, seed: int) -> np.ndarray:
    n = ny * nz
    target = int(round(n / acceleration))
    kz_idx, ky_idx = np.meshgrid(np.arange(nz), np.arange(ny), indexing="ij")
    rho = normalised_radius(ky_idx, kz_idx, ny, nz)

    n_center = min(target, int(round(center_fraction * n)))
    order = np.argsort(rho.ravel(), kind="stable")          # nearest the centre first
    chosen = np.zeros(n, dtype=bool)
    chosen[order[:n_center]] = True

    remaining = target - n_center
    if remaining > 0:
        rho_max = float(rho.max()) * (1.0 + 1e-9)
        w = np.clip(1.0 - rho.ravel() / rho_max, 1e-12, None) ** power
        w[chosen] = 0.0
        rng = np.random.default_rng(seed)
        pick = rng.choice(n, size=remaining, replace=False, p=w / w.sum())
        chosen[pick] = True
    out = chosen.reshape(nz, ny)
    out.setflags(write=False)
    return out


def sampling_mask(ny: int, nz: int, acceleration: float = 1.0,
                  center_fraction: float = 0.03, power: float = 3.0,
                  seed: int = 0) -> np.ndarray:
    """Boolean ``(nz, ny)`` mask of the acquired ``(kz, ky)`` lines.

    ``acceleration = 1`` returns the full grid. ``center_fraction`` is the share
    of all lines forming the fully sampled central block (taken by normalised
    radius). The returned array is read-only and cached.
    """
    if acceleration < 1.0:
        raise ValueError("acceleration must be at least 1")
    if not 0.0 <= center_fraction <= 1.0:
        raise ValueError("center_fraction must be in [0, 1]")
    if acceleration == 1.0:
        return np.ones((nz, ny), dtype=bool)
    return _mask(ny, nz, float(acceleration), float(center_fraction),
                 float(power), int(seed))


def acquired_encodes(ny: int, nz: int, **kw) -> List[Tuple[int, int]]:
    """The acquired ``(ky, kz)`` pairs in raster order (kz outer, ky inner)."""
    m = sampling_mask(ny, nz, **kw)
    kz, ky = np.nonzero(m)
    return list(zip(ky.tolist(), kz.tolist()))
