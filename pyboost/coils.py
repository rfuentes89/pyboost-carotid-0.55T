"""A SYNTHETIC multi-channel receive array, for testing parallel-imaging code.

Read this before using it
-------------------------
This is a test bench, **not a model of the user's 20-channel head/neck coil**. Its
geometry (number of loops, ring layout, loop radius, distance from the object) is
invented here; the real coil's geometry and sensitivities are not in this repo and
not in any source read. Results obtained with it test whether the SENSE chain is
implemented correctly (unfolding, phase handling, map estimation); they say
nothing reliable about the real coil's g-factor.

Model: each channel is a circular wire loop on a cylinder around the object, its
plane tangent to the cylinder. The field is computed by numerical Biot-Savart and
the receive sensitivity is the circularly polarised transverse part, with B0 along
z:  ``S = Bx - i*By``  (quasi-static: at 0.55T the wavelength is long, so no
dielectric or standing-wave effects are modelled). Maps are scaled so the
root-sum-of-squares over channels is 1 at the centre of the volume.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _loop_field(points: np.ndarray, centre: np.ndarray, e1: np.ndarray,
                e2: np.ndarray, radius: float, n_seg: int = 48) -> np.ndarray:
    """Biot-Savart field (unit current, mu0/4pi = 1) of one loop at ``points`` (N, 3)."""
    th = (np.arange(n_seg) + 0.5) * 2 * np.pi / n_seg
    pos = centre + radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    dl = radius * (2 * np.pi / n_seg) * (-np.sin(th)[:, None] * e1
                                          + np.cos(th)[:, None] * e2)
    b = np.zeros_like(points)
    for p, d in zip(pos, dl):
        r = points - p
        r3 = (np.einsum("ij,ij->i", r, r) + 1e-12) ** 1.5
        b += np.cross(d, r) / r3[:, None]
    return b


def synthetic_array(shape: Tuple[int, int, int], voxel: Tuple[float, float, float],
                    n_azimuth: int = 5, n_rings: int = 4,
                    cylinder_radius: float = 0.085, loop_radius: float = 0.03,
                    ring_pitch: float = 0.03) -> np.ndarray:
    """Complex sensitivities, shape ``(n_azimuth * n_rings, nz, ny, nx)``.

    ``shape`` is ``(nz, ny, nx)`` and ``voxel`` the voxel size ``(dz, dy, dx)`` in
    metres. Defaults: 5 x 4 = 20 loops of 30 mm radius on a 85 mm-radius cylinder,
    rings 30 mm apart along z: **invented** (see the module docstring).
    """
    nz, ny, nx = shape
    z = (np.arange(nz) - (nz - 1) / 2) * voxel[0]
    y = (np.arange(ny) - (ny - 1) / 2) * voxel[1]
    x = (np.arange(nx) - (nx - 1) / 2) * voxel[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    maps = []
    zhat = np.array([0.0, 0.0, 1.0])
    for r in range(n_rings):
        zc = (r - (n_rings - 1) / 2) * ring_pitch
        for a in range(n_azimuth):
            phi = 2 * np.pi * (a + 0.5 * (r % 2)) / n_azimuth
            centre = np.array([cylinder_radius * np.cos(phi),
                               cylinder_radius * np.sin(phi), zc])
            e1 = np.array([-np.sin(phi), np.cos(phi), 0.0])
            b = _loop_field(pts, centre, e1, zhat, loop_radius)
            s = (b[:, 0] - 1j * b[:, 1]).reshape(shape)
            maps.append(s)
    maps = np.stack(maps)
    ctr = maps[:, nz // 2, ny // 2, nx // 2]
    return maps / np.sqrt(np.sum(np.abs(ctr) ** 2))


def noise_correlation(maps: np.ndarray) -> float:
    """Mean absolute overlap between channels' sensitivity magnitudes (0 = disjoint)."""
    m = np.abs(maps).reshape(maps.shape[0], -1)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    g = m @ m.T
    n = g.shape[0]
    return float((g.sum() - n) / (n * (n - 1)))
