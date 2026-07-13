"""Concentric 2D carotid phantom for the MRzero simulation pipeline.

Replaces the 1D isochromat model of the Koma reference with a spatial cross
section: a circular **lumen** (blood), a concentric vessel-**wall** annulus,
surrounding **muscle**, and a subcutaneous **fat** ring near the neck boundary.
Everything outside a neck ellipse is air (excluded, to keep the voxel count
down).

The generator emits per-pixel maps of PD, T1, T2, T2dash and B0 at 0.55T, and a
converter turns them into an ``mr0.CustomVoxelPhantom`` on a grid that matches
the imaging FOV so the simulated k-space reconstructs to a real image.

Relaxation values (0.55T) follow the Koma ``carotid_phantom`` (RR_sim.jl:196-207)
plus muscle. Fat carries its chemical-shift offset as a B0 term (~-80 Hz at
0.55T) so the spectrally-selective FatSat pulse behaves correctly in space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# (PD, T1[s], T2[s]). Air is handled separately (PD = 0, excluded).
TISSUE_PROPERTIES: Dict[str, Dict[str, float]] = {
    "blood":  dict(PD=0.70, T1=1.122, T2=0.263),
    "wall":   dict(PD=0.60, T1=0.750, T2=0.090),
    "muscle": dict(PD=0.70, T1=0.600, T2=0.045),
    "fat":    dict(PD=1.00, T1=0.183, T2=0.093),
}
# Integer labels for the tissue map (0 = air).
LABELS = {"air": 0, "blood": 1, "wall": 2, "muscle": 3, "fat": 4}

FAT_PPM = -3.4e-6
GAMMA = 42.576e6  # Hz/T


@dataclass
class CarotidGeometry:
    """Concentric neck cross section (zoomed carotid view).

    Vessel-wall imaging uses a small FOV at high resolution, so the defaults are
    a ~60 mm zoomed patch where the ~1-2 mm wall is resolved. Radii are in
    metres; the pixel grid matches the FOV so the simulated k-space reconstructs
    to a real image. ``fov`` must equal the imaging FOV of the sequence.
    """
    fov: float = 0.06                 # [m], must match the imaging FOV
    matrix: int = 96                  # phantom grid (pixels per side)
    b0: float = 0.55                  # [T]
    center: tuple = (0.008, -0.004)   # vessel offset from isocentre [m] (anatomical)
    lumen_radius: float = 4.0e-3      # common carotid lumen ~8 mm diameter
    wall_thickness: float = 1.5e-3    # vessel wall (thickened for visibility)
    neck_semi_axes: tuple = (0.028, 0.024)   # neck ellipse (a, b) [m]
    fat_ring_thickness: float = 6.0e-3       # subcutaneous fat rind [m]
    t2dash: float = 0.03              # static-dephasing time constant [s]
    apply_fat_offset: bool = False    # see note below

    # NOTE on fat off-resonance. Physically fat sits ~-80 Hz from water, and the
    # FatSat pulse is spectrally selective there. In this simplified model,
    # however, placing fat at that B0 offset also throws it onto the bSSFP
    # frequency-response profile (80 Hz x TR=7 ms ~ 200 deg/TR), which corrupts
    # its readout signal and masks the FatSat effect. Keeping fat on-resonance
    # (apply_fat_offset=False) reproduces the validated FatSat suppression
    # (~6x). Chemical-shift displacement / fat bSSFP banding is a deferred
    # refinement (would need a multi-peak fat model consistent with the readout).

    def fat_offset_hz(self) -> float:
        return FAT_PPM * self.b0 * GAMMA


def carotid_phantom_maps(geom: CarotidGeometry | None = None) -> Dict[str, np.ndarray]:
    """Return per-pixel maps: ``label, PD, T1, T2, T2dash, B0`` (all matrix x matrix)."""
    g = geom or CarotidGeometry()
    n = g.matrix
    # Pixel centre coordinates in metres, spanning the FOV.
    axis = (np.arange(n) - (n - 1) / 2) * (g.fov / n)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")

    r_vessel = np.hypot(xx - g.center[0], yy - g.center[1])
    # Neck ellipse membership and its inner (muscle) boundary for the fat rind.
    a, b = g.neck_semi_axes
    ell = (xx / a) ** 2 + (yy / b) ** 2
    a_in = a - g.fat_ring_thickness
    b_in = b - g.fat_ring_thickness
    ell_in = (xx / a_in) ** 2 + (yy / b_in) ** 2

    label = np.full((n, n), LABELS["air"], dtype=np.int8)
    label[ell <= 1.0] = LABELS["fat"]        # subcutaneous fat rind
    label[ell_in <= 1.0] = LABELS["muscle"]  # muscle fills the interior
    # Carotid vessel drawn on top of muscle: wall annulus then lumen.
    label[r_vessel <= g.lumen_radius + g.wall_thickness] = LABELS["wall"]
    label[r_vessel <= g.lumen_radius] = LABELS["blood"]

    pd = np.zeros((n, n)); t1 = np.ones((n, n)); t2 = np.ones((n, n)) * 0.1
    for name, lab in LABELS.items():
        if name == "air":
            continue
        m = label == lab
        pd[m] = TISSUE_PROPERTIES[name]["PD"]
        t1[m] = TISSUE_PROPERTIES[name]["T1"]
        t2[m] = TISSUE_PROPERTIES[name]["T2"]

    t2dash = np.where(label != LABELS["air"], g.t2dash, 0.0)
    fat_b0 = g.fat_offset_hz() if g.apply_fat_offset else 0.0
    b0 = np.where(label == LABELS["fat"], fat_b0, 0.0)
    return {"label": label, "PD": pd, "T1": t1, "T2": t2, "T2dash": t2dash, "B0": b0}


def to_mrzero_phantom(maps: Dict[str, np.ndarray], fov: float):
    """Convert phantom maps into an ``mr0.CustomVoxelPhantom`` (air excluded)."""
    import MRzeroCore as mr0

    n = maps["PD"].shape[0]
    axis = (np.arange(n) - (n - 1) / 2) / n          # normalised positions [-0.5, 0.5)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    keep = maps["PD"] > 0                              # drop air voxels

    pos = np.stack([xx[keep] * fov, yy[keep] * fov, np.zeros(keep.sum())], axis=1)
    return mr0.CustomVoxelPhantom(
        pos=pos.tolist(),
        PD=maps["PD"][keep].tolist(),
        T1=maps["T1"][keep].tolist(),
        T2=maps["T2"][keep].tolist(),
        T2dash=maps["T2dash"][keep].tolist(),
        B0=maps["B0"][keep].tolist(),
        D=0.0,
        voxel_size=fov / n,
        voxel_shape="box",
    )
