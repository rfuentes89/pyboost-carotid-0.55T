"""Named protocol presets. Each says where every number comes from."""

from __future__ import annotations

from .params import ReactParams


def react_3d_208x208x88(acceleration: float = 5.0, **overrides) -> ReactParams:
    """3D REACT at 0.96 mm isotropic, matrix 208 x 208 x 88, accelerated.

    Provenance of each number:

    * matrix 208 x 208 x 88, voxel 0.96 mm isotropic (FOV 199.7 mm, slab
      84.5 mm): **the user's own BOOST and CMRA protocols**, not this repo and
      not a REACT publication;
    * ``acceleration = 5``: chosen by the user. Published REACT uses 3-10x
      (Gietzen 2024: compressed SENSE 7; Janssen 2025: CS 4 and 8; Isaak 2021:
      SENSE 3.2); the sampling pattern is this code's design (:mod:`pyboost.sampling`);
    * ``readout_bandwidth = 350`` Hz/pixel: **derived, not chosen**. At nx = 208
      the slew-limited readout pair does not fit in the 5.67 ms Dixon spacing
      below about 346 Hz/pixel (300 fails, 345 fails, 350 builds); this is the
      lowest round value that fits. Cost: noise grows with sqrt(BW), so about
      7.4% more than at 300 Hz/pixel, and chemical-shift displacement is
      80/350 = 0.23 pixel;
    * everything else (T2-prep, TI, flip angle, Dixon echo times, ``tfe_factor``
      22, ``shot_interval``) is the ``ReactParams`` default with its own
      provenance.
    """
    base = dict(nx=208, ny=208, nz=88, fov=0.2,
                slice_thickness=88 * 0.96e-3, acceleration=acceleration,
                readout_bandwidth=350.0)
    base.update(overrides)
    return ReactParams(**base)
