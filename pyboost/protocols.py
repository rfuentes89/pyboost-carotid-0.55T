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
    * ``readout_bandwidth``: **derived, not chosen**. The ADC dwell must sit on a 2.5 us
      grid so the readout stays on the 10 us gradient raster, so at nx = 208 only
      15 us (320.5 Hz/pixel) or 12.5 us (384.6 Hz/pixel) exist. The slew-limited
      readout pair does not fit in the 5.67 ms Dixon spacing with 15 us (requests of
      300 and 345 Hz/pixel both round to it and are rejected) and fits with 12.5 us.
      The preset therefore asks for exactly 384.6 Hz/pixel. An earlier note called this
      "350 Hz/pixel, 7.4% more noise than 300"; that understated it: the effective
      bandwidth is 384.6, so noise is about 11% above the 2D default (effective
      312.5 Hz/pixel, sqrt(384.6/312.5)) and chemical-shift displacement is
      80/384.6 = 0.21 pixel;
    * everything else (T2-prep, TI, flip angle, Dixon echo times, ``tfe_factor``
      22, ``shot_interval``) is the ``ReactParams`` default with its own
      provenance.
    """
    base = dict(nx=208, ny=208, nz=88, fov=0.2,
                slice_thickness=88 * 0.96e-3, acceleration=acceleration,
                readout_bandwidth=1.0 / (208 * 12.5e-6))   # 12.5 us dwell = 384.6 Hz/px
    base.update(overrides)
    return ReactParams(**base)
