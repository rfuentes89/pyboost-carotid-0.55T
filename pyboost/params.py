"""BOOST and REACT sequence parameters.

:class:`BoostParams` mirrors ``seq_params`` and the general timing block of the
Koma reference (``RR_sim.jl:15-50``) and adds the imaging-geometry fields that a
real spatial readout needs (the 1D contrast simulation had none).

:class:`ReactParams` describes the REACT angiography variant, which shares the
preparation modules but replaces the balanced bSSFP readout with a spoiled
dual-echo Dixon train (see :mod:`pyboost.readout_dixon`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BoostParams:
    # --- Cardiac / segmentation (RR_sim.jl:25-33) ---
    rr: float = 1.3                      # nominal RR interval [s]
    dummy_heart_beats: int = 3           # heartbeats to reach steady state
    inav_lines: int = 6                  # bSSFP start-up (iNAV) ramp pulses
    im_segments: int = 30                # imaging TRs (phase-encode lines) per shot
    centric: bool = False                # centric k-space ordering (ky=0 acquired first)

    # --- Timing (RR_sim.jl:16-33) ---
    trf: float = 500e-6                  # imaging RF (block) duration [s]
    tr: float = 7e-3                     # bSSFP TR [s]  ("RF Low SAR")
    te: float = 3.51e-3                  # bSSFP TE [s]  (~TR/2)
    inav_flip_angle: float = 3.2         # iNAV ramp start flip angle [deg]

    # --- Flip angles per contrast (RR_sim.jl:36) ---
    # [bright-blood contrast, reference contrast]
    im_flip_angle: Tuple[float, float] = (110.0, 80.0)

    # --- Preparation modules (RR_sim.jl:22,37,38) ---
    t2prep_duration: float = 50e-3       # T2-prep echo time [s]
    fatsat_flip_angle: float = 180.0     # FatSat flip angle [deg]
    fatsat_duration: float = 26.624e-3   # gaussian FatSat duration [s] (RR_sim.jl:21)
    ir_inversion_time: float = 90e-3     # IR inversion time TI [s]
    inversion_kind: str = "block"        # "block" (MRzero-simulatable) | "adiabatic" (scanner)

    # --- Imaging geometry (new: required for real spatial encoding) ---
    fov: float = 200e-3                  # in-plane FOV [m]
    nx: int = 128                        # readout samples (frequency encode)
    ny: int = 120                        # phase-encode lines (matrix, multiple of im_segments is convenient)
    slice_thickness: float = 5e-3        # [m]
    readout_bandwidth: float = 400.0     # Hz/pixel; low BW favours SNR at low field
    tbw_excitation: float = 2.0          # time-bandwidth product of the slice-select sinc

    def __post_init__(self) -> None:
        if self.te > self.tr:
            raise ValueError("TE must not exceed TR")
        if self.nx % 2 or self.ny % 2:
            raise ValueError("nx and ny should be even")

    @property
    def n_shots(self) -> int:
        """Acquisition heartbeats needed to fill k-space in segments."""
        return -(-self.ny // self.im_segments)  # ceil division


def null_time_after_t2prep(t1: float, t2: float, t2prep_te: float) -> float:
    """Inversion time [s] that nulls a tissue placed *after* a T2 preparation.

    REACT inverts magnetization that the T2-prep has already attenuated, so the
    plain STIR formula ``TI = T1*ln2`` does not apply. Starting from ``M0``:

    * after a T2-prep of echo time ``TE_prep``:  ``Mz = M0*E2``, ``E2 = exp(-TE_prep/T2)``
    * after the non-selective 180 deg:           ``Mz = -M0*E2``
    * after recovering for ``TI``:               ``Mz = M0*(1 - (1 + E2)*exp(-TI/T1))``

    Setting that to zero gives ``TI = T1 * ln(1 + E2)``, which correctly reduces
    to ``T1*ln2`` when there is no T2-prep (``E2 = 1``).

    Passing ``t2prep_te = 0`` therefore returns the ordinary STIR null time.
    """
    if t1 <= 0 or t2 <= 0:
        raise ValueError("T1 and T2 must be positive")
    e2 = math.exp(-t2prep_te / t2)
    return t1 * math.log1p(e2)


@dataclass
class ReactParams:
    """REACT (Relaxation-Enhanced Angiography without Contrast and Triggering).

    Flow-independent, non-gated 3D angiography: ``T2-prep -> non-selective IR
    (short TI) -> spoiled dual-echo Dixon train``. Introduced by Yoneyama et al.,
    *Magn Reson Imaging* 2019;63:137-146 (doi:10.1016/j.mri.2019.08.017).

    Defaults are traceable to the published protocols; values that had to be
    derived for 0.55T (because every REACT paper is at 1.5T or 3T) are marked
    DERIVED and explained. Times are in SI seconds, matching
    :class:`BoostParams`.
    """

    # --- Preparation (published REACT values, identical at 1.5T and 3T) ---
    # T2-prep 50 ms: Pennig 2020 (Clin Neuroradiol, 3T) and Isaak 2021 (JCMR,
    # 1.5T). Independently corroborated at 0.55T by Castillo-Passi et al.,
    # MRM 2024, whose whole-heart CMRA at this field also uses 50 ms.
    t2prep_duration: float = 50e-3
    trf: float = 500e-6                  # hard-pulse duration [s] (user's scanner)
    inversion_kind: str = "block"        # "block" (MRzero-simulatable) | "adiabatic" (scanner)
    # Optional spectral fat saturation. REACT relies on Dixon instead, so this is
    # only used when build_react_sequence(use_fatsat=True); the duration matches
    # the module already tuned for 0.55T in prep.fat_sat.
    fatsat_duration: float = 26.624e-3
    fatsat_flip_angle: float = 180.0

    # TI. DERIVED, and the only parameter with no usable literature anchor: the
    # single published value is "~70 ms" at 1.5T (Isaak 2021), and T1 is ~32%
    # shorter at 0.55T (Campbell-Washburn, Radiology 2019). Left as None so it is
    # computed from the measured 0.55T relaxation table via
    # :func:`null_time_after_t2prep` -- see ``ti_null_tissue``.
    ti: float | None = None
    ti_null_tissue: str = "fat"
    # Why fat: Pennig 2020 describes the module as a non-volume-selective STIR,
    # i.e. its stated job is nulling short-T1 fat. With the 0.55T table this
    # lands at ~84 ms, close to the 70 ms published at 1.5T -- a useful sanity
    # check that the derivation is not wandering. Set to "muscle" to null
    # background muscle instead, or set ``ti`` outright to override.

    # --- Readout: spoiled dual-echo Dixon ---
    # Flip angle 15 deg: Pennig 2020 states the low flip angle is chosen
    # deliberately to maximise arterial signal.
    flip_angle: float = 15.0
    # Dixon echo spacing. DERIVED: at 0.55T the fat-water shift is only ~80 Hz,
    # so opposed phase sits at 1/(2*80) = 6.3 ms versus 2.3 ms at 1.5T. REACT
    # uses "semi-flexible echo times", not exact opposed phase: its 1.5T spacing
    # of 2.08 ms (TE1 1.72, TE2 3.80; Isaak 2021 Table 1) corresponds to 162.6
    # deg of water-fat phase evolution. Preserving that angle at 0.55T gives
    # ~5.7 ms. Computed in readout_dixon.dixon_echo_spacing(); None = derive.
    dixon_phase_deg: float = 162.6
    delta_te: float | None = None
    te1: float | None = None             # None = shortest achievable
    tr: float | None = None              # None = shortest achievable
    spoiler_cycles: float = 4.0          # phase wraps per voxel across the spoiler
    rf_spoil_increment: float = 117.0    # quadratic RF spoiling increment [deg]

    # --- Shot structure (non-gated: REACT for the neck uses no ECG) ---
    # TFE factor. DERIVED: Isaak 2021 uses 60 at TR 5.8 ms, i.e. a 348 ms shot.
    # At 0.55T the Dixon spacing forces TR ~15.8 ms, so matching that shot window
    # (which is what sets how far the prepared contrast decays) allows ~22.
    tfe_factor: int = 22
    shot_interval: float = 1.0           # time from one preparation to the next [s]
    dummy_shots: int = 2                 # shots run without ADC to reach steady state
    centric: bool = True                 # centre of k-space right after the TI

    # --- Imaging geometry ---
    fov: float = 200e-3
    nx: int = 128
    ny: int = 120
    nz: int = 1                          # partitions; 1 = single-slice 2D, >1 = 3D slab
    slice_thickness: float = 5e-3        # slab thickness when nz > 1 [m]
    readout_bandwidth: float = 300.0     # Hz/pixel -- DERIVED, see below
    # The long Dixon spacing at 0.55T is not wasted time: it is spent on a long,
    # low-bandwidth readout, which is exactly the SNR lever low field needs.
    # But the budget is tighter than a back-of-envelope suggests. Echo spacing is
    # consumed by the *whole* readout pair -- both gradient ramps plus the
    # slew-limited full-area rewinder (~1.6 ms at 25 T/m/s) -- not just the ADC
    # windows. At nx=128 / FOV 200 mm the floor is ~280 Hz/pixel; below that
    # _build_kernel raises rather than silently mistiming the echoes. 300 leaves
    # a little margin. Chemical-shift displacement is then 80/300 = 0.27 pixel.
    tbw_excitation: float = 2.0

    def __post_init__(self) -> None:
        if self.nx % 4:
            raise ValueError("nx should be a multiple of 4 (keeps the ADC on the raster)")
        if self.ny % 2 or self.nz < 1:
            raise ValueError("ny must be even and nz at least 1")
        if self.tfe_factor < 1:
            raise ValueError("tfe_factor must be at least 1")
        if self.inversion_kind not in ("block", "adiabatic"):
            raise ValueError(f"unknown inversion kind {self.inversion_kind!r}")

    @property
    def n_encodes(self) -> int:
        """Total phase-encode steps (ky x kz)."""
        return self.ny * self.nz

    @property
    def n_shots(self) -> int:
        """Shots needed to fill k-space at ``tfe_factor`` encodes per shot."""
        return -(-self.n_encodes // self.tfe_factor)  # ceil division

    def resolved_ti(self) -> float:
        """The TI actually used: explicit if set, else derived (see ``ti``)."""
        if self.ti is not None:
            return self.ti
        from .phantom import TISSUE_PROPERTIES
        try:
            tissue = TISSUE_PROPERTIES[self.ti_null_tissue]
        except KeyError:
            raise ValueError(
                f"ti_null_tissue={self.ti_null_tissue!r} is not in TISSUE_PROPERTIES "
                f"({sorted(TISSUE_PROPERTIES)})"
            ) from None
        return null_time_after_t2prep(tissue["T1"], tissue["T2"],
                                      self.t2prep_duration)
