"""BOOST sequence parameters.

Mirrors ``seq_params`` and the general timing block of the Koma reference
(``RR_sim.jl:15-50``) and adds the imaging-geometry fields that a real spatial
readout needs (the 1D contrast simulation had none).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BoostParams:
    # --- Cardiac / segmentation (RR_sim.jl:25-33) ---
    rr: float = 1.3                      # nominal RR interval [s]
    dummy_heart_beats: int = 3           # heartbeats to reach steady state
    inav_lines: int = 6                  # bSSFP start-up (iNAV) ramp pulses
    im_segments: int = 30                # imaging TRs (phase-encode lines) per shot

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
    ir_inversion_time: float = 90e-3     # adiabatic IR inversion time TI [s]

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
