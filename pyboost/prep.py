"""Magnetization-preparation modules for BOOST, ported from the Koma reference.

Each function returns a ``list`` of *blocks*, where a block is itself a list of
PyPulseq events to be handed to ``Sequence.add_block(*events)``. Returning the
block structure (instead of mutating a ``Sequence``) keeps the modules pure and
easy to unit-test (timing and flip angles can be checked on the returned events).

References to the Koma implementation are given per function
(``RR_sim.jl`` line ranges).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pypulseq as pp

from .system import fat_frequency

Block = List  # a block is a list of PyPulseq events
Blocks = List[Block]


def _spoiler(system: pp.Opts, amp_mT_m: float, flat_time: float,
             rise_time: float, axis: str = "z") -> object:
    """Trapezoidal spoiler, amplitude given in mT/m (Koma uses ~8 mT/m)."""
    return pp.make_trapezoid(
        axis,
        amplitude=amp_mT_m * 1e-3 * system.gamma,
        flat_time=flat_time,
        rise_time=rise_time,
        system=system,
    )


def fat_sat(system: pp.Opts, flip_angle_deg: float = 180.0,
            duration: float = 26.624e-3, fat_ppm: float = -3.4e-6) -> Blocks:
    """Spectrally-selective gaussian fat saturation (``RR_sim.jl:60-81``).

    The pulse is centred on the fat resonance (``freq_offset = fat_ppm*B0*gamma``,
    about -80 Hz at 0.55T) and bracketed by two opposed spoilers. At low field
    the fat-water separation is small, so the pulse is long (~27 ms) and its
    bandwidth deliberately narrow to avoid saturating on-resonance water.
    """
    df = fat_frequency(system, fat_ppm)
    # Snap the RF duration onto the gradient raster so the RF block (dead time +
    # shape + ringdown, all multiples of the raster) lands on the block raster.
    # Koma's 26.624 ms is off the 10 us grid.
    duration = round(duration / system.grad_raster_time) * system.grad_raster_time
    rf = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=duration,
        freq_offset=df,
        bandwidth=abs(df),           # ~80 Hz: covers fat, keeps water outside
        system=system,
        use="saturation",
    )
    sp1 = _spoiler(system, -8.0, flat_time=3000e-6, rise_time=500e-6)
    sp2 = _spoiler(system, +8.0, flat_time=3000e-6, rise_time=500e-6)
    return [[sp1], [rf], [sp2]]


# MLEV-16 phase pattern for a train of 180 deg pulses (+1 -> +y, -1 -> -y). Any
# prefix of length 1, 2, 4, 8 or 16 is balanced. These are plain hard pulses, so
# this is *not* claimed to reproduce the composite or adiabatic refocusing used in
# the published MLEV4/MLEV8 T2 preparations (Paredes 2025, 0.55T) or in REACT
# itself, whose pulse details were not available; it is the same idea (phase
# cycling the refocusing train) on this repo's hard-pulse building blocks.
_MLEV16 = (+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, +1, -1, -1, +1)
_VALID_REFOCUS = (1, 2, 4, 8, 16)


def t2_prep(system: pp.Opts, te: float = 50e-3, trf: float = 500e-6,
            n_refocus: int = 1) -> Blocks:
    """MLEV-style composite T2 preparation (``RR_sim.jl:83-95``).

    ``90x -- tau -- [180(+-y) -- 2tau] x n -- ... -- (-90x) -- spoiler``, with
    ``tau = TE/(2n)`` so TE, measured between the two 90 deg pulse centres, is
    unchanged by ``n_refocus``. The 180s follow the MLEV-16 phase pattern
    (``y, y, -y, -y, -y, y, y, -y, ...``). Restores T2-weighted magnetization to
    +z; the spoiler dephases residual transverse signal. This is the block that
    gives BOOST its bright-blood contrast.

    ``n_refocus`` is 1 (the original single-180 block, kept as the default so
    BOOST and MRA are unchanged), 2, 4, 8 or 16. REACT is described with four
    refocusing pulses (Gietzen 2025, Table 1), so :class:`ReactParams` uses 4.

    What the extra pulses buy is robustness, not different contrast, and the
    ideal-pulse MRzero simulation cannot show it: with no relaxation, a +-20% B1
    error leaves Mz at 0.83 for ``n_refocus=1`` but at 1.000 for 4 or 8 (measured
    with a Bloch simulation of the returned blocks; ``tests/test_prep.py`` pins
    it), and without the +-y cycling 8 pulses fall back to 0.96.

    Delays are snapped to the block raster, so TE can differ from the request by
    up to ``n_refocus`` raster steps (80 us at 8 pulses). For the default
    ``te=50 ms, n_refocus=1`` nothing moves.
    """
    if n_refocus not in _VALID_REFOCUS:
        raise ValueError(f"n_refocus must be one of {_VALID_REFOCUS}, got {n_refocus}")
    tau = te / (2 * n_refocus)
    lead = tau - 1.5 * trf            # 90 end -> first 180 start (and last 180 -> -90)
    mid = 2 * tau - 2 * trf           # between consecutive 180s
    if lead <= 0 or (n_refocus > 1 and mid <= 0):
        raise ValueError(
            f"T2-prep TE={te*1e3:.1f} ms too short for RF duration with "
            f"{n_refocus} refocusing pulse(s)")
    raster = system.block_duration_raster

    def snap(t: float) -> float:
        return round(t / raster) * raster

    rf_90x = pp.make_block_pulse(np.pi / 2, duration=trf, phase_offset=0.0,
                                 system=system, use="preparation")
    rf_m90x = pp.make_block_pulse(np.pi / 2, duration=trf, phase_offset=np.pi,
                                  system=system, use="preparation")
    sp = _spoiler(system, 8.0, flat_time=6000e-6, rise_time=600e-6)

    blocks: Blocks = [[rf_90x], [pp.make_delay(snap(lead))]]
    for k in range(n_refocus):
        phase = np.pi / 2 if _MLEV16[k] > 0 else -np.pi / 2
        blocks.append([pp.make_block_pulse(np.pi, duration=2 * trf,
                                           phase_offset=phase, system=system,
                                           use="preparation")])
        blocks.append([pp.make_delay(snap(lead if k == n_refocus - 1 else mid))])
    blocks += [[rf_m90x], [sp]]
    return blocks


def inversion(system: pp.Opts, post_delay: float, kind: str = "block",
              block_duration: float = 1e-3, beta: float = 670.0, mu: float = 5.0,
              duration: float = 10.24e-3) -> Blocks:
    """Inversion pulse + spoiler + TI delay.

    ``kind``:

    * ``"block"`` -- a hard 180 deg pulse. This is the default because it is what
      the MRzero PDG simulator models correctly (it treats each RF as an
      instantaneous rotation). At 0.55T B1 homogeneity is good and SAR is low, so
      a hard inversion is perfectly usable.
    * ``"adiabatic"`` -- a hyperbolic-secant pulse mirroring the Koma reference
      (``RR_sim.jl:97-124``: beta=6.7e2, mu=5, Trf=10.24 ms). B1-insensitive and
      preferable on the scanner, BUT the MRzero PDG does **not** reproduce
      adiabatic inversion (the frequency sweep is lost in the rotation model, so
      it saturates instead of inverts). Use it for the exported ``.seq`` /
      KomaMRI (Bloch), not for MRzero validation.

    ``post_delay`` is the time from the end of the inversion spoiler to the next
    block (the remaining inversion-recovery delay; the caller computes it so TI
    is measured to the start of k-space acquisition).
    """
    if post_delay < 0:
        raise ValueError(
            f"inversion post_delay is negative ({post_delay*1e3:.2f} ms): TI is "
            "shorter than the inversion + spoiler + start-up train."
        )
    if kind == "block":
        rf = pp.make_block_pulse(np.pi, duration=block_duration, system=system,
                                 use="inversion")
    elif kind == "adiabatic":
        rf = pp.make_adiabatic_pulse("hypsec", beta=beta, mu=mu, duration=duration,
                                     system=system, use="inversion")
    else:
        raise ValueError(f"unknown inversion kind {kind!r} (block|adiabatic)")
    sp = _spoiler(system, 8.0, flat_time=6000e-6, rise_time=600e-6)
    return [[rf], [sp], [pp.make_delay(post_delay)]]
