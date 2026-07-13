"""Balanced SSFP (bSSFP) segmented readout with real Cartesian encoding.

This is the piece the Koma reference never had: the 1D contrast simulation used
a single ADC sample per TR with no spatial encoding. Here every TR is a fully
balanced bSSFP block (all three gradient axes have zero net moment over the TR),
with a frequency-encoded readout, a per-line phase encode, alternating RF phase,
and an iNAV start-up ramp -- mirroring the flip-angle schedule of
``RR_sim.jl:126-147``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pypulseq as pp

from .params import BoostParams

Block = List
Blocks = List[Block]


@dataclass
class _Kernel:
    """Reusable, line-independent bSSFP building blocks and fill delays."""
    gz: object
    gzr: object          # slice rephaser (refocuses slice at the echo)
    gz_rew: object       # extra slice rewinder so total gz over TR is zero
    gx: object           # readout gradient
    gx_pre: object       # readout prephaser (-gx.area/2)
    gx_post: object      # readout rewinder (-gx.area/2)
    adc: object
    pre_dur: float       # fixed duration of the prephaser block
    post_dur: float      # fixed duration of the rewinder block
    te_fill: float       # delay before readout so the echo lands at TE
    tr_fill: float       # trailing delay so the block totals TR
    delta_k: float


def _build_kernel(system: pp.Opts, p: BoostParams) -> _Kernel:
    delta_k = 1.0 / p.fov

    # Choose the ADC dwell on a 2.5 us grid (25x the 100 ns ADC raster). For an
    # nx divisible by 4 this keeps the readout duration nx*dwell on the 10 us
    # gradient raster, so the readout gradient flat top and the ADC line up and
    # check_timing stays clean. The effective bandwidth is snapped accordingly.
    dwell_grid = 25 * system.adc_raster_time          # 2.5 us
    dwell = round((1.0 / (p.readout_bandwidth * p.nx)) / dwell_grid) * dwell_grid
    dwell = max(dwell, dwell_grid)
    adc_dur = p.nx * dwell
    if abs(round(adc_dur / system.grad_raster_time) * system.grad_raster_time
           - adc_dur) > 1e-12:
        raise ValueError("readout duration is off the gradient raster; use nx % 4 == 0")

    # Slice-select excitation (flip is set per-TR; geometry is fixed here).
    _, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(p.im_flip_angle[0]), duration=p.trf,
        slice_thickness=p.slice_thickness, time_bw_product=p.tbw_excitation,
        return_gz=True, system=system, use="excitation",
    )

    # Readout + ADC.
    gx = pp.make_trapezoid("x", flat_area=p.nx * delta_k, flat_time=adc_dur,
                           system=system)
    adc = pp.make_adc(p.nx, dwell=dwell, delay=gx.rise_time, system=system)

    # Prephaser / rewinder natural durations, then a common fixed duration so
    # the TR is constant regardless of the (line-dependent) phase-encode area.
    gx_pre0 = pp.make_trapezoid("x", area=-gx.area / 2, system=system)
    gy_max = pp.make_trapezoid("y", area=(p.ny / 2) * delta_k, system=system)
    pre_dur = max(pp.calc_duration(gx_pre0), pp.calc_duration(gy_max),
                  pp.calc_duration(gzr))

    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=pre_dur, system=system)
    gzr = pp.make_trapezoid("z", area=gzr.area, duration=pre_dur, system=system)

    # Slice fully balanced: total gz = gz + gzr + gz_rew = 0.
    gz_rew_area = -(gz.area + gzr.area)
    gx_post = pp.make_trapezoid("x", area=-gx.area / 2, system=system)
    gz_rew0 = pp.make_trapezoid("z", area=gz_rew_area, system=system)
    post_dur = max(pp.calc_duration(gx_post), pp.calc_duration(gy_max),
                   pp.calc_duration(gz_rew0))
    gx_post = pp.make_trapezoid("x", area=-gx.area / 2, duration=post_dur, system=system)
    gz_rew = pp.make_trapezoid("z", area=gz_rew_area, duration=post_dur, system=system)

    # Timing: place the echo (ADC centre) at TE and pad the TR to a constant.
    dur1 = pp.calc_duration(gz)
    t_rfc = gz.rise_time + p.trf / 2                  # RF centre from block start
    t_adcc = adc.delay + adc_dur / 2                  # ADC centre from block start
    te_fill = p.te - (dur1 - t_rfc) - pre_dur - t_adcc
    dur3 = pp.calc_duration(gx, adc)
    tr_fill = p.tr - dur1 - pre_dur - max(te_fill, 0.0) - dur3 - post_dur
    if te_fill < -1e-9:
        raise ValueError(
            f"TE={p.te*1e3:.2f} ms too short for the readout: reduce bandwidth "
            f"or TBW (te_fill={te_fill*1e3:.3f} ms)."
        )
    if tr_fill < -1e-9:
        raise ValueError(
            f"TR={p.tr*1e3:.2f} ms too short for the bSSFP kernel "
            f"(tr_fill={tr_fill*1e3:.3f} ms)."
        )
    # Snap fills onto the block-duration raster to keep check_timing happy.
    raster = system.grad_raster_time
    te_fill = round(max(te_fill, 0.0) / raster) * raster
    tr_fill = round(max(tr_fill, 0.0) / raster) * raster

    return _Kernel(gz=gz, gzr=gzr, gz_rew=gz_rew, gx=gx, gx_pre=gx_pre,
                   gx_post=gx_post, adc=adc, pre_dur=pre_dur, post_dur=post_dur,
                   te_fill=te_fill, tr_fill=tr_fill, delta_k=delta_k)


def _tr_blocks(system: pp.Opts, p: BoostParams, k: _Kernel, flip_deg: float,
               rf_phase: float, pe_area: float, acquire: bool) -> Blocks:
    """One balanced bSSFP TR (echo at TE, total duration TR)."""
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(abs(flip_deg)), duration=p.trf,
        slice_thickness=p.slice_thickness, time_bw_product=p.tbw_excitation,
        phase_offset=rf_phase, return_gz=True, system=system, use="excitation",
    )
    gy_pre = pp.make_trapezoid("y", area=pe_area, duration=k.pre_dur, system=system)
    gy_rew = pp.make_trapezoid("y", area=-pe_area, duration=k.post_dur, system=system)

    blocks: Blocks = [[rf, gz], [k.gzr, k.gx_pre, gy_pre]]
    if k.te_fill > 0:
        blocks.append([pp.make_delay(k.te_fill)])
    blocks.append([k.gx, k.adc] if acquire else [k.gx])
    blocks.append([k.gx_post, gy_rew, k.gz_rew])
    if k.tr_fill > 0:
        blocks.append([pp.make_delay(k.tr_fill)])
    return blocks


def bssfp_readout(system: pp.Opts, p: BoostParams,
                  pe_lines: Sequence[int], im_flip_angle: float,
                  phase_start: int = 0, acquire: bool = True) -> Blocks:
    """A segmented bSSFP shot: iNAV start-up ramp + imaging TRs.

    Parameters
    ----------
    pe_lines
        Phase-encode line indices (0..ny-1) acquired in this segment. Its length
        is the number of imaging TRs; usually ``p.im_segments``.
    im_flip_angle
        Imaging flip angle [deg] for this contrast (bright-blood vs reference).
    phase_start
        Parity offset for the alternating (0, pi) RF phase, so segments chain
        into a continuous phase-cycled train.
    acquire
        If False the imaging TRs still play (identical magnetization evolution)
        but record no ADC -- used for the dummy heartbeats that drive the system
        to steady state.

    The first ``p.inav_lines`` TRs ramp the flip angle linearly from
    ``inav_flip_angle`` toward ``im_flip_angle`` with no acquisition (iNAV
    start-up, ``RR_sim.jl:129-135``); the remaining TRs acquire one line each.
    """
    k = _build_kernel(system, p)
    out: Blocks = []
    tr_index = phase_start
    n_ramp = p.inav_lines

    for i in range(n_ramp):
        if n_ramp > 0:
            slope = (im_flip_angle - p.inav_flip_angle) / n_ramp
            flip = min(slope * i + p.inav_flip_angle, im_flip_angle)
        else:
            flip = im_flip_angle
        rf_phase = np.pi * (tr_index % 2)
        out += _tr_blocks(system, p, k, flip, rf_phase, pe_area=0.0, acquire=False)
        tr_index += 1

    for line in pe_lines:
        pe_area = (line - p.ny / 2) * k.delta_k
        rf_phase = np.pi * (tr_index % 2)
        out += _tr_blocks(system, p, k, im_flip_angle, rf_phase, pe_area, acquire=acquire)
        tr_index += 1

    return out
