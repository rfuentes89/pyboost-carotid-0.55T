"""Spoiled dual-echo Dixon readout for REACT at 0.55T.

This is the one block REACT cannot borrow from BOOST. :mod:`pyboost.readout`
builds a *balanced* bSSFP train; REACT needs the opposite -- Yoneyama's original
description is a "3D non-balanced dual-echo Dixon" readout, i.e. a spoiled
gradient echo that samples two echoes per excitation so water and fat can be
separated in reconstruction.

Two monopolar readouts per TR
-----------------------------
Both echoes are read with the *same* gradient polarity, separated by a full-area
rewinder that walks k-space from ``+kmax`` back to ``-kmax``. A bipolar pair
would save the rewinder (~1.6 ms), but its opposite-polarity eddy-current and
phase errors are precisely what corrupts two-point water/fat separation, and at
0.55T the echo spacing is long enough that the saving buys nothing. See
:func:`dixon_echo_spacing` for why.

Why the low field changes everything
------------------------------------
The fat-water shift scales with B0: about -80 Hz at 0.55T against -220 Hz at
1.5T. Every echo-spacing number in the REACT literature is therefore unusable
here and has to be re-derived; that derivation lives in
:func:`dixon_echo_spacing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pypulseq as pp

from .params import ReactParams
from .system import FAT_PPM, fat_frequency

Block = List
Blocks = List[Block]


def dixon_echo_spacing(system: pp.Opts, phase_deg: float = 162.6,
                       fat_ppm: float = FAT_PPM) -> float:
    """Water-fat echo spacing [s] giving ``phase_deg`` of relative phase.

    REACT does not sit at exact opposed phase -- Yoneyama et al. describe
    "semi-flexible echo times". The published 1.5T protocol (Isaak 2021, JCMR,
    Table 1: TE1 1.72 ms, TE2 3.80 ms) has a spacing of 2.08 ms, and at 1.5T the
    shift is 3.4 ppm * 63.87 MHz = 217 Hz, so that spacing corresponds to::

        360 * 217 Hz * 2.08 ms = 162.6 deg

    Holding that *phase* fixed rather than the *time* is what transfers the
    protocol across field strengths. At 0.55T (about 80 Hz) the same 162.6 deg
    needs ~5.7 ms -- a factor 2.7 longer than at 1.5T, which is the single
    dominant constraint on the whole sequence.

    For reference, exact opposed phase (180 deg) would be 6.3 ms at 0.55T.
    """
    df = abs(fat_frequency(system, fat_ppm))
    if df <= 0:
        raise ValueError("fat frequency offset is zero; check system.B0")
    return (phase_deg / 360.0) / df


@dataclass
class _DixonKernel:
    """Line-independent events and fill delays for one dual-echo TR."""
    gz: object            # slice/slab-select gradient
    gzr: object           # slice rephaser (played in the prephaser block)
    gx: object            # readout gradient (identical for both echoes)
    gx_pre: object        # readout prephaser, area = -gx.area/2
    gx_rew: object        # inter-echo rewinder, area = -gx.area
    gx_spoil: object      # end-of-TR spoiler on the readout axis
    adc: object
    pre_dur: float        # fixed duration of the prephaser block
    post_dur: float       # fixed duration of the spoiler/rewind block
    te_fill: float        # delay before echo 1 so it lands at TE1
    dixon_fill: float     # delay between the echoes so the spacing is exactly delta_te
    tr_fill: float        # trailing delay so the block totals TR
    delta_ky: float
    delta_kz: float
    te1: float            # achieved TE1 [s]
    delta_te: float       # achieved echo spacing [s]
    tr: float             # achieved TR [s]


def _rf_spoil_phase(n: int, increment_deg: float) -> float:
    """Quadratic RF spoiling phase [rad] for excitation ``n``.

    ``phi_n = increment/2 * (n^2 + n + 2)`` (Zur et al.). The ADC must carry the
    same phase so the receiver demodulates coherently.
    """
    return np.deg2rad((increment_deg / 2.0 * (n * n + n + 2)) % 360.0)


def _build_kernel(system: pp.Opts, p: ReactParams) -> _DixonKernel:
    delta_ky = 1.0 / p.fov
    # For a 3D slab the partition direction is encoded over the slab thickness.
    delta_kz = (1.0 / p.slice_thickness) if p.nz > 1 else 0.0

    target_dte = p.delta_te if p.delta_te is not None else \
        dixon_echo_spacing(system, p.dixon_phase_deg)

    # ADC dwell on a 2.5 us grid (25x the ADC raster), same trick as
    # readout.py: for nx % 4 == 0 the readout duration stays on the 10 us
    # gradient raster so gradient and ADC line up and check_timing stays clean.
    dwell_grid = 25 * system.adc_raster_time
    dwell = round((1.0 / (p.readout_bandwidth * p.nx)) / dwell_grid) * dwell_grid
    dwell = max(dwell, dwell_grid)
    adc_dur = p.nx * dwell

    # Excitation: slab-selective sinc. Even for nz == 1 this is slice-selective,
    # matching readout.py, so the 2D phantom pipeline keeps working.
    _, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(p.flip_angle), duration=p.trf,
        slice_thickness=p.slice_thickness, time_bw_product=p.tbw_excitation,
        return_gz=True, system=system, use="excitation",
    )

    gx = pp.make_trapezoid("x", flat_area=p.nx * delta_ky, flat_time=adc_dur,
                           system=system)
    adc = pp.make_adc(p.nx, dwell=dwell, delay=gx.rise_time, system=system)

    # Monopolar inter-echo rewinder: undo the whole readout traversal so echo 2
    # starts from -kmax again, exactly like echo 1.
    gx_rew = pp.make_trapezoid("x", area=-gx.area, system=system)

    # The echo spacing we can actually achieve, echo centre to echo centre.
    # Because rise_time == fall_time the ADC centre coincides with the centre of
    # the trapezoid, so each echo sits at duration(gx)/2 within its own block.
    natural_dte = pp.calc_duration(gx) + pp.calc_duration(gx_rew)
    dixon_fill = target_dte - natural_dte
    if dixon_fill < -1e-9:
        max_bw = 1.0 / max(adc_dur + dixon_fill, 1e-6)
        raise ValueError(
            f"Dixon spacing {target_dte*1e3:.2f} ms is shorter than the readout "
            f"pair needs ({natural_dte*1e3:.2f} ms) at "
            f"{p.readout_bandwidth:.0f} Hz/pixel. The readout is too long: raise "
            f"readout_bandwidth above ~{max_bw:.0f} Hz/pixel, lower nx, or widen "
            f"the slew rate."
        )

    # Prephaser block: readout prephaser + slice rephaser + phase encodes, all
    # padded to one fixed duration so TR does not depend on the encode index.
    gx_pre0 = pp.make_trapezoid("x", area=-gx.area / 2, system=system)
    gy_max = pp.make_trapezoid("y", area=(p.ny / 2) * delta_ky, system=system)
    pre_candidates = [pp.calc_duration(gx_pre0), pp.calc_duration(gy_max),
                      pp.calc_duration(gzr)]
    if p.nz > 1:
        # In 3D the partition encode is folded into the *same* z gradient as the
        # slice rephaser, so the block must fit their combined worst-case area,
        # not whichever of the two is larger on its own.
        gz_worst = abs(gzr.area) + (p.nz / 2) * delta_kz
        pre_candidates.append(
            pp.calc_duration(pp.make_trapezoid("z", area=gz_worst, system=system)))
    pre_dur = max(pre_candidates)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=pre_dur, system=system)
    gzr = pp.make_trapezoid("z", area=gzr.area, duration=pre_dur, system=system)

    # End-of-TR spoiler on the readout axis. `spoiler_cycles` counts full phase
    # wraps across one voxel; the leftover +gx.area/2 from echo 2 already
    # contributes, so only the remainder is played.
    voxel_x = p.fov / p.nx
    spoil_area = p.spoiler_cycles / voxel_x
    gx_spoil = pp.make_trapezoid("x", area=max(spoil_area - gx.area / 2, 0.0),
                                 system=system)
    post_candidates = [pp.calc_duration(gx_spoil), pp.calc_duration(gy_max)]
    if p.nz > 1:
        post_candidates.append(
            pp.calc_duration(pp.make_trapezoid("z", area=(p.nz / 2) * delta_kz,
                                               system=system)))
    post_dur = max(post_candidates)
    gx_spoil = pp.make_trapezoid("x", area=max(spoil_area - gx.area / 2, 0.0),
                                 duration=post_dur, system=system)

    raster = system.grad_raster_time
    dixon_fill = round(max(dixon_fill, 0.0) / raster) * raster

    # TE1 is measured from the RF centre to the centre of echo 1.
    dur_rf = pp.calc_duration(gz)
    t_rf_centre = gz.rise_time + p.trf / 2
    te1_min = (dur_rf - t_rf_centre) + pre_dur + pp.calc_duration(gx) / 2
    if p.te1 is None:
        te_fill = 0.0
        te1 = te1_min
    else:
        te_fill = p.te1 - te1_min
        if te_fill < -1e-9:
            raise ValueError(
                f"TE1={p.te1*1e3:.2f} ms is shorter than the minimum achievable "
                f"{te1_min*1e3:.2f} ms. Raise readout_bandwidth or shorten trf."
            )
        te_fill = round(te_fill / raster) * raster
        te1 = te1_min + te_fill

    # Recompute the true spacing after rasterisation, then the TR.
    delta_te = natural_dte + dixon_fill
    tr_min = (dur_rf + pre_dur + te_fill + pp.calc_duration(gx)
              + pp.calc_duration(gx_rew) + dixon_fill + pp.calc_duration(gx)
              + post_dur)
    if p.tr is None:
        tr_fill = 0.0
        tr = tr_min
    else:
        tr_fill = p.tr - tr_min
        if tr_fill < -1e-9:
            raise ValueError(
                f"TR={p.tr*1e3:.2f} ms is shorter than the minimum achievable "
                f"{tr_min*1e3:.2f} ms for this dual-echo kernel."
            )
        tr_fill = round(tr_fill / raster) * raster
        tr = tr_min + tr_fill

    return _DixonKernel(gz=gz, gzr=gzr, gx=gx, gx_pre=gx_pre, gx_rew=gx_rew,
                        gx_spoil=gx_spoil, adc=adc, pre_dur=pre_dur,
                        post_dur=post_dur, te_fill=te_fill, dixon_fill=dixon_fill,
                        tr_fill=tr_fill, delta_ky=delta_ky, delta_kz=delta_kz,
                        te1=te1, delta_te=delta_te, tr=tr)


def _tr_blocks(system: pp.Opts, p: ReactParams, k: _DixonKernel,
               ky: int, kz: int, tr_index: int, acquire: bool) -> Blocks:
    """One spoiled dual-echo TR: excitation, echo 1, rewinder, echo 2, spoiler."""
    phase = _rf_spoil_phase(tr_index, p.rf_spoil_increment)
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(p.flip_angle), duration=p.trf,
        slice_thickness=p.slice_thickness, time_bw_product=p.tbw_excitation,
        phase_offset=phase, return_gz=True, system=system, use="excitation",
    )
    pe_y = (ky - p.ny / 2) * k.delta_ky
    gy_pre = pp.make_trapezoid("y", area=pe_y, duration=k.pre_dur, system=system)
    gy_rew = pp.make_trapezoid("y", area=-pe_y, duration=k.post_dur, system=system)

    pre_events = [k.gzr, k.gx_pre, gy_pre]
    post_events = [k.gx_spoil, gy_rew]
    if p.nz > 1:
        pe_z = (kz - p.nz / 2) * k.delta_kz
        # The partition encode rides on the slice axis together with the slice
        # rephaser, so it is folded into a single gradient rather than stacked.
        gz_pre = pp.make_trapezoid("z", area=k.gzr.area + pe_z,
                                   duration=k.pre_dur, system=system)
        pre_events = [gz_pre, k.gx_pre, gy_pre]
        post_events.append(
            pp.make_trapezoid("z", area=-pe_z, duration=k.post_dur, system=system))

    # The ADC carries the RF spoiling phase so the receiver stays coherent.
    adc1 = pp.make_adc(p.nx, dwell=k.adc.dwell, delay=k.gx.rise_time,
                       phase_offset=phase, system=system)
    adc2 = pp.make_adc(p.nx, dwell=k.adc.dwell, delay=k.gx.rise_time,
                       phase_offset=phase, system=system)

    blocks: Blocks = [[rf, gz], pre_events]
    if k.te_fill > 0:
        blocks.append([pp.make_delay(k.te_fill)])
    blocks.append([k.gx, adc1] if acquire else [k.gx])
    blocks.append([k.gx_rew])
    if k.dixon_fill > 0:
        blocks.append([pp.make_delay(k.dixon_fill)])
    blocks.append([k.gx, adc2] if acquire else [k.gx])
    blocks.append(post_events)
    if k.tr_fill > 0:
        blocks.append([pp.make_delay(k.tr_fill)])
    return blocks


def dixon_readout(system: pp.Opts, p: ReactParams,
                  encodes: Sequence[Tuple[int, int]],
                  phase_start: int = 0, acquire: bool = True) -> Blocks:
    """A REACT shot: ``len(encodes)`` spoiled dual-echo TRs.

    Parameters
    ----------
    encodes
        ``(ky, kz)`` index pairs acquired in this shot. For a 2D acquisition
        (``nz == 1``) the ``kz`` entry is ignored.
    phase_start
        Running excitation counter so RF spoiling phase continues across shots
        instead of restarting (which would break the spoiling).
    acquire
        If False the TRs play identically but record no ADC -- used for the
        dummy shots that drive the magnetization to steady state.

    Unlike the bSSFP readout there is no flip-angle ramp: a spoiled train has no
    banding transient to catch, and REACT's low 15 deg flip barely perturbs Mz,
    which is exactly what preserves the prepared contrast through the shot.
    """
    k = _build_kernel(system, p)
    out: Blocks = []
    for i, (ky, kz) in enumerate(encodes):
        out += _tr_blocks(system, p, k, ky, kz, phase_start + i, acquire)
    return out


def kernel_report(system: pp.Opts, p: ReactParams) -> dict:
    """Achieved timing of the dual-echo kernel, for tests and for the writer."""
    k = _build_kernel(system, p)
    df = abs(fat_frequency(system))
    return {
        "te1": k.te1,
        "te2": k.te1 + k.delta_te,
        "delta_te": k.delta_te,
        "tr": k.tr,
        "adc_duration": p.nx * k.adc.dwell,
        "bandwidth_hz_px": 1.0 / (p.nx * k.adc.dwell),
        "fat_freq_hz": -df,
        "water_fat_phase_deg": 360.0 * df * k.delta_te,
        "chemical_shift_px": df * p.nx * k.adc.dwell,
        "dixon_fill": k.dixon_fill,
        "shot_duration": len(range(p.tfe_factor)) * k.tr,
    }
