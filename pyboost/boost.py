"""Assemble the full BOOST carotid sequence (``RR_sim.jl:150-194``).

Structure per heartbeat, for the two interleaved contrasts:

* contrast 0 (bright-blood): T2-prep -> adiabatic IR (TI) -> bSSFP readout at
  ``im_flip_angle[0]``;
* contrast 1 (reference):     FatSat -> bSSFP readout at ``im_flip_angle[1]``.

The black-blood image is the phase-sensitive subtraction of the two contrasts,
formed in post-processing (see ``scripts/validate_with_mrzero.py``).

``dummy_heart_beats`` heartbeats run the identical structure without recording
ADC to reach steady state; the following ``n_shots`` heartbeats fill k-space in
interleaved phase-encode segments. Both contrasts sample the *same* lines in a
shot so they can be subtracted.
"""

from __future__ import annotations

from typing import List

import pypulseq as pp

from .params import BoostParams
from .prep import fat_sat, t2_prep, inversion
from .readout import bssfp_readout

Block = List
Blocks = List[Block]


def _dur(blocks: Blocks) -> float:
    return sum(pp.calc_duration(*b) for b in blocks)


def _shot_lines(shot: int, p: BoostParams) -> List[int]:
    """Phase-encode lines for one shot (interleaved shots cover 0..ny-1).

    With ``centric`` ordering the lines within a shot are played nearest-to-DC
    first, so the centre of k-space is sampled right after the preparation while
    the black-blood contrast is still fresh (before the bSSFP transient washes it
    out). Otherwise lines are played in ascending (linear) order.
    """
    lines = list(range(shot, p.ny, p.n_shots))
    if p.centric:
        lines.sort(key=lambda i: abs(i - p.ny / 2))
    return lines


def _contrast_blocks(system: pp.Opts, p: BoostParams, contrast: int,
                     lines: List[int], acquire: bool) -> Blocks:
    """Prep + readout for one contrast within a heartbeat."""
    if contrast == 0:  # bright-blood: T2-prep + inversion
        preps = t2_prep(system, te=p.t2prep_duration, trf=p.trf)
        # TI measured from the inversion to the start of imaging, mirroring
        # RR_sim.jl:170: IR_inversion_time - inav_lines*TR - Trf - TE.
        ti_delay = p.ir_inversion_time - p.inav_lines * p.tr - p.trf - p.te
        preps += inversion(system, post_delay=ti_delay, kind=p.inversion_kind)
    else:              # reference: fat saturation only
        preps = fat_sat(system, flip_angle_deg=p.fatsat_flip_angle,
                        duration=p.fatsat_duration)
    readout = bssfp_readout(system, p, lines, p.im_flip_angle[contrast],
                            acquire=acquire)
    return preps + readout


def build_boost_sequence(p: BoostParams | None = None,
                         system: pp.Opts | None = None,
                         add_trigger: bool = True) -> pp.Sequence:
    """Build and return the full BOOST :class:`pypulseq.Sequence`.

    ``add_trigger`` emits an ECG (``physio1``) trigger marker at the start of
    each heartbeat. NOTE: this is a trigger *output* marker documenting the gated
    structure; true prospective cardiac gating (trigger *wait*) is vendor
    specific and must be enabled on the scanner. The fixed ``RR`` here models the
    nominal cardiac cycle.
    """
    from .system import scanner_055T
    if system is None:
        system = scanner_055T()
    if p is None:
        p = BoostParams()

    seq = pp.Sequence(system=system)
    trig = pp.make_trigger("physio1", duration=system.grad_raster_time,
                           system=system) if add_trigger else None

    total_hb = p.dummy_heart_beats + p.n_shots
    for hb in range(total_hb):
        acquire = hb >= p.dummy_heart_beats
        shot = hb - p.dummy_heart_beats if acquire else 0
        lines = _shot_lines(shot, p)
        for contrast in (0, 1):
            if trig is not None:
                seq.add_block(trig)
            blocks = _contrast_blocks(system, p, contrast, lines, acquire)
            for b in blocks:
                seq.add_block(*b)
            # Fill the rest of the RR interval (RR_sim.jl:189-190).
            rr_fill = p.rr - _dur(blocks) - pp.calc_duration(trig) if trig \
                else p.rr - _dur(blocks)
            if rr_fill < 0:
                raise ValueError(
                    f"RR={p.rr*1e3:.0f} ms too short for contrast {contrast} "
                    f"(overshoot {-rr_fill*1e3:.1f} ms). Shorten prep/readout or "
                    "raise RR."
                )
            # Snap onto the block raster to keep check_timing clean.
            raster = system.block_duration_raster
            rr_fill = round(rr_fill / raster) * raster
            if rr_fill > 0:
                seq.add_block(pp.make_delay(rr_fill))

    seq.set_definition("Name", "BOOST_carotid_055T")
    seq.set_definition("FOV", [p.fov, p.fov, p.slice_thickness])
    return seq
