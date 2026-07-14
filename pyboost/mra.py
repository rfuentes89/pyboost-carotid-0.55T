"""Bright-blood carotid MR angiography (relaxation-based) at 0.55T.

Unlike black-blood (which needs flow), bright-blood MRA here rests on *relaxation*
contrast, which the simulators reproduce faithfully:

* **T2 preparation** — blood has a long T2 (~263 ms at 0.55T) versus muscle
  (~45 ms), so after a T2-prep of echo time ``t2prep_duration`` blood keeps most
  of its signal while muscle decays. This is the dominant blood-to-background
  mechanism (CMRA / NATIVE-style).
* **bSSFP readout** — intrinsically bright-blood (signal ∝ sqrt(T2/T1)).
* **Fat saturation** — suppresses the subcutaneous / perivascular fat.

The sequence is a single bright-blood contrast per heartbeat (no dual-contrast
subtraction), otherwise sharing the ECG-gated, segmented, centric bSSFP machinery
of the BOOST build.
"""

from __future__ import annotations

import pypulseq as pp

from .params import BoostParams
from .prep import fat_sat, t2_prep, inversion
from .readout import bssfp_readout
from .boost import _dur, _shot_lines


def build_mra_sequence(p: BoostParams | None = None,
                       system: pp.Opts | None = None,
                       use_t2prep: bool = True, use_fatsat: bool = True,
                       use_inversion: bool = False,
                       add_trigger: bool = True) -> pp.Sequence:
    """Build a bright-blood MRA :class:`pypulseq.Sequence`.

    Per heartbeat: ``[inversion?] -> [T2-prep?] -> [FatSat?] -> bSSFP readout`` at
    ``im_flip_angle[0]``. ``use_inversion`` is off by default (an inversion
    darkens blood, which we want bright); it is exposed for background-nulling
    experiments. ``dummy_heart_beats`` drive the steady state; the following
    ``n_shots`` heartbeats fill k-space in centric segments.
    """
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

        preps = []
        if use_inversion:
            ti_delay = p.ir_inversion_time - p.inav_lines * p.tr - p.trf - p.te
            preps += inversion(system, post_delay=ti_delay, kind=p.inversion_kind)
        if use_t2prep:
            preps += t2_prep(system, te=p.t2prep_duration, trf=p.trf)
        if use_fatsat:
            preps += fat_sat(system, flip_angle_deg=p.fatsat_flip_angle,
                             duration=p.fatsat_duration)
        readout = bssfp_readout(system, p, lines, p.im_flip_angle[0],
                                acquire=acquire)
        blocks = preps + readout

        if trig is not None:
            seq.add_block(trig)
        for b in blocks:
            seq.add_block(*b)

        rr_fill = p.rr - _dur(blocks) - (pp.calc_duration(trig) if trig else 0.0)
        if rr_fill < 0:
            raise ValueError(
                f"RR={p.rr*1e3:.0f} ms too short for the MRA prep+readout "
                f"(overshoot {-rr_fill*1e3:.1f} ms)."
            )
        raster = system.block_duration_raster
        rr_fill = round(rr_fill / raster) * raster
        if rr_fill > 0:
            seq.add_block(pp.make_delay(rr_fill))

    seq.set_definition("Name", "MRA_carotid_055T")
    seq.set_definition("FOV", [p.fov, p.fov, p.slice_thickness])
    return seq


from .system import scanner_055T  # noqa: E402  (avoid a cycle at import time)
