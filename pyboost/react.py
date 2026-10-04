"""REACT: flow-independent, non-gated carotid angiography at 0.55T.

Structure per shot::

    [T2-prep 50 ms] -> [non-selective IR, short TI] -> [spoiled dual-echo Dixon train]

The contrast is purely relaxation-driven, which is what makes REACT independent
of flow and free of any triggering:

* **T2-prep** keeps blood (T2 ~263 ms at 0.55T) while muscle (~55 ms) decays.
* **Non-selective inversion with a short TI** then nulls the short-T1 background.
  Blood, with a long T1, is still strongly inverted at that TI, so on magnitude
  images it is bright while fat and muscle sit near zero. This is the "STIR"
  half of the mechanism described by Pennig 2020.
* **Dual-echo Dixon** separates water from fat in reconstruction, which is what
  lets the technique hold fat suppression over a large field of view where a
  spectral fat-sat would fail.

Unlike :mod:`pyboost.boost` and :mod:`pyboost.mra` this sequence emits **no ECG
trigger**: the published neck protocol (Pennig et al., Clin Neuroradiol 2020,
fixed 2:46 min scan) is explicitly untriggered, which is the "without ... and
Triggering" in the name.

Reference
---------
Yoneyama M, Zhang S, Hu HH, et al. "Free-breathing non-contrast-enhanced
flow-independent MR angiography using magnetization-prepared 3D non-balanced
dual-echo Dixon method." Magn Reson Imaging 2019;63:137-146.
doi:10.1016/j.mri.2019.08.017
"""

from __future__ import annotations

from typing import List, Tuple

import pypulseq as pp

from .params import ReactParams
from .prep import t2_prep, inversion, fat_sat
from .readout_dixon import dixon_readout, kernel_report

Block = List
Blocks = List[Block]


def _dur(blocks: Blocks) -> float:
    return sum(pp.calc_duration(*b) for b in blocks)


def encode_order(p: ReactParams) -> List[Tuple[int, int]]:
    """Full ``(ky, kz)`` encode list in acquisition order.

    With ``centric`` ordering the encodes are sorted by distance from the centre
    of k-space. This is not cosmetic: the prepared magnetization recovers
    throughout the shot, so whatever is acquired late carries progressively less
    of the REACT contrast. :func:`shot_encodes` decides how this list is divided
    between shots.
    """
    encodes = [(ky, kz) for kz in range(p.nz) for ky in range(p.ny)]
    if p.centric:
        encodes.sort(key=lambda e: (abs(e[0] - p.ny / 2) ** 2
                                    + abs(e[1] - p.nz / 2) ** 2))
    return encodes


def shot_encodes(shot: int, p: ReactParams) -> List[Tuple[int, int]]:
    """The encodes acquired in one shot.

    With ``centric`` ordering the shots are *interleaved* through the
    centre-sorted list: shot ``s`` takes entries ``s, s+N, s+2N, ...`` (``N`` =
    ``p.n_shots``). Every shot therefore starts next to the centre of k-space and
    climbs to high spatial frequencies, so the decay of the prepared
    magnetization along the train weights k-space as a smooth function of radius.
    Handing each shot a contiguous block instead puts the centre in shot 0 only
    and makes that weighting a sawtooth that resets every ``tfe_factor`` lines.

    This is an interpretation, not a published algorithm: Gietzen 2025 describes
    a "low-high" profile order in which every shot starts close to the centre of
    k-space and gives no rule for dividing the encodes between shots, and no
    0.55T paper found describes the order of a REACT-type readout. Without
    ``centric`` the shots are contiguous slices of the raster order.

    The train is ``ceil(n_encodes / n_shots)`` lines long, which can be shorter
    than ``tfe_factor`` (120 encodes at ``tfe_factor=22`` give 6 shots of 20).
    """
    order = encode_order(p)
    if p.centric:
        return order[shot::p.n_shots]
    start = shot * p.tfe_factor
    return order[start:start + p.tfe_factor]


def build_react_sequence(p: ReactParams | None = None,
                         system: pp.Opts | None = None,
                         use_t2prep: bool = True,
                         use_inversion: bool = True,
                         use_fatsat: bool = False) -> pp.Sequence:
    """Build a REACT :class:`pypulseq.Sequence`.

    Parameters
    ----------
    use_t2prep, use_inversion
        The two REACT preparation modules. Both default on; turning either off
        is for isolating its contribution in simulation, not for scanning.
    use_fatsat
        Off by default -- REACT suppresses fat through the Dixon reconstruction,
        not a spectral pulse. Exposed as a fallback: the repo's FatSat module is
        already tuned for 0.55T (~27 ms, ~80 Hz bandwidth), so enabling it gives
        a usable single-echo-style fat suppression if the two-point water/fat
        reconstruction is not available yet.

    The TI is measured from the inversion pulse to the *start of acquisition*,
    matching the convention used in :mod:`pyboost.boost`.
    """
    from .system import scanner_055T
    if system is None:
        # The user-supplied limits for this scanner, not the library default
        # (26 mT/m, 45 T/m/s), which exceeds them and is shared with BOOST/MRA.
        system = scanner_055T(max_grad=23.0, max_slew=25.0,
                              rf_ringdown_time=20e-6)
    if p is None:
        p = ReactParams()

    seq = pp.Sequence(system=system)
    ti = p.resolved_ti()

    # The derived TI is a continuous quantity (T1*ln(1+E2)), so it almost never
    # lands on the block raster. Snap the recovery delay once, up front, and
    # report the TI actually played rather than the nominal one.
    achieved_ti = ti
    if use_inversion:
        overhead = _dur(inversion(system, post_delay=0.0, kind=p.inversion_kind))
        raster = system.block_duration_raster
        post_delay = round((ti - overhead) / raster) * raster
        if post_delay < 0:
            raise ValueError(
                f"TI={ti*1e3:.1f} ms is shorter than the inversion pulse plus its "
                f"spoiler ({overhead*1e3:.1f} ms). Raise TI, shorten the inversion, "
                f"or null a longer-T1 tissue."
            )
        achieved_ti = overhead + post_delay

    total_shots = p.dummy_shots + p.n_shots
    tr_counter = 0
    for s in range(total_shots):
        acquire = s >= p.dummy_shots
        shot = s - p.dummy_shots if acquire else 0
        encodes = shot_encodes(shot, p)
        if not encodes:
            continue

        preps: Blocks = []
        if use_t2prep:
            preps += t2_prep(system, te=p.t2prep_duration, trf=p.trf,
                             n_refocus=p.t2prep_refocus)
        if use_inversion:
            # `inversion` builds [180, spoiler, delay]; the delay it is given is
            # what remains of TI after the pulse and its spoiler, so TI ends
            # exactly when the first excitation starts.
            preps += inversion(system, post_delay=post_delay,
                               kind=p.inversion_kind)
        if use_fatsat:
            preps += fat_sat(system, flip_angle_deg=p.fatsat_flip_angle,
                             duration=p.fatsat_duration)

        readout = dixon_readout(system, p, encodes, phase_start=tr_counter,
                                acquire=acquire)
        tr_counter += len(encodes)
        blocks = preps + readout

        for b in blocks:
            seq.add_block(*b)

        # Recovery to the next preparation. REACT is untriggered, so this is a
        # plain relaxation delay, not an RR interval.
        fill = p.shot_interval - _dur(blocks)
        if fill < 0:
            raise ValueError(
                f"shot_interval={p.shot_interval*1e3:.0f} ms is too short for the "
                f"REACT prep + {len(encodes)} TRs (overshoot {-fill*1e3:.1f} ms). "
                f"Raise shot_interval or lower tfe_factor."
            )
        raster = system.block_duration_raster
        fill = round(fill / raster) * raster
        if fill > 0:
            seq.add_block(pp.make_delay(fill))

    rep = kernel_report(system, p)
    seq.set_definition("Name", "REACT_carotid_055T")
    seq.set_definition("FOV", [p.fov, p.fov,
                               p.slice_thickness * (p.nz if p.nz > 1 else 1)])
    # Traceability: the derived 0.55T values travel with the .seq file.
    seq.set_definition("ReactTE1", rep["te1"])
    seq.set_definition("ReactTE2", rep["te2"])
    seq.set_definition("ReactDeltaTE", rep["delta_te"])
    seq.set_definition("ReactTR", rep["tr"])
    seq.set_definition("ReactTI", achieved_ti)
    seq.set_definition("ReactT2prep", p.t2prep_duration)
    seq.set_definition("ReactFlipAngle", p.flip_angle)
    seq.set_definition("ReactTFEFactor", p.tfe_factor)
    return seq
