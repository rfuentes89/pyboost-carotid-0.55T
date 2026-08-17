"""Approximate blood-flow (flow-void) modelling for the carotid simulation.

Why this exists. Black-blood vessel-wall contrast comes from **flow**: blood
leaves the imaging slice between the preparation and the readout, so the lumen is
filled with *fresh* spins that never experienced the preparation. Neither the
MRzero PDG nor the static Koma reference models moving spins, which is why a
static phantom gives lumen ~ wall (blood and wall T1 are too close for inversion
to separate them).

The model here is the **complete through-plane washout limit**: during the
readout the lumen holds fully-relaxed blood, while the stationary tissues carry
the full preparation history. It is implemented with two simulations that share
one k-space trajectory, so their signals simply add:

1. **static compartment** -- everything except blood, simulated with the *full*
   sequence (preparations included);
2. **blood compartment** -- blood only, simulated with a *preparation-stripped*
   copy of the same sequence: every preparation RF pulse has its flip angle set
   to zero while all delays, gradients and ADCs stay untouched. With no RF to tip
   it, blood magnetization simply stays at equilibrium through the prep interval
   -- exactly "blood that flowed in after the preparation".

Because the two runs share timing and gradients, ``signal_static + signal_blood``
is the k-space of the flowing phantom, and it reconstructs with the very same
reconstruction used for the static case.

This is an approximation (the fast-flow limit; no velocity grading, no pulsatility
and no in-plane flow). The rigorous alternative is KomaMRI's ``FlowPath`` /
``spin_reset`` motion model.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .params import BoostParams
from .phantom import LABELS, to_mrzero_phantom


def compartment_maps(maps: Dict[str, np.ndarray],
                     keep: Iterable[str]) -> Dict[str, np.ndarray]:
    """Copy of ``maps`` with every tissue outside ``keep`` set to PD = 0.

    ``to_mrzero_phantom`` drops PD = 0 voxels, so this splits the phantom into
    compartments that can be simulated separately and added back together.
    """
    keep_labels = {LABELS[name] for name in keep}
    out = {k: v.copy() for k, v in maps.items()}
    mask = np.isin(maps["label"], list(keep_labels))
    out["PD"] = np.where(mask, maps["PD"], 0.0)
    return out


def split_phantoms(maps: Dict[str, np.ndarray], fov: float):
    """Return ``(static_phantom, blood_phantom)`` for the flow model."""
    static = [n for n in LABELS if n not in ("air", "blood")]
    return (to_mrzero_phantom(compartment_maps(maps, static), fov),
            to_mrzero_phantom(compartment_maps(maps, ["blood"]), fov))


def locate_imaging_reps(seq0, p: BoostParams) -> List[int]:
    """Indices of the bSSFP repetitions (iNAV ramp + imaging TRs).

    The iNAV ramp always starts at ``p.inav_flip_angle`` (3.2 deg by default),
    which no preparation pulse uses, so each ramp start is an unambiguous
    landmark; the block that follows it spans ``inav_lines + im_segments``
    repetitions.
    """
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])
    span = p.inav_lines + p.im_segments
    idx: List[int] = []
    for start in np.where(np.abs(angles - p.inav_flip_angle) < 0.5)[0]:
        idx.extend(range(int(start), min(int(start) + span, len(seq0))))
    return sorted(set(idx))


def locate_prep_pulses(seq0, p: BoostParams) -> List[int]:
    """Indices of the preparation pulses (T2-prep, inversion, FatSat)."""
    imaging = set(locate_imaging_reps(seq0, p))
    return [i for i in range(len(seq0)) if i not in imaging]


def strip_preparation(seq0, p: BoostParams) -> List[int]:
    """Zero the flip angle of every preparation pulse, **in place**.

    Timing, gradients and ADC events are untouched, so the k-space trajectory is
    identical to the unstripped sequence -- only the preparation's effect on the
    magnetization disappears. Returns the indices that were zeroed.
    """
    import torch

    prep_idx = locate_prep_pulses(seq0, p)
    for i in prep_idx:
        seq0[i].pulse.angle = torch.zeros_like(
            torch.as_tensor(seq0[i].pulse.angle, dtype=torch.float32))
    return prep_idx


def simulate_with_flow(seq_path: str, maps: Dict[str, np.ndarray], fov: float,
                       p: BoostParams) -> Tuple[object, object, object, object]:
    """Simulate the flowing phantom in the complete-washout limit.

    Imports ``seq_path`` twice (once stripped), simulates the static tissues and
    the fresh blood separately and adds their signals.

    Returns ``(signal_flow, kspace, signal_static, signal_blood)`` so callers can
    also reconstruct the individual compartments.
    """
    import MRzeroCore as mr0

    static_obj, blood_obj = split_phantoms(maps, fov)

    seq_static = mr0.Sequence.import_file(seq_path)
    signal_static, kspace = mr0.util.simulate(seq_static, static_obj)

    seq_blood = mr0.Sequence.import_file(seq_path)
    stripped = strip_preparation(seq_blood, p)
    if not stripped:
        raise ValueError("no preparation pulses found to strip -- check params")
    signal_blood, _ = mr0.util.simulate(seq_blood, blood_obj)

    return signal_static + signal_blood, kspace, signal_static, signal_blood
