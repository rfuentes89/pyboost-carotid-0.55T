"""Differentiate through MRzero w.r.t. the imaging flip angle.

MRzero's simulation is differentiable; the only barrier is our pypulseq ``.seq``
round-trip, which freezes the pulse angles as constants on import. The trick here
is to import the sequence (so all gradients, timing and spoilers are correct),
then *replace the imaging pulses' angle with a torch tensor*. ``mr0.util.simulate``
then returns a signal that is differentiable w.r.t. the flip angle, so it can be
optimized by gradient descent through the real simulator -- no surrogate model.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

import MRzeroCore as mr0

from .params import BoostParams
from .mra import build_mra_sequence


def import_mra_for_optimization(p: BoostParams, system, base_flip_deg: float = 90.0,
                                path: str = "/tmp/mra_opt_diff.seq"
                                ) -> Tuple[object, List[int]]:
    """Build + import an MRA sequence and locate its imaging pulses.

    Returns the imported ``mr0.Sequence`` and the indices of the bSSFP *imaging*
    repetitions (the constant-flip run), whose angle we make differentiable. The
    T2-prep 90 deg pulses are isolated (bracketed by the 180) and so are excluded
    from the run of >= 3 equal angles.
    """
    seq = build_mra_sequence(p, system, use_t2prep=True, use_fatsat=False,
                             add_trigger=False)
    seq.write(path)
    seq0 = mr0.Sequence.import_file(path)
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])

    idx: List[int] = []
    i = 0
    while i < len(seq0):
        if abs(angles[i] - base_flip_deg) < 0.5:
            j = i
            while j + 1 < len(seq0) and abs(angles[j + 1] - base_flip_deg) < 0.5:
                j += 1
            if j - i + 1 >= 3:                       # a real imaging run
                idx.extend(range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return seq0, idx


def set_imaging_flip(seq0, img_idx: List[int], flip_deg: torch.Tensor) -> None:
    """Set every imaging pulse's angle to ``flip_deg`` (differentiable)."""
    for i in img_idx:
        seq0[i].pulse.angle = flip_deg * (np.pi / 180.0)


def locate_inav_ramp(seq0, inav_flip_deg: float = 3.2, base_flip_deg: float = 90.0
                     ) -> Tuple[List[int], dict]:
    """iNAV start-up ramp reps (angles strictly between the ramp start and the
    imaging flip). Returns their indices and base angles [deg]."""
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])
    idx = [i for i in range(len(seq0))
           if inav_flip_deg - 0.5 <= angles[i] < base_flip_deg - 0.5]
    return idx, {i: float(angles[i]) for i in idx}


def set_imaging_flip_coupled(seq0, img_idx: List[int], ramp_idx: List[int],
                             base_ramp: dict, flip_deg: torch.Tensor,
                             inav_flip_deg: float = 3.2,
                             base_flip_deg: float = 90.0) -> None:
    """Set the imaging flip AND scale the iNAV ramp to end at that flip.

    The ramp keeps its shape but its span tracks the flip:
    ``angle_i = inav + (base_i - inav) * (flip - inav) / (base_flip - inav)``,
    differentiable w.r.t. ``flip_deg``.
    """
    for i in img_idx:
        seq0[i].pulse.angle = flip_deg * (np.pi / 180.0)
    span = (flip_deg - inav_flip_deg) / (base_flip_deg - inav_flip_deg)
    for i in ramp_idx:
        angle = inav_flip_deg + (base_ramp[i] - inav_flip_deg) * span
        seq0[i].pulse.angle = angle * (np.pi / 180.0)


def locate_t2prep_delays(seq0) -> Tuple[List[int], dict]:
    """Reps carrying the T2-prep TE/2 delays: each 180 deg pulse and the 90 deg
    before it. Returns their indices and a snapshot of their base ``event_time``
    (which is scaled to change TE)."""
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])
    te_idx: List[int] = []
    for j in range(len(seq0)):
        if abs(angles[j] - 180.0) < 1.0:
            te_idx += [j - 1, j]
    te_idx = sorted(set(i for i in te_idx if i >= 0))
    base_et = {i: seq0[i].event_time.detach().clone() for i in te_idx}
    return te_idx, base_et


def set_t2prep_te(seq0, te_idx: List[int], base_et: dict, te: torch.Tensor,
                  base_te: float = 0.06) -> None:
    """Scale the T2-prep delay events so the echo time equals ``te`` (torch
    tensor). MRzero relaxes over ``event_time``, so the signal is differentiable
    w.r.t. ``te``."""
    for i in te_idx:
        seq0[i].event_time = base_et[i] * (te / base_te)


def central_signal(seq0, obj, per: int) -> torch.Tensor:
    """Central-k |signal| of the first contrast block (the echo peak)."""
    signal, kspace = mr0.util.simulate(seq0, obj)
    k = kspace.detach().cpu().numpy()[:per]
    c = int(np.argmin(np.abs(k[:, 0]) + np.abs(k[:, 1])))
    return signal.reshape(-1)[c].abs()


def differentiable_flip_signal(seq0, img_idx: List[int], flip_deg: torch.Tensor,
                               obj, per: int) -> torch.Tensor:
    """Convenience: set the imaging flip and return the central |signal|.

    Differentiable w.r.t. ``flip_deg`` (see :func:`set_imaging_flip`).
    """
    set_imaging_flip(seq0, img_idx, flip_deg)
    return central_signal(seq0, obj, per)
