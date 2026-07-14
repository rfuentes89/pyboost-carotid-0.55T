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


def differentiable_flip_signal(seq0, img_idx: List[int], flip_deg: torch.Tensor,
                               obj, per: int) -> torch.Tensor:
    """Central-k |signal| with the imaging flip set to ``flip_deg`` (torch tensor).

    ``per`` is the number of samples in the first contrast block; the central
    k-space sample within it is the echo peak. Differentiable w.r.t. ``flip_deg``.
    """
    for i in img_idx:
        seq0[i].pulse.angle = flip_deg * (np.pi / 180.0)
    signal, kspace = mr0.util.simulate(seq0, obj)
    k = kspace.detach().cpu().numpy()[:per]
    c = int(np.argmin(np.abs(k[:, 0]) + np.abs(k[:, 1])))
    return signal.reshape(-1)[c].abs()
