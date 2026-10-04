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


def locate_fatsat(seq0) -> List[int]:
    """FatSat pulse reps -- the only ones with a nonzero RF frequency offset
    (the spectrally-selective pulse sits on the fat resonance). Their angle is
    made differentiable to optimize fat suppression."""
    return [i for i in range(len(seq0))
            if abs(float(seq0[i].pulse.freq_offset)) > 1.0]


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


def locate_react_inversion(seq0, imaging_flip_deg: float = 15.0
                           ) -> Tuple[List[int], dict]:
    """REACT's inversion repetitions -- the ones carrying the TI delay.

    A REACT shot contains *two* 180 deg pulses: the T2-prep refocusing pulse and
    the non-selective inversion. They are told apart by what follows them -- the
    inversion is the one immediately preceding the imaging train, whereas the
    T2-prep's 180 is followed by the tip-up 90.

    Returns the indices and, per index, ``(base_event_time, delay_position)``
    where ``delay_position`` is the entry holding the TI delay (the longest one).
    """
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])
    idx = [i for i in range(len(seq0) - 1)
           if abs(angles[i] - 180.0) < 1.0
           and abs(angles[i + 1] - imaging_flip_deg) < 0.5]
    info = {}
    for i in idx:
        et = seq0[i].event_time.detach().clone()
        info[i] = (et, int(torch.argmax(et)))
    return idx, info


def set_react_ti(seq0, inv_idx: List[int], info: dict, ti: torch.Tensor) -> None:
    """Set REACT's inversion time to ``ti`` (a torch tensor), differentiably.

    Only the TI delay event is rescaled, not the whole repetition: the inversion
    pulse and its spoiler have fixed durations, so scaling them too would make
    the achieved TI drift from the requested one. The delay absorbs the whole
    difference, which keeps ``ti`` exact and the gradient clean.
    """
    for i in inv_idx:
        base_et, pos = info[i]
        others = base_et.sum() - base_et[pos]
        # The delay is what remains of TI after the inversion pulse and its
        # spoiler. Below that it would go negative -- relaxation running
        # backwards -- and MRzero would simulate it without complaint, so refuse.
        if float(ti) < float(others):
            raise ValueError(
                f"TI = {float(ti)*1e3:.1f} ms is shorter than the inversion pulse "
                f"plus its spoiler ({float(others)*1e3:.1f} ms): the recovery delay "
                f"would be negative.")
        # Rebuild by concatenation rather than in-place assignment so autograd
        # keeps a clean path from `ti` to the delay event.
        et = torch.cat([base_et[:pos], (ti - others).reshape(1),
                        base_et[pos + 1:]])
        seq0[i].event_time = et


def import_react_for_optimization(p, system, path: str = "/tmp/react_opt_diff.seq"
                                  ) -> Tuple[object, List[int], dict]:
    """Build + import a REACT sequence and locate its inversion repetitions.

    The inversion is forced to ``block``: MRzero's PDG model cannot reproduce an
    adiabatic frequency sweep (it would saturate rather than invert), so TI
    optimization has to run on the hard-pulse variant. The optimum transfers --
    an adiabatic inversion inverts more robustly, not differently.
    """
    from .react import build_react_sequence
    from dataclasses import replace
    p = replace(p, inversion_kind="block")
    seq = build_react_sequence(p, system)
    seq.write(path)
    seq0 = mr0.Sequence.import_file(path)
    inv_idx, info = locate_react_inversion(seq0, p.flip_angle)
    if not inv_idx:
        raise RuntimeError("no REACT inversion repetition found; check the "
                           "sequence structure or the imaging flip angle")
    return seq0, inv_idx, info


def locate_react_recovery(seq0, imaging_flip_deg: float = 15.0
                          ) -> Tuple[List[int], dict, float]:
    """REACT's inter-shot recovery repetitions, and the shot's fixed overhead.

    REACT is untriggered, so the time between preparations is a free parameter
    rather than an RR interval. Pypulseq emits that recovery as a trailing delay
    block, which MRzero folds into the *last imaging repetition of each shot* --
    identified here as an imaging pulse whose successor is the next shot's
    T2-prep 90 deg. The final shot has no trailing delay and is skipped.

    Returns ``(indices, info, shot_overhead)`` where ``info[i]`` is
    ``(base_event_time, delay_position)`` and ``shot_overhead`` is everything in
    one shot except that delay (prep modules plus the imaging train), in
    seconds. ``shot_interval = shot_overhead + recovery_delay``.
    """
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])
    idx = [i for i in range(len(seq0) - 1)
           if abs(angles[i] - imaging_flip_deg) < 0.5
           and abs(angles[i + 1] - 90.0) < 1.0]
    info = {}
    for i in idx:
        et = seq0[i].event_time.detach().clone()
        info[i] = (et, int(torch.argmax(et)))
    if not idx:
        return idx, info, 0.0
    first = idx[0]
    base_et, pos = info[first]
    overhead = sum(float(seq0[j].event_time.sum()) for j in range(first + 1))
    overhead -= float(base_et[pos])
    return idx, info, overhead


def set_react_shot_interval(seq0, rec_idx: List[int], info: dict,
                            shot_overhead: float,
                            shot_interval: torch.Tensor) -> None:
    """Set the preparation-to-preparation interval, differentiably.

    The recovery delay absorbs the difference, so ``shot_interval`` is exactly
    what plays. Raises if the requested interval is shorter than the shot's own
    fixed overhead, which would otherwise silently produce a negative delay.
    """
    slack = shot_interval - shot_overhead
    if float(slack) < 0:
        raise ValueError(
            f"shot_interval={float(shot_interval)*1e3:.0f} ms is shorter than the "
            f"shot itself ({shot_overhead*1e3:.0f} ms of prep + readout)."
        )
    for i in rec_idx:
        base_et, pos = info[i]
        et = torch.cat([base_et[:pos], slack.reshape(1), base_et[pos + 1:]])
        seq0[i].event_time = et


def react_dc_signal(seq0, obj, nx: int) -> torch.Tensor:
    """Complex signal at the true k-space centre of echo 1, differentiable.

    Echo 1 is what carries the REACT contrast; echo 2 exists to encode fat. The
    DC point is taken globally rather than per-TR because with centric ordering
    it is acquired first, while the prepared magnetization is still fresh.
    """
    signal, kspace = mr0.util.simulate(seq0, obj)
    sig = signal.reshape(-1)
    k = kspace.detach().cpu().numpy()
    n_tr = len(sig) // (2 * nx)
    idx = np.concatenate([np.arange(2 * i * nx, 2 * i * nx + nx)
                          for i in range(n_tr)])
    c = int(idx[np.argmin(np.linalg.norm(k[idx][:, :3], axis=1))])
    return sig[c]


def disc_weight(kr: torch.Tensor, radius: float) -> torch.Tensor:
    """Fourier transform of a uniform disc of ``radius`` [m], equal to 1 at k = 0.

    ``kr`` is the in-plane spatial frequency in cycles/m (what MRzero reports):
    ``D(k) = 2 J1(2 pi R k) / (2 pi R k)``.
    """
    x = 2.0 * np.pi * radius * kr
    safe = torch.where(x < 1e-9, torch.ones_like(x), x)
    d = 2.0 * torch.special.bessel_j1(safe) / safe
    return torch.where(x < 1e-9, torch.ones_like(x), d)


def react_signals(seq0, obj, nx: int, radii) -> Tuple[torch.Tensor, dict]:
    """One simulation, every REACT contrast metric.

    Returns ``(dc, {radius: A})``: ``dc`` is the single sample nearest the k-space
    centre (what :func:`react_dc_signal` returns) and ``A`` the disc-vessel
    amplitude of :func:`react_object_signal` for each radius in ``radii`` [m].
    Sharing the simulation matters: it is the slow part, and it keeps the two
    metrics strictly comparable.
    """
    signal, kspace = mr0.util.simulate(seq0, obj)
    sig = signal.reshape(-1)
    k = kspace.detach().cpu().numpy()
    if np.abs(k[:, 2]).max() > 1e-3:
        raise NotImplementedError(
            "react_object_signal models a 2D disc; the sequence has kz encoding")
    n_tr = len(sig) // (2 * nx)
    idx = np.concatenate([np.arange(2 * i * nx, 2 * i * nx + nx)
                          for i in range(n_tr)])
    kr_np = np.linalg.norm(k[idx][:, :2], axis=1)
    dc = sig[int(idx[np.argmin(kr_np)])]
    kr = torch.from_numpy(kr_np).float()
    out = {}
    for r in radii:
        d = disc_weight(kr, r)
        out[r] = (sig[idx] * d).sum() / d.sum()
    return dc, out


def react_object_signal(seq0, obj, radius: float, nx: int) -> torch.Tensor:
    """Complex signal at the centre of a uniform disc vessel, over the whole scan.

    :func:`react_dc_signal` returns the single k-space-centre sample, which is
    acquired in TR 0 of the first shot, so it scores only ``|Mz(TI)|`` and never
    sees how the prepared magnetization decays along the 20-odd TRs of a shot.
    A vessel of radius ``R`` does not live at k = 0: its spectrum is the disc
    transform ``D(k)`` and every acquired sample contributes with that weight. For
    a point voxel the simulated stream is ``S(k) = m(TR(k))``, the magnetization
    available when that line was acquired, so the amplitude at the centre of a
    fully sampled disc is::

        A = | sum_k S(k) D(k) | / sum_k D(k)

    which is 1 if ``S`` were uniform. Only echo 1 is used (echo 2 exists to
    encode fat). ``obj`` should be a single voxel with a *small* ``voxel_size``:
    MRzero multiplies the signal by the voxel's own sinc envelope along the
    readout, which would otherwise zero the outer part of every line.

    Checked against an explicit disc of 113 voxels (R = 3 mm, 0.5 mm pitch) in the
    same sequence: 0.0931 against 0.0922, 1%.

    Limits: a 2D disc. For 3D (``nz > 1``) a vessel along z would weigh only the
    ``kz = 0`` plane, which this does not model, so it refuses rather than
    return a number that looks right.
    """
    return react_signals(seq0, obj, nx, [radius])[1][radius]


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
