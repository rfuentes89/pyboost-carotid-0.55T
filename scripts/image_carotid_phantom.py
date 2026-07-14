#!/usr/bin/env python
"""Image the concentric carotid phantom with the BOOST .seq and reconstruct.

Hybrid pipeline, end to end: PyPulseq builds a multi-shot BOOST sequence (block
inversion, centric ordering, blood-null TI) -> imported into MRzero -> simulated
on the 2D carotid phantom -> both contrasts reconstructed (reco_adjoint) and
combined phase-sensitively.

WHAT THIS SHOWS (and its ceiling). The result is a T1-weighted inversion-recovery
image: the long-T1 vessel darkens relative to short-T1 muscle. It does NOT show
true black-blood lumen suppression, because blood (T1~1122 ms) and vessel wall
(T1~750 ms) are too close in T1 -- inverting to null blood darkens the whole
vessel, wall included. Real carotid black-blood relies on FLOW (blood leaves the
imaging slice), which none of the available simulators model. Simulation here
validates the prep-pulse / T1 physics; flow-void contrast needs the scanner.

Usage
-----
    python scripts/image_carotid_phantom.py [--out carotid_boost.png] [--ti 0.54]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MRzeroCore as mr0
from pyboost import build_boost_sequence, BoostParams, scanner_055T
from pyboost.phantom import carotid_phantom_maps, to_mrzero_phantom, CarotidGeometry


def grid_and_reconstruct(signal, kspace, p, fov, slice_thk):
    """Two contrast images via MRzero's adjoint reconstruction (multi-shot).

    Acquisition order per heartbeat is contrast-0 segment then contrast-1
    segment; segments from different shots interleave different phase-encode
    lines. We gather all segments of a contrast (across shots) and hand their
    true (kx, ky) coordinates to ``reco_adjoint``, which places each sample
    correctly regardless of order. The alternating (0, pi) bSSFP RF phase is
    demodulated per within-segment line index (line j -> tr_index inav_lines+j).
    """
    import torch

    nx, im_seg, n_shots = p.nx, p.im_segments, p.n_shots
    per_seg = im_seg * nx
    # Demodulation phase for one segment (im_seg lines x nx samples).
    seg_phase = np.repeat(
        np.exp(-1j * np.pi * ((p.inav_lines + np.arange(im_seg)) % 2)), nx)
    seg_phase = torch.tensor(seg_phase, dtype=signal.dtype)

    images = []
    for contrast in (0, 1):
        s_parts, k_parts = [], []
        for shot in range(n_shots):
            start = (shot * 2 + contrast) * per_seg
            block = slice(start, start + per_seg)
            s_parts.append(signal[block].reshape(-1) * seg_phase)
            k_parts.append(kspace[block])
        s = torch.cat(s_parts).reshape(-1, 1)
        k = torch.cat(k_parts)
        img = mr0.reco_adjoint(s, k, resolution=(nx, p.ny, 1),
                               FOV=(fov, fov, slice_thk))
        images.append(img.detach().cpu().numpy().reshape(p.ny, nx))
    return images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="carotid_boost.png")
    ap.add_argument("--matrix", type=int, default=48, help="acquisition matrix")
    ap.add_argument("--segments", type=int, default=16,
                    help="phase-encode lines acquired per heartbeat (short window "
                         "keeps FatSat effective)")
    ap.add_argument("--ti", type=float, default=0.54,
                    help="inversion time [s] (blood-null ~0.54 s, see optimize_ti.py)")
    args = ap.parse_args()

    fov = 0.06
    geom = CarotidGeometry(fov=fov, matrix=96)
    maps = carotid_phantom_maps(geom)
    obj = to_mrzero_phantom(maps, fov)
    print(f"phantom: {len(obj.PD)} voxels on {geom.matrix}^2 grid, FOV {fov*1e3:.0f} mm")

    system = scanner_055T()
    n = args.matrix
    # Multi-shot: a short acquisition window (im_segments TRs) per heartbeat with
    # FatSat reapplied each heartbeat keeps fat saturated -- a single long
    # readout (train >> fat T1) would let fat recover and defeat the FatSat.
    # Block inversion (MRzero can't simulate adiabatic), centric ordering (ky=0
    # sampled right after the prep), and the blood-null TI give real black-blood.
    p = BoostParams(fov=fov, nx=n, ny=n, im_segments=args.segments,
                    dummy_heart_beats=1, inversion_kind="block", centric=True,
                    ir_inversion_time=args.ti)
    seq = build_boost_sequence(p, system, add_trigger=False)
    seq.write("/tmp/carotid_boost.seq")
    seq0 = mr0.Sequence.import_file("/tmp/carotid_boost.seq")
    print(f"sequence: {len(seq0)} reps, acquisition {n}x{n}, "
          f"{p.n_shots} shots x {p.im_segments} lines "
          f"(window {p.im_segments*p.tr*1e3:.0f} ms/heartbeat)")

    print("simulating (this can take a minute) ...")
    signal, kspace = mr0.util.simulate(seq0, obj)
    bright, reference = grid_and_reconstruct(signal, kspace, p, fov,
                                             p.slice_thickness)
    # Align the image y-axis with the phantom label display convention.
    bright, reference = np.flipud(bright), np.flipud(reference)
    # Phase-sensitive black-blood: project the inverted bright-blood contrast onto
    # the phase of the reference. At the blood-null TI the lumen ~ 0 (dark) while
    # the shorter-T1 wall/muscle keep a large magnitude (bright). A plain
    # |bright - reference| cannot do this (it can't null the longest-T1 tissue).
    ref_mag = np.abs(reference)
    psir = np.real(bright * np.conj(reference)) / (ref_mag + 1e-3 * ref_mag.max())
    black_blood = np.abs(psir)

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].imshow(maps["label"], cmap="viridis"); ax[0].set_title("phantom (tissues)")
    ax[1].imshow(np.abs(bright), cmap="gray"); ax[1].set_title("bright-blood (T2prep+IR)")
    ax[2].imshow(ref_mag, cmap="gray"); ax[2].set_title("reference (FatSat)")
    ax[3].imshow(black_blood, cmap="gray")
    ax[3].set_title(f"phase-sensitive IR\nTI={args.ti*1e3:.0f} ms")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Carotid phantom @ 0.55T — T1-weighted IR (whole vessel dark); "
                 "true black-blood needs flow, not modelled")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Quick numeric report on the lumen vs wall in each image.
    lab = _downsample_labels(maps["label"], n)
    for name, img in [("bright", np.abs(bright)), ("reference", ref_mag),
                      ("black-blood", black_blood)]:
        def m(l):
            return img[lab == l].mean() if np.any(lab == l) else float("nan")
        print(f"{name:12s} lumen={m(1):.3g}  wall={m(2):.3g}  "
              f"muscle={m(3):.3g}  fat={m(4):.3g}")
    print("\nNOTE: lumen ~ wall in every contrast -- blood (T1~1122 ms) and wall "
          "(T1~750 ms)\nare too close in T1 for inversion to null blood "
          "selectively; the whole vessel\ndarkens. Real carotid black-blood "
          "relies on FLOW (blood leaves the slice), which\nno simulator here "
          "models. Simulation validates the prep/T1 physics, not flow-void.")
    return 0


def _downsample_labels(label, n):
    """Nearest-neighbour resample the tissue map to the acquisition matrix."""
    src = label.shape[0]
    idx = (np.arange(n) * src / n).astype(int)
    return label[np.ix_(idx, idx)]


if __name__ == "__main__":
    raise SystemExit(main())
