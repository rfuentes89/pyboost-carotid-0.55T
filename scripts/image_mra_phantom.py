#!/usr/bin/env python
"""Image the carotid phantom with the bright-blood MRA sequence.

Relaxation-based bright-blood MRA (T2-prep + FatSat + bSSFP) reconstructs to a
real image where the lumen is bright and the background muscle/fat are
suppressed -- the contrast is entirely T2/T1-driven, so unlike black-blood it is
faithfully simulatable here.

Usage
-----
    python scripts/image_mra_phantom.py [--out mra_carotid.png] [--matrix 48]
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
from pyboost import build_mra_sequence, BoostParams, scanner_055T
from pyboost.phantom import carotid_phantom_maps, to_mrzero_phantom, CarotidGeometry


def reconstruct(signal, kspace, p, fov, slice_thk):
    """Magnitude image from the single-contrast, multi-shot MRA acquisition."""
    import torch

    nx, im_seg, n_shots = p.nx, p.im_segments, p.n_shots
    per_seg = im_seg * nx
    seg_phase = torch.tensor(
        np.repeat(np.exp(-1j * np.pi * ((p.inav_lines + np.arange(im_seg)) % 2)), nx),
        dtype=signal.dtype)
    s_parts, k_parts = [], []
    for shot in range(n_shots):                       # one segment per heartbeat
        block = slice(shot * per_seg, (shot + 1) * per_seg)
        s_parts.append(signal[block].reshape(-1) * seg_phase)
        k_parts.append(kspace[block])
    img = mr0.reco_adjoint(torch.cat(s_parts).reshape(-1, 1), torch.cat(k_parts),
                           resolution=(nx, p.ny, 1), FOV=(fov, fov, slice_thk))
    # reco_adjoint returns the image transposed w.r.t. the phantom (x,y) grid.
    return np.abs(img.detach().cpu().numpy().reshape(p.ny, nx)).T


def _downsample_labels(label, n):
    src = label.shape[0]
    idx = (np.arange(n) * src / n).astype(int)
    return label[np.ix_(idx, idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mra_carotid.png")
    ap.add_argument("--matrix", type=int, default=48)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--t2prep", type=float, default=0.05, help="T2-prep TE [s]")
    args = ap.parse_args()

    fov = 0.06
    geom = CarotidGeometry(fov=fov, matrix=96)
    maps = carotid_phantom_maps(geom)
    obj = to_mrzero_phantom(maps, fov)
    print(f"phantom: {len(obj.PD)} voxels, FOV {fov*1e3:.0f} mm")

    system = scanner_055T()
    n = args.matrix
    p = BoostParams(fov=fov, nx=n, ny=n, im_segments=args.segments,
                    dummy_heart_beats=1, centric=True, t2prep_duration=args.t2prep)
    seq = build_mra_sequence(p, system, use_t2prep=True, use_fatsat=True,
                             add_trigger=False)
    seq.write("/tmp/mra_carotid.seq")
    seq0 = mr0.Sequence.import_file("/tmp/mra_carotid.seq")
    print(f"sequence: {len(seq0)} reps, {n}x{n}, T2prep {args.t2prep*1e3:.0f} ms")

    print("simulating ...")
    signal, kspace = mr0.util.simulate(seq0, obj)
    img = reconstruct(signal, kspace, p, fov, p.slice_thickness)

    # reconstruct() transposes to match the phantom grid, so the raw label map
    # aligns with the image for both display and per-tissue statistics.
    label_disp = maps["label"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(label_disp, cmap="viridis"); ax[0].set_title("phantom (tissues)")
    ax[1].imshow(img, cmap="gray"); ax[1].set_title("bright-blood MRA\n(T2prep+FatSat+bSSFP)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Carotid bright-blood MRA @ 0.55T — lumen bright "
                 f"(T2prep {args.t2prep*1e3:.0f} ms)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")

    lab = _downsample_labels(label_disp, n)
    def m(l):
        return img[lab == l].mean() if np.any(lab == l) else float("nan")
    print(f"lumen={m(1):.3g}  wall={m(2):.3g}  muscle={m(3):.3g}  fat={m(4):.3g}")
    print(f"blood/muscle contrast = {m(1)/max(m(3),1e-9):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
