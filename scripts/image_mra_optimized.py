#!/usr/bin/env python
"""Bright-blood MRA phantom image at a sub-optimal vs the optimized flip angle.

Uses the optimization results (flip ~110 deg, T2-prep TE ~79 ms) to show the
lumen conspicuity gain over a poorly-chosen flip.

Usage
-----
    python scripts/image_mra_optimized.py [--out mra_optimized.png]
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
from scripts.image_mra_phantom import reconstruct, _downsample_labels


def image_at(system, obj, fov, flip, te, matrix, segments):
    p = BoostParams(fov=fov, nx=matrix, ny=matrix, im_segments=segments,
                    dummy_heart_beats=1, centric=True, t2prep_duration=te,
                    im_flip_angle=(flip, 80.0))
    seq = build_mra_sequence(p, system, use_t2prep=True, use_fatsat=True,
                             add_trigger=False)
    seq.write("/tmp/mra_o.seq")
    seq0 = mr0.Sequence.import_file("/tmp/mra_o.seq")
    signal, kspace = mr0.util.simulate(seq0, obj)
    return reconstruct(signal, kspace, p, fov, p.slice_thickness)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mra_optimized.png")
    ap.add_argument("--matrix", type=int, default=48)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--te", type=float, default=0.079)
    args = ap.parse_args()

    fov = 0.06
    maps = carotid_phantom_maps(CarotidGeometry(fov=fov, matrix=96))
    obj = to_mrzero_phantom(maps, fov)
    system = scanner_055T()
    lab = _downsample_labels(maps["label"], args.matrix)

    def stats(img):
        m = lambda l: img[lab == l].mean() if np.any(lab == l) else float("nan")
        return m(1), m(3)                              # lumen, muscle

    print("simulating (two flips) ...")
    imgs = {}
    for flip in (40.0, 110.0):
        img = image_at(system, obj, fov, flip, args.te, args.matrix, args.segments)
        lu, mu = stats(img)
        imgs[flip] = (img, lu, mu)
        print(f"flip={flip:.0f} deg  lumen={lu:.3g}  muscle={mu:.3g}  "
              f"lumen/muscle={lu/max(mu,1e-9):.2f}")

    vmax = max(imgs[f][0].max() for f in imgs)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(maps["label"], cmap="viridis"); ax[0].set_title("phantom (tissues)")
    for a, flip in zip(ax[1:], (40.0, 110.0)):
        img, lu, mu = imgs[flip]
        a.imshow(img, cmap="gray", vmax=vmax)
        a.set_title(f"MRA flip {flip:.0f} deg\nlumen/muscle = {lu/max(mu,1e-9):.2f}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Bright-blood MRA @ 0.55T: sub-optimal vs optimized flip "
                 f"(T2prep {args.te*1e3:.0f} ms)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
