#!/usr/bin/env python
"""Flip-angle optimization: fixed vs coupled iNAV ramp (real MRzero gradient).

``optimize_flip.py`` optimizes the imaging flip with the iNAV catalyzation ramp
held fixed (it ramps to 90 deg regardless of the imaging flip). Physically the
ramp should end at the imaging flip. This script optimizes the flip both ways --
ramp fixed and ramp coupled to the flip -- and compares the contrast curves and
optima.

Usage
-----
    python scripts/optimize_flip_coupled.py [--out flip_coupled.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MRzeroCore as mr0
from pyboost import (BoostParams, scanner_055T, import_mra_for_optimization,
                     locate_inav_ramp, set_imaging_flip, set_imaging_flip_coupled,
                     central_signal)
from pyboost.phantom import TISSUE_PROPERTIES

BLOOD, MUSCLE = TISSUE_PROPERTIES["blood"], TISSUE_PROPERTIES["muscle"]


def _voxel(tp):
    return mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                  T2=tp["T2"], T2dash=0.03, D=0.0, voxel_size=5e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="flip_coupled.png")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    system = scanner_055T()
    p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                    centric=True, t2prep_duration=0.06, im_flip_angle=(90.0, 80.0))
    seq0, img_idx = import_mra_for_optimization(p, system, base_flip_deg=90.0)
    ramp_idx, base_ramp = locate_inav_ramp(seq0, p.inav_flip_angle, 90.0)
    per = p.im_segments * p.nx
    blood, muscle = _voxel(BLOOD), _voxel(MUSCLE)
    print(f"{len(img_idx)} imaging pulses, {len(ramp_idx)} iNAV ramp pulses")

    def apply(flip, couple):
        if couple:
            set_imaging_flip_coupled(seq0, img_idx, ramp_idx, base_ramp, flip,
                                     p.inav_flip_angle, 90.0)
        else:
            set_imaging_flip(seq0, img_idx, flip)

    def contrast(flip, couple):
        apply(flip, couple)
        return central_signal(seq0, blood, per) - central_signal(seq0, muscle, per)

    def optimize(couple):
        flip = torch.tensor(45.0, requires_grad=True)
        opt = torch.optim.Adam([flip], lr=4.0)
        for _ in range(args.iters):
            opt.zero_grad()
            (-contrast(flip, couple)).backward()
            opt.step()
            with torch.no_grad():
                flip.clamp_(3.0, 178.0)
        return flip.item()

    results = {}
    fa = np.arange(20, 180, 10)
    for couple in (False, True):
        opt_flip = optimize(couple)
        with torch.no_grad():
            curve = np.array([contrast(torch.tensor(float(a)), couple).item()
                              for a in fa])
        results[couple] = (opt_flip, curve)
        print(f"{'coupled' if couple else 'fixed  '} ramp: Adam optimum flip = "
              f"{opt_flip:.0f} deg, grid peak = {fa[np.argmax(curve)]} deg")

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for couple, color in ((False, "#2980b9"), (True, "#c0392b")):
        opt_flip, curve = results[couple]
        lbl = "coupled ramp" if couple else "fixed ramp (90 deg)"
        ax.plot(fa, curve, "-o", ms=4, color=color, label=lbl)
        ax.axvline(opt_flip, color=color, ls=":", lw=1.5)
    ax.set_xlabel("imaging flip angle [deg]")
    ax.set_ylabel("blood - muscle contrast")
    ax.set_title("Flip optimization: fixed vs coupled iNAV ramp @ 0.55T "
                 "(MRzero gradient)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
