#!/usr/bin/env python
"""Differentiable optimization of the bSSFP flip angle for bright-blood MRA.

T2-prep is fixed at 60 ms (the sweet spot from ``optimize_mra.py``); the imaging
flip angle is optimized by gradient descent (Adam) to maximize the blood-to-
muscle contrast. The gradient flows through the *real* MRzero simulation: the
sequence is imported, then the imaging pulses' angle is replaced by a torch
parameter so ``mr0.util.simulate`` is differentiable w.r.t. it (see
``pyboost.diffopt``). A brute-force MRzero sweep is overlaid to confirm the
gradient lands on the true optimum.

Usage
-----
    python scripts/optimize_flip.py [--out flip_opt.png] [--t2prep 0.06]
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
                     differentiable_flip_signal)
from pyboost.phantom import TISSUE_PROPERTIES

BLOOD, MUSCLE = TISSUE_PROPERTIES["blood"], TISSUE_PROPERTIES["muscle"]


def _voxel(tp):
    return mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                  T2=tp["T2"], T2dash=0.03, D=0.0, voxel_size=5e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="flip_opt.png")
    ap.add_argument("--t2prep", type=float, default=0.06)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    system = scanner_055T()
    p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                    centric=True, t2prep_duration=args.t2prep,
                    im_flip_angle=(90.0, 80.0))
    seq0, img_idx = import_mra_for_optimization(p, system, base_flip_deg=90.0)
    per = p.im_segments * p.nx
    blood, muscle = _voxel(BLOOD), _voxel(MUSCLE)
    print(f"imported {len(seq0)} reps; {len(img_idx)} imaging pulses made "
          f"differentiable; T2prep {args.t2prep*1e3:.0f} ms")

    def contrast(flip):
        b = differentiable_flip_signal(seq0, img_idx, flip, blood, per)
        m = differentiable_flip_signal(seq0, img_idx, flip, muscle, per)
        return b, m

    # --- gradient descent through MRzero --------------------------------
    flip = torch.tensor(45.0, requires_grad=True)
    opt = torch.optim.Adam([flip], lr=4.0)
    traj = []
    for it in range(args.iters):
        opt.zero_grad()
        b, m = contrast(flip)
        (-(b - m)).backward()                          # maximize blood - muscle
        opt.step()
        with torch.no_grad():
            flip.clamp_(3.0, 178.0)
        traj.append((flip.item(), (b - m).item()))
    flip_opt = flip.item()
    with torch.no_grad():
        b_opt, m_opt = contrast(torch.tensor(flip_opt))
    print(f"Adam optimum (via MRzero gradient): flip = {flip_opt:.1f} deg  "
          f"blood={b_opt.item():.3f} muscle={m_opt.item():.3f}  "
          f"contrast={(b_opt-m_opt).item():.3f}  blood/muscle={b_opt.item()/m_opt.item():.2f}")

    # --- brute-force MRzero sweep (same engine, no grad) ----------------
    fa = np.arange(20, 180, 10)
    cb, cm = [], []
    with torch.no_grad():
        for a in fa:
            b, m = contrast(torch.tensor(float(a)))
            cb.append(b.item()); cm.append(m.item())
    cb, cm = np.array(cb), np.array(cm)
    print(f"brute-force MRzero peak: flip = {fa[np.argmax(cb-cm)]} deg")

    # --- figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fa, cb, "-o", ms=4, color="#c0392b", label="blood")
    ax1.plot(fa, cm, "-o", ms=4, color="#27ae60", label="muscle")
    ax1.plot(fa, cb - cm, "--s", ms=4, color="#8e44ad", label="contrast blood-muscle")
    ax1.axvline(flip_opt, color="0.4", ls=":", lw=1.5,
                label=f"Adam optimum {flip_opt:.0f} deg")
    ax1.set_xlabel("flip angle [deg]"); ax1.set_ylabel("central-k signal")
    ax1.set_title(f"MRzero contrast vs flip (T2prep {args.t2prep*1e3:.0f} ms)")
    ax1.legend(fontsize=8)

    tr = np.array(traj)
    ax2.plot(tr[:, 0], color="#2980b9")
    ax2.axhline(flip_opt, color="0.4", ls=":", lw=1.5)
    ax2.set_xlabel("Adam iteration"); ax2.set_ylabel("flip angle [deg]")
    ax2.set_title("Gradient descent through MRzero converges")
    fig.suptitle("Differentiable flip optimization (real MRzero gradient) "
                 "for bright-blood MRA @ 0.55T")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
