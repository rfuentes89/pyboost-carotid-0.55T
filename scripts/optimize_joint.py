#!/usr/bin/env python
"""Joint differentiable optimization of flip angle AND T2-prep TE for MRA.

Both parameters are optimized at once by gradient descent through the *real*
MRzero simulation: the imaging pulses' angle is a torch tensor (flip), and the
T2-prep delay ``event_time`` is scaled by another torch tensor (TE). Adam
maximizes the blood-to-muscle contrast over the 2D landscape.

Usage
-----
    python scripts/optimize_joint.py [--out joint_opt.png]
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
                     locate_t2prep_delays, set_imaging_flip, set_t2prep_te,
                     central_signal)
from pyboost.phantom import TISSUE_PROPERTIES

BLOOD, MUSCLE = TISSUE_PROPERTIES["blood"], TISSUE_PROPERTIES["muscle"]
BASE_TE = 0.06


def _voxel(tp):
    return mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                  T2=tp["T2"], T2dash=0.03, D=0.0, voxel_size=5e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="joint_opt.png")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    system = scanner_055T()
    p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                    centric=True, t2prep_duration=BASE_TE, im_flip_angle=(90.0, 80.0))
    seq0, img_idx = import_mra_for_optimization(p, system, base_flip_deg=90.0)
    te_idx, base_et = locate_t2prep_delays(seq0)
    per = p.im_segments * p.nx
    blood, muscle = _voxel(BLOOD), _voxel(MUSCLE)
    print(f"{len(img_idx)} imaging pulses (flip) + {len(te_idx)} T2prep-delay reps "
          f"(TE) made differentiable")

    def contrast(flip, te):
        set_imaging_flip(seq0, img_idx, flip)
        set_t2prep_te(seq0, te_idx, base_et, te, base_te=BASE_TE)
        b = central_signal(seq0, blood, per)
        m = central_signal(seq0, muscle, per)
        return b, m

    # --- joint gradient descent -----------------------------------------
    flip = torch.tensor(50.0, requires_grad=True)
    te_ms = torch.tensor(40.0, requires_grad=True)      # optimize TE in ms
    opt = torch.optim.Adam([{"params": [flip], "lr": 4.0},
                            {"params": [te_ms], "lr": 3.0}])
    traj = []
    for it in range(args.iters):
        opt.zero_grad()
        b, m = contrast(flip, te_ms * 1e-3)
        (-(b - m)).backward()
        opt.step()
        with torch.no_grad():
            flip.clamp_(3.0, 178.0)
            te_ms.clamp_(5.0, 200.0)
        traj.append((flip.item(), te_ms.item(), (b - m).item()))
    fo, to = flip.item(), te_ms.item()
    with torch.no_grad():
        b, m = contrast(torch.tensor(fo), torch.tensor(to) * 1e-3)
    print(f"Joint optimum: flip={fo:.0f} deg, T2prep TE={to:.0f} ms  "
          f"contrast={(b-m).item():.3f}  blood/muscle={b.item()/m.item():.2f}")

    # --- 2D contrast landscape ------------------------------------------
    fa = np.arange(30, 171, 20)
    te = np.arange(10, 161, 20)
    Z = np.zeros((len(te), len(fa)))
    with torch.no_grad():
        for ii, t in enumerate(te):
            for jj, a in enumerate(fa):
                b, m = contrast(torch.tensor(float(a)), torch.tensor(float(t)) * 1e-3)
                Z[ii, jj] = (b - m).item()
    print(f"grid maximum: flip={fa[np.argmax(Z)%len(fa)]} deg, "
          f"TE={te[np.argmax(Z)//len(fa)]} ms")

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                   extent=[fa[0], fa[-1], te[0], te[-1]])
    fig.colorbar(im, label="blood - muscle contrast")
    tr = np.array(traj)
    ax.plot(tr[:, 0], tr[:, 1], "-", color="white", lw=1.2, alpha=0.8)
    ax.plot(tr[0, 0], tr[0, 1], "wo", ms=7, label="start")
    ax.plot(fo, to, "r*", ms=16, label=f"optimum ({fo:.0f} deg, {to:.0f} ms)")
    ax.set_xlabel("flip angle [deg]"); ax.set_ylabel("T2-prep TE [ms]")
    ax.set_title("Joint flip + T2prep optimization (MRzero gradient) @ 0.55T")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
