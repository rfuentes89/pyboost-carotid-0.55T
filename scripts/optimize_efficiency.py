#!/usr/bin/env python
"""CNR-efficiency optimization of flip + T2-prep TE for bright-blood MRA.

Pure blood-muscle contrast pushes the T2-prep TE up (~95 ms), sacrificing signal
and time. In a cardiac-gated scan the total time is fixed (n_heartbeats x RR), so
CNR / sqrt(total_time) reduces to plain contrast. The meaningful cost of a long
T2-prep is that it is *overhead* stealing time from the readout, so we optimize

    efficiency = (S_blood - S_muscle) / sqrt(TE + T_readout)

which penalizes a long prep and yields a shorter, more practical TE. Both
parameters are differentiated through the real MRzero simulation.

Usage
-----
    python scripts/optimize_efficiency.py [--out efficiency_opt.png]
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
    ap.add_argument("--out", default="efficiency_opt.png")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    system = scanner_055T()
    p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                    centric=True, t2prep_duration=BASE_TE, im_flip_angle=(90.0, 80.0))
    seq0, img_idx = import_mra_for_optimization(p, system, base_flip_deg=90.0)
    te_idx, base_et = locate_t2prep_delays(seq0)
    per = p.im_segments * p.nx
    blood, muscle = _voxel(BLOOD), _voxel(MUSCLE)
    t_readout = p.im_segments * p.tr                    # readout duration [s]
    print(f"readout time {t_readout*1e3:.0f} ms (overhead penalty on TE)")

    def contrast(flip, te):
        set_imaging_flip(seq0, img_idx, flip)
        set_t2prep_te(seq0, te_idx, base_et, te, base_te=BASE_TE)
        return central_signal(seq0, blood, per) - central_signal(seq0, muscle, per)

    def optimize(objective):
        flip = torch.tensor(60.0, requires_grad=True)
        te_ms = torch.tensor(50.0, requires_grad=True)
        opt = torch.optim.Adam([{"params": [flip], "lr": 4.0},
                                {"params": [te_ms], "lr": 3.0}])
        for _ in range(args.iters):
            opt.zero_grad()
            c = contrast(flip, te_ms * 1e-3)
            (-objective(c, te_ms * 1e-3)).backward()
            opt.step()
            with torch.no_grad():
                flip.clamp_(3.0, 178.0); te_ms.clamp_(5.0, 200.0)
        return flip.item(), te_ms.item()

    eff = lambda c, te: c / torch.sqrt(te + t_readout)
    con = lambda c, te: c
    f_eff, te_eff = optimize(eff)
    f_con, te_con = optimize(con)
    print(f"CNR-efficiency optimum: flip={f_eff:.0f} deg, TE={te_eff:.0f} ms")
    print(f"pure-contrast optimum:  flip={f_con:.0f} deg, TE={te_con:.0f} ms")

    # --- TE curves at the efficiency-optimal flip -----------------------
    te = np.arange(10, 161, 15)
    with torch.no_grad():
        c_curve = np.array([contrast(torch.tensor(f_eff),
                                     torch.tensor(float(t)) * 1e-3).item() for t in te])
    e_curve = c_curve / np.sqrt(te * 1e-3 + t_readout)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(te, c_curve, "-o", color="#8e44ad", label="contrast (blood-muscle)")
    ax1.axvline(te_con, color="#8e44ad", ls=":", lw=1.5, label=f"contrast opt {te_con:.0f} ms")
    ax1.set_xlabel("T2-prep TE [ms]"); ax1.set_ylabel("contrast", color="#8e44ad")
    ax2 = ax1.twinx()
    ax2.plot(te, e_curve, "-s", color="#16a085", label="CNR efficiency")
    ax2.axvline(te_eff, color="#16a085", ls="--", lw=1.5, label=f"efficiency opt {te_eff:.0f} ms")
    ax2.set_ylabel("contrast / sqrt(TE + T_readout)", color="#16a085")
    ax1.set_title(f"Contrast vs CNR-efficiency (flip {f_eff:.0f} deg) @ 0.55T")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
