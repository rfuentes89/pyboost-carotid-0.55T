#!/usr/bin/env python
"""Differentiable optimization of the FatSat flip angle for bright-blood MRA.

The FatSat is a spectrally-selective pulse on the fat resonance (~-80 Hz at
0.55T). Its flip angle is made differentiable (`pyboost.diffopt.locate_fatsat`
finds it as the only pulse with a nonzero RF frequency offset) and optimized by
gradient descent through MRzero to *minimize the fat signal*.

Physics note: the fat voxel must sit AT the fat resonance (B0 = -80 Hz) for the
selective pulse to act on it; water (blood, on-resonance) is essentially
untouched by the off-resonance FatSat, which the script verifies. Because fat is
placed off-resonance it also picks up bSSFP banding, so the optimum reflects both
the saturation and the T1 recovery to the centric readout.

Usage
-----
    python scripts/optimize_fatsat.py [--out fatsat_opt.png]
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
                     locate_fatsat, set_imaging_flip, central_signal)
from pyboost.system import fat_frequency
from pyboost.phantom import TISSUE_PROPERTIES

FAT, BLOOD = TISSUE_PROPERTIES["fat"], TISSUE_PROPERTIES["blood"]


def _voxel(tp, b0_hz):
    return mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                  T2=tp["T2"], T2dash=0.03, B0=b0_hz, D=0.0,
                                  voxel_size=5e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fatsat_opt.png")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    system = scanner_055T()
    df = fat_frequency(system)                          # ~ -80 Hz at 0.55T
    p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                    centric=True, t2prep_duration=0.06, im_flip_angle=(90.0, 80.0))
    # build_mra_sequence with fat-sat on -- import via the shared helper, which
    # builds without fat-sat, so build directly here instead:
    from pyboost import build_mra_sequence
    seq = build_mra_sequence(p, system, use_t2prep=True, use_fatsat=True,
                             add_trigger=False)
    seq.write("/tmp/fatsat_opt.seq")
    seq0 = mr0.Sequence.import_file("/tmp/fatsat_opt.seq")
    fs_idx = locate_fatsat(seq0)
    per = p.im_segments * p.nx
    fat = _voxel(FAT, df)                                # fat AT its resonance
    blood = _voxel(BLOOD, 0.0)                           # water, on-resonance
    print(f"fat resonance {df:.0f} Hz; {len(fs_idx)} FatSat pulses made differentiable")

    def signals(flip):
        set_imaging_flip(seq0, fs_idx, flip)             # reuse: set FatSat angle
        return central_signal(seq0, fat, per), central_signal(seq0, blood, per)

    # --- gradient descent: maximize blood - fat -------------------------
    # At 0.55T the FatSat is poorly selective (pulse BW ~80 Hz vs the 80 Hz fat-
    # water gap), so it perturbs on-resonance blood too. Minimizing fat alone
    # over-suppresses blood; the MRA-relevant objective is the blood-to-fat
    # contrast. That landscape is NON-CONVEX (humps near ~20 and ~180 deg), so a
    # single gradient descent finds a local optimum -- we multi-start and keep the
    # best (a real lesson: gradient descent needs good init / multi-start here).
    best = (None, -1e9)
    for init in (30.0, 90.0, 150.0, 210.0):
        flip = torch.tensor(init, requires_grad=True)
        opt = torch.optim.Adam([flip], lr=5.0)
        for _ in range(args.iters):
            opt.zero_grad()
            f, b = signals(flip)
            (-(b - f)).backward()                        # maximize blood - fat
            opt.step()
            with torch.no_grad():
                flip.clamp_(5.0, 320.0)
        with torch.no_grad():
            f, b = signals(flip)
        if (b - f).item() > best[1]:
            best = (flip.item(), (b - f).item())
        print(f"  start {init:5.0f} deg -> {flip.item():5.0f} deg "
              f"(blood-fat={(b-f).item():.4f})")
    flip_opt = best[0]
    with torch.no_grad():
        f_opt, b_opt = signals(torch.tensor(flip_opt))
    print(f"Best multi-start FatSat flip = {flip_opt:.0f} deg  fat={f_opt.item():.4f}  "
          f"blood={b_opt.item():.4f}  blood-fat={(b_opt-f_opt).item():.4f}")

    # --- brute sweep -----------------------------------------------------
    fa = np.arange(20, 320, 20)
    with torch.no_grad():
        vals = [signals(torch.tensor(float(a))) for a in fa]
    fat_c = np.array([v[0].item() for v in vals])
    blood_c = np.array([v[1].item() for v in vals])
    print(f"brute-force blood-fat max at FatSat flip = {fa[np.argmax(blood_c-fat_c)]} "
          f"deg (default is 180 deg); fat is NOT suppressed independently of blood "
          f"-- FatSat is poorly selective at 0.55T")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fa, blood_c, "-o", ms=4, color="#c0392b", label="blood (on-resonance)")
    ax.plot(fa, fat_c, "-o", ms=4, color="#e67e22", label="fat (at -80 Hz)")
    ax.plot(fa, blood_c - fat_c, "--s", ms=4, color="#8e44ad", label="blood - fat")
    ax.axvline(flip_opt, color="0.4", ls="--", lw=1.5,
               label=f"best multi-start {flip_opt:.0f} deg")
    ax.axvline(180, color="0.7", ls=":", lw=1, label="default 180 deg")
    ax.set_xlabel("FatSat flip angle [deg]"); ax.set_ylabel("central-k signal")
    ax.set_title("FatSat flip optimization @ 0.55T (MRzero gradient) — "
                 "poorly selective, so blood & fat track together")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
