#!/usr/bin/env python
"""Optimize REACT's inversion time against the simulated 0.55T contrast.

Why this exists
---------------
``ReactParams.resolved_ti()`` derives TI analytically as ``T1*ln(1 + E2)``, the
point where a chosen tissue crosses zero. That derivation is exact for a single
isochromat at the instant of the first excitation, but the sequence is not a
single instant: magnetization keeps recovering through the shot, the 15 deg
train nibbles at Mz, and -- most importantly -- the shot interval does not fully
recover blood (T1 = 1122 ms at 0.55T), so the tissue that suffers most from a
short interval is exactly the one REACT wants brightest.

So the analytic TI is the right starting point and the wrong final answer. This
script measures the real contrast through MRzero and moves TI by gradient
descent, which is the same trick the repo already uses for the flip angle and
the T2-prep echo time.

Usage
-----
    python scripts/optimize_react_ti.py [--steps 40] [--sweep]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MRzeroCore as mr0
from pyboost import ReactParams, scanner_055T
from pyboost.diffopt import (import_react_for_optimization, set_react_ti,
                             react_dc_signal)
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES


def voxel(name: str):
    t = TISSUES[name]
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=t["PD"], T1=t["T1"], T2=t["T2"],
        T2dash=0.03, D=0.0, voxel_size=0.005,
    )


def contrast(seq0, inv_idx, info, ti, objs, nx):
    """Blood signal minus background. Differentiable w.r.t. ``ti``."""
    set_react_ti(seq0, inv_idx, info, ti)
    blood = react_dc_signal(seq0, objs["blood"], nx).abs()
    muscle = react_dc_signal(seq0, objs["muscle"], nx).abs()
    fat = react_dc_signal(seq0, objs["fat"], nx).abs()
    return blood - muscle - fat, blood, muscle, fat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--sweep", action="store_true",
                    help="also print a coarse TI sweep (slow but shows the shape)")
    ap.add_argument("--shot-interval", type=float, default=1000.0,
                    help="recovery between shots [ms]")
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    p = ReactParams(nx=16, ny=8, tfe_factor=8, dummy_shots=0,
                    shot_interval=args.shot_interval * 1e-3)
    seq0, inv_idx, info = import_react_for_optimization(p, system)
    objs = {n: voxel(n) for n in ("blood", "muscle", "fat")}

    ti0 = p.resolved_ti()
    print(f"analytic TI (nulls {p.ti_null_tissue}) = {ti0*1e3:.1f} ms")
    print(f"shot interval = {p.shot_interval*1e3:.0f} ms, "
          f"blood T1 = {TISSUES['blood']['T1']*1e3:.0f} ms "
          f"({'under-recovered' if p.shot_interval < 3*TISSUES['blood']['T1'] else 'ok'})\n")

    if args.sweep:
        print(f"{'TI [ms]':>9}{'blood':>10}{'muscle':>10}{'fat':>10}{'contrast':>11}")
        for ti_ms in range(40, 221, 20):
            ti = torch.tensor(ti_ms * 1e-3)
            with torch.no_grad():
                c, b, m, f = contrast(seq0, inv_idx, info, ti, objs, p.nx)
            print(f"{ti_ms:>9}{b.item():>10.4f}{m.item():>10.4f}"
                  f"{f.item():>10.4f}{c.item():>11.4f}")
        print()

    ti = torch.tensor(float(ti0), requires_grad=True)
    opt = torch.optim.Adam([ti], lr=args.lr)
    best = (-np.inf, float(ti0))
    print(f"{'step':>5}{'TI [ms]':>10}{'blood':>10}{'muscle':>10}{'fat':>10}"
          f"{'contrast':>11}")
    for step in range(args.steps):
        opt.zero_grad()
        c, b, m, f = contrast(seq0, inv_idx, info, ti, objs, p.nx)
        (-c).backward()
        if c.item() > best[0]:
            best = (c.item(), float(ti.item()))
        if step % 5 == 0 or step == args.steps - 1:
            print(f"{step:>5}{ti.item()*1e3:>10.2f}{b.item():>10.4f}"
                  f"{m.item():>10.4f}{f.item():>10.4f}{c.item():>11.4f}")
        opt.step()
        with torch.no_grad():                 # keep TI physically realisable
            ti.clamp_(min=0.02, max=0.4)

    print(f"\nbest TI = {best[1]*1e3:.1f} ms (contrast {best[0]:.4f}) "
          f"vs analytic {ti0*1e3:.1f} ms")
    print("Set it explicitly with ReactParams(ti=<seconds>) or "
          "scripts/write_react_055T.py --ti <ms>.")
    print("NOTE: optimized on a hard-pulse inversion because MRzero cannot model "
          "an\n      adiabatic sweep. The optimum transfers -- an adiabatic pulse "
          "inverts more\n      robustly, not differently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
