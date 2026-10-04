#!/usr/bin/env python
"""Sweep REACT's shot interval (preparation-to-preparation recovery) at 0.55T.

Why this parameter is different from BOOST's
--------------------------------------------
In a cardiac-gated sequence the shot spacing *is* the RR interval: the scan time
is fixed at ``n_shots x RR`` and there is nothing to trade. REACT is untriggered,
so the interval is free, and it decides how much of blood's magnetization (T1
1122 ms in the table) has come back before the next T2-prep.

Why the figure of merit is efficiency, and why it is not the whole story
------------------------------------------------------------------------
Contrast rises with the interval, so it alone would push the answer to infinity.
The number of shots is fixed by k-space, so a longer interval buys contrast with
scan time; contrast per unit sqrt(time) is the honest figure when the saved time
could be spent on averages::

    efficiency = contrast / sqrt(T_shot)

Under a hard scan-time ceiling maximise contrast subject to it instead (see the
table: the published practice is 1-2 heartbeats, 1.0-2.0 s).

What was wrong before
---------------------
* The contrast was the single k-space-centre sample, acquired in TR 0 of the
  first shot (``pyboost.diffopt.react_dc_signal``). It is now the amplitude at the
  centre of a uniform disc vessel over the whole scan
  (``react_object_signal``; radii 3.0 and 2.2 mm from ultrasound reference
  values, see ``optimize_react_ti.py``). The old metric is printed as ``dc``.
* It simulated a toy shot (``nx=16, ny=8, tfe=8``). It now uses the real
  :class:`ReactParams` geometry.
* An efficiency "decoy" near 700-800 ms reported by earlier versions was a
  transient: with 2 preparations before the measured one blood has not reached
  steady state at short intervals. 12 dummy shots (default) remove it.

The objective is ``water`` = blood - muscle by default (what the Dixon water image
shows); ``--objective full`` also subtracts fat.

Usage
-----
    python scripts/optimize_react_shot_interval.py [--ti 84.2] [--lo 1000 --hi 4000]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MRzeroCore as mr0
from pyboost import ReactParams, scanner_055T
from pyboost.diffopt import (import_react_for_optimization, set_react_ti,
                             locate_react_recovery, set_react_shot_interval,
                             react_signals)
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES

RADII = {"3.0": 3.0e-3, "2.2": 2.2e-3}
NAMES = ("blood", "muscle", "fat")


def voxel(name: str):
    t = TISSUES[name]
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=t["PD"], T1=t["T1"], T2=t["T2"],
        T2dash=0.03, D=0.0, voxel_size=1e-4,
    )


def analytic_optimum(t1: float) -> float:
    """T maximizing (1 - exp(-T/T1)) / sqrt(T): solves 2x = exp(x) - 1."""
    lo, hi = 1e-3, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 2 * mid - (math.exp(mid) - 1) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * t1


def measure(seq0, rec_idx, info, overhead, interval, objs, nx, objective):
    """{'dc': contrast, 'R3.0': contrast, ...} at one shot interval."""
    set_react_shot_interval(seq0, rec_idx, info, overhead, torch.tensor(interval))
    sig = {"dc": {}, **{f"R{k}": {} for k in RADII}}
    with torch.no_grad():
        for n, obj in objs.items():
            dc, areas = react_signals(seq0, obj, nx, list(RADII.values()))
            sig["dc"][n] = float(dc.abs())
            for k, r in RADII.items():
                sig[f"R{k}"][n] = float(areas[r].abs())
    return {m: v["blood"] - v["muscle"] - (v["fat"] if objective == "full" else 0.0)
            for m, v in sig.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=float, default=1000.0, help="sweep start [ms]")
    ap.add_argument("--hi", type=float, default=4000.0, help="sweep end [ms]")
    ap.add_argument("--n", type=int, default=7, help="sweep points")
    ap.add_argument("--objective", choices=("water", "full"), default="water")
    ap.add_argument("--ti", type=float, default=None,
                    help="inversion time [ms] (default: the derived one)")
    ap.add_argument("--dummy-shots", type=int, default=12,
                    help="preparations before the measured ones (steady state)")
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    p = ReactParams(dummy_shots=args.dummy_shots)
    seq0, inv_idx, inv_info = import_react_for_optimization(p, system)
    ti = args.ti * 1e-3 if args.ti is not None else float(p.resolved_ti())
    set_react_ti(seq0, inv_idx, inv_info, torch.tensor(ti))
    rec_idx, rec_info, overhead = locate_react_recovery(seq0, p.flip_angle)
    if not rec_idx:
        print("No recovery repetitions found -- need at least 2 shots.")
        return 1
    objs = {n: voxel(n) for n in NAMES}

    t1_blood = TISSUES["blood"]["T1"]
    print(f"objective {args.objective}, TI {ti*1e3:.1f} ms, {p.n_shots} shots, "
          f"{args.dummy_shots} dummy shots, shot overhead {overhead*1e3:.0f} ms")
    print(f"blood T1 {t1_blood*1e3:.0f} ms; analytic anchor (pure T1 recovery, "
          f"lower bound) {analytic_optimum(t1_blood)*1e3:.0f} ms; "
          f"current default {ReactParams().shot_interval*1e3:.0f} ms\n")

    metrics = ["dc"] + [f"R{k}" for k in RADII]
    print(f"{'T_shot [ms]':>12}" + "".join(f"{m+' c':>10}{m+' eff':>10}" for m in metrics))
    rows = []
    for ms in np.linspace(args.lo, args.hi, args.n):
        c = measure(seq0, rec_idx, rec_info, overhead, float(ms) * 1e-3, objs,
                    p.nx, args.objective)
        eff = {m: v / math.sqrt(ms * 1e-3) for m, v in c.items()}
        rows.append((float(ms), c, eff))
        print(f"{ms:>12.0f}" + "".join(f"{c[m]:>10.4f}{eff[m]:>10.4f}" for m in metrics),
              flush=True)

    print("\nper metric: efficiency optimum, and what 1.5 s and 2.0 s keep of the "
          "3.0 s row (contrast / efficiency)")
    ref = min(rows, key=lambda r: abs(r[0] - 3000.0))
    for m in metrics:
        best = max(rows, key=lambda r: r[2][m])
        s = f"  {m:>5}: efficiency optimum {best[0]:.0f} ms"
        for target in (1500.0, 2000.0):
            r = min(rows, key=lambda r: abs(r[0] - target))
            if abs(r[0] - target) < 1.0 and ref[1][m] > 0:
                s += (f";  {target/1e3:.1f} s -> {r[1][m]/ref[1][m]*100:.0f}% / "
                      f"{r[2][m]/ref[2][m]*100:.0f}%")
        print(s)

    full = ReactParams()
    print(f"\nscan time at the default geometry ({full.n_shots} shots, 2D): "
          + ", ".join(f"{ms/1e3:.1f} s -> {full.n_shots*ms/1e3:.0f} s"
                      for ms in (1500, 2000, 3000)))
    print("Set it with ReactParams(shot_interval=<seconds>). In 3D the shot count "
          "multiplies by nz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
