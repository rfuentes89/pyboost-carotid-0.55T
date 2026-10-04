#!/usr/bin/env python
"""Sweep REACT's inversion time against three contrast objectives (MRzero, 0.55T).

Why three objectives
--------------------
The TI used so far was chosen to maximise ``blood - muscle - fat``. That penalises
fat, but REACT removes fat with the Dixon reconstruction, so the quantity that
matters is the contrast in the **water image**: blood against the tissues that
survive the separation. Whether the optimum moves with the objective is exactly
what this script measures, instead of assuming it does not:

* ``water``  = blood - muscle            (fat removed by Dixon; the REACT case)
* ``fat``    = blood - muscle - fat      (the objective used before; penalises fat)
* ``wall``   = blood - vessel wall       (the nearest tissue to the lumen)

Why a dense sweep and not gradient descent
------------------------------------------
Efficiency landscapes in this sequence have been multi-modal (see
``optimize_react_shot_interval.py``), and descent from one start reports whichever
basin it lands in. A 5 ms grid from 5 to 200 ms, with the two published 1.5T
inversion delays (7.8 ms for the original REACT, 12.2 ms for MTC-REACT; Erdem
2025) and the ~70 ms of Isaak 2021 added as explicit rows, is cheap here.

Read the comparison with care: those published delays are for a different field
strength, tissue values and an unspecified time reference, so they locate the
region the literature uses, not a number to match.

Usage
-----
    python scripts/optimize_react_ti.py [--shot-interval 3.0]
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
from pyboost.params import null_time_after_t2prep
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES

TISSUE_NAMES = ("blood", "wall", "muscle", "fat")
PUBLISHED = {7.8: "original REACT, 1.5T (Erdem 2025)",
             12.2: "MTC-REACT shortest, 1.5T (Erdem 2025)",
             70.0: "Isaak 2021 Table 1, 1.5T"}


def voxel(name: str):
    t = TISSUES[name]
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=t["PD"], T1=t["T1"], T2=t["T2"],
        T2dash=0.03, D=0.0, voxel_size=0.005,
    )


def measure(seq0, inv_idx, info, ti_s, objs, nx):
    set_react_ti(seq0, inv_idx, info, torch.tensor(float(ti_s)))
    with torch.no_grad():
        return {n: float(react_dc_signal(seq0, objs[n], nx).abs())
                for n in TISSUE_NAMES}


def objectives(s):
    return {"water": s["blood"] - s["muscle"],
            "fat": s["blood"] - s["muscle"] - s["fat"],
            "wall": s["blood"] - s["wall"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot-interval", type=float,
                    default=ReactParams().shot_interval,
                    help="recovery between preparations [s]")
    ap.add_argument("--lo", type=float, default=5.0, help="sweep start [ms]")
    ap.add_argument("--hi", type=float, default=200.0, help="sweep end [ms]")
    ap.add_argument("--step", type=float, default=5.0, help="sweep step [ms]")
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    # Two dummy shots so the measured one starts from the steady state the real
    # scan reaches, rather than from fully relaxed magnetization.
    p = ReactParams(nx=16, ny=8, tfe_factor=8, dummy_shots=2,
                    shot_interval=args.shot_interval)
    seq0, inv_idx, info = import_react_for_optimization(p, system)
    objs = {n: voxel(n) for n in TISSUE_NAMES}

    base, pos = info[inv_idx[0]]
    min_ti = float(base.sum() - base[pos]) * 1e3        # inversion pulse + spoiler [ms]
    grid = sorted(set(np.arange(args.lo, args.hi + 1e-9, args.step).tolist())
                  | set(PUBLISHED))
    print(f"shot interval {p.shot_interval:.1f} s, T2-prep {p.t2prep_duration*1e3:.0f} ms, "
          f"{p.t2prep_refocus} refocusing, steady state after 2 dummy shots")
    print(f"shortest TI this implementation can play: {min_ti:.1f} ms (inversion "
          f"pulse + spoiler); rows below it are not simulated\n")
    print("analytic nulling time after the T2-prep (T1*ln(1+E2)), from "
          "TISSUE_PROPERTIES:")
    for n in TISSUE_NAMES:
        t = TISSUES[n]
        print(f"  {n:7s} T1={t['T1']*1e3:5.0f} ms  T2={t['T2']*1e3:4.0f} ms  -> "
              f"{null_time_after_t2prep(t['T1'], t['T2'], p.t2prep_duration)*1e3:6.1f} ms")
    print()

    head = (f"{'TI [ms]':>8}{'blood':>8}{'wall':>8}{'muscle':>8}{'fat':>8}"
            f"{'water':>9}{'fat-obj':>9}{'wall-obj':>9}  note")
    print(head)
    print("-" * len(head))
    rows = []
    for ti in grid:
        if ti < min_ti:
            note = PUBLISHED.get(round(ti, 1), "")
            print(f"{ti:>8.1f}{'-- shorter than the inversion pulse + spoiler --':>66}"
                  f"  {note}")
            continue
        s = measure(seq0, inv_idx, info, ti * 1e-3, objs, p.nx)
        o = objectives(s)
        rows.append((ti, s, o))
        note = PUBLISHED.get(round(ti, 1), "")
        print(f"{ti:>8.1f}{s['blood']:>8.4f}{s['wall']:>8.4f}{s['muscle']:>8.4f}"
              f"{s['fat']:>8.4f}{o['water']:>9.4f}{o['fat']:>9.4f}{o['wall']:>9.4f}  {note}")

    print("\noptimum TI per objective (grid resolution "
          f"{args.step:g} ms):")
    best = {}
    for k in ("water", "fat", "wall"):
        ti, s, o = max(rows, key=lambda r: r[2][k])
        best[k] = (ti, o[k])
        # flatness: how wide is the region within 95% of the maximum?
        within = [r[0] for r in rows if r[2][k] >= 0.95 * o[k]]
        print(f"  {k:6s}: TI = {ti:6.1f} ms (contrast {o[k]:.4f}); "
              f"within 95% of max for {min(within):.0f}-{max(within):.0f} ms")

    cur = ReactParams().resolved_ti() * 1e3
    s_cur = measure(seq0, inv_idx, info, cur * 1e-3, objs, p.nx)
    o_cur = objectives(s_cur)
    print(f"\ncurrent default TI = {cur:.1f} ms:")
    for k in ("water", "fat", "wall"):
        print(f"  {k:6s}: {o_cur[k]:.4f}  ({o_cur[k]/best[k][1]*100:5.1f}% of the "
              f"{k}-objective optimum at {best[k][0]:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
