#!/usr/bin/env python
"""Optimize REACT's shot interval (preparation-to-preparation recovery) at 0.55T.

Why this parameter is different from BOOST's
--------------------------------------------
In a cardiac-gated sequence the shot spacing *is* the RR interval: the scan time
is fixed at ``n_shots x RR`` and there is nothing to trade. REACT is untriggered,
so the interval is genuinely free -- and it is the parameter that decides how
much of blood's magnetization has come back before the next T2-prep. Blood has
the longest T1 in the neck (1122 ms at 0.55T), so it is the tissue that suffers
most from a short interval, and it is exactly the tissue REACT wants brightest.

Why the objective is efficiency, not contrast
---------------------------------------------
Contrast rises monotonically with the interval, so optimizing it alone would
push the answer to infinity. The number of shots is fixed by k-space, so a
longer interval buys contrast with pure scan time. The honest figure of merit is
therefore contrast per unit sqrt(time)::

    efficiency = (S_blood - S_muscle - S_fat) / sqrt(n_shots * T_shot)

``n_shots`` is a constant here, so it scales the efficiency but cannot move its
maximum; the optimization runs on ``sqrt(T_shot)`` and the reported optimum is
independent of matrix size.

Why a sweep and not plain gradient descent
------------------------------------------
The efficiency landscape is **not unimodal**. There is a local maximum near
700 ms -- the point where fat has recovered just enough for the inversion to
null it properly -- then a dip around 900-1100 ms where muscle recovers faster
than blood, and only past ~1.4 s does blood's slow T1 recovery take over and
drive efficiency to its true maximum near 3 s. Gradient descent started at the
1 s default converges into the 700 ms basin and reports it as the answer; it is
about 24% worse than the global optimum. So this script sweeps coarsely first
and only then refines locally.

Analytic anchor
---------------
If contrast simply followed blood's recovery, ``C(T) ~ 1 - exp(-T/T1)``, then
maximizing ``C/sqrt(T)`` gives ``2x = exp(x) - 1`` with ``x = T/T1``, i.e.
``x ~ 1.26`` and ``T ~ 1.4 s``. The simulation lands considerably higher (~3 s)
because the T2-prep and inversion make the real contrast rise more slowly than
that pure-recovery model: the anchor is a lower bound on the answer, not a
prediction of it.

Usage
-----
    python scripts/optimize_react_shot_interval.py [--refine] [--max-scan 300]
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
                             react_dc_signal)
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES


def voxel(name: str):
    t = TISSUES[name]
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=t["PD"], T1=t["T1"], T2=t["T2"],
        T2dash=0.03, D=0.0, voxel_size=0.005,
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


def measure(seq0, rec_idx, info, overhead, interval, objs, nx):
    """Contrast and efficiency at a given shot interval. Differentiable."""
    set_react_shot_interval(seq0, rec_idx, info, overhead, interval)
    blood = react_dc_signal(seq0, objs["blood"], nx).abs()
    muscle = react_dc_signal(seq0, objs["muscle"], nx).abs()
    fat = react_dc_signal(seq0, objs["fat"], nx).abs()
    contrast = blood - muscle - fat
    return contrast / torch.sqrt(interval), contrast, blood, muscle, fat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refine", action="store_true",
                    help="polish the sweep winner with a few gradient steps")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lo", type=float, default=500.0, help="sweep start [ms]")
    ap.add_argument("--hi", type=float, default=6000.0, help="sweep end [ms]")
    ap.add_argument("--n", type=int, default=19, help="sweep points")
    ap.add_argument("--max-scan", type=float, default=None,
                    help="scan-time ceiling [s]; also report the best interval "
                         "that fits it for the full protocol")
    ap.add_argument("--dummy-shots", type=int, default=2,
                    help="preparations before the measured one (steady state)")
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    p = ReactParams(nx=16, ny=8, tfe_factor=8, dummy_shots=args.dummy_shots)
    seq0, inv_idx, inv_info = import_react_for_optimization(p, system)
    set_react_ti(seq0, inv_idx, inv_info, torch.tensor(float(p.resolved_ti())))
    rec_idx, rec_info, overhead = locate_react_recovery(seq0, p.flip_angle)
    if not rec_idx:
        print("No recovery repetitions found -- need at least 2 shots.")
        return 1
    objs = {n: voxel(n) for n in ("blood", "muscle", "fat")}

    t1_blood = TISSUES["blood"]["T1"]
    anchor = analytic_optimum(t1_blood)
    print(f"blood T1 = {t1_blood*1e3:.0f} ms, shot overhead = {overhead*1e3:.0f} ms "
          f"({p.tfe_factor} TRs + prep)")
    print(f"analytic anchor (pure T1 recovery): {anchor*1e3:.0f} ms")
    print(f"current default: {ReactParams().shot_interval*1e3:.0f} ms\n")

    # --- Coarse sweep: the landscape has a decoy local maximum ------------
    print(f"{'T_shot [ms]':>12}{'blood':>9}{'muscle':>9}{'fat':>9}"
          f"{'contrast':>10}{'efficiency':>12}")
    grid = np.linspace(args.lo, args.hi, args.n)
    rows = []
    for ms in grid:
        with torch.no_grad():
            eff, c, b, m, f = measure(seq0, rec_idx, rec_info, overhead,
                                      torch.tensor(float(ms) * 1e-3), objs, p.nx)
        rows.append((float(ms) * 1e-3, eff.item(), c.item(), b.item()))
        print(f"{ms:>12.0f}{b.item():>9.4f}{m.item():>9.4f}{f.item():>9.4f}"
              f"{c.item():>10.4f}{eff.item():>12.4f}")

    best_t, best_eff, best_c, best_b = max(rows, key=lambda r: r[1])

    # --- Optional local refinement around the sweep winner ----------------
    if args.refine:
        t = torch.tensor(best_t, requires_grad=True)
        opt = torch.optim.Adam([t], lr=args.lr)
        for _ in range(args.steps):
            opt.zero_grad()
            eff, c, b, _, _ = measure(seq0, rec_idx, rec_info, overhead, t,
                                      objs, p.nx)
            (-eff).backward()
            if eff.item() > best_eff:
                best_t, best_eff, best_c, best_b = (float(t.item()), eff.item(),
                                                    c.item(), b.item())
            opt.step()
            with torch.no_grad():
                t.clamp_(min=overhead + 0.05, max=args.hi * 1e-3)
        print(f"\nrefined around the sweep winner -> {best_t*1e3:.0f} ms")

    with torch.no_grad():
        eff_d, c_d, b_d, _, _ = measure(seq0, rec_idx, rec_info, overhead,
                                        torch.tensor(1.0), objs, p.nx)

    # Flag the decoy so a future reader does not "fix" this back to descent.
    early = [r for r in rows if r[0] < 1.2]
    if early:
        loc_t, loc_eff, _, _ = max(early, key=lambda r: r[1])
        if loc_eff < best_eff and loc_t < best_t:
            print(f"\nNOTE: there is a local efficiency maximum at "
                  f"{loc_t*1e3:.0f} ms ({loc_eff:.4f}), {(1-loc_eff/best_eff)*100:.0f}% "
                  f"below the global one. Gradient descent from the 1 s default "
                  f"lands there.")

    print(f"\nbest T_shot = {best_t*1e3:.0f} ms  "
          f"(= {best_t/t1_blood:.1f} x blood T1; analytic anchor "
          f"{anchor*1e3:.0f} ms is a lower bound)")
    print(f"  vs the 1000 ms default: contrast {c_d.item():.4f} -> {best_c:.4f} "
          f"({(best_c/c_d.item() - 1)*100:+.0f}%), blood {b_d.item():.4f} -> "
          f"{best_b:.4f} ({(best_b/b_d.item() - 1)*100:+.0f}%)")
    print(f"  efficiency {eff_d.item():.4f} -> {best_eff:.4f} "
          f"({(best_eff/eff_d.item() - 1)*100:+.0f}%)")

    # --- What it costs on the real protocol -------------------------------
    full = ReactParams()
    print(f"\nScan time for the full protocol ({full.ny}x{full.nz}, tfe "
          f"{full.tfe_factor} -> {full.n_shots} shots):")
    print(f"  at 1000 ms : {full.n_shots*1.0:6.1f} s")
    print(f"  at {best_t*1e3:.0f} ms : {full.n_shots*best_t:6.1f} s")
    if args.max_scan is not None:
        cap = args.max_scan / full.n_shots
        feasible = [r for r in rows if r[0] <= cap]
        if feasible:
            ct, ce, cc, _ = max(feasible, key=lambda r: r[2])
            print(f"  under a {args.max_scan:.0f} s ceiling the interval must stay "
                  f"below {cap*1e3:.0f} ms;\n  best contrast there is {cc:.4f} at "
                  f"{ct*1e3:.0f} ms.")
        else:
            print(f"  a {args.max_scan:.0f} s ceiling is not reachable: even the "
                  f"shortest swept interval needs "
                  f"{full.n_shots*rows[0][0]:.0f} s.")
    print("\nSet it with ReactParams(shot_interval=<seconds>).")
    print("Caveat: efficiency (contrast / sqrt(time)) is the right metric only if "
          "you would\nspend saved time on averages or more encodes. Under a hard "
          "scan-time ceiling,\nmaximize contrast subject to that ceiling instead "
          "-- see --max-scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
