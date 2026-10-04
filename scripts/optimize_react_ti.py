#!/usr/bin/env python
"""Sweep REACT's inversion time, scored over the whole shot (MRzero, 0.55T).

What changed from the first version of this script
--------------------------------------------------
The first version scored the single k-space-centre sample. That sample is acquired
in TR 0 of the first shot, so it measured ``|Mz(TI)|`` and nothing about how the
prepared magnetization decays along the ~20 TRs of a shot. It also simulated a toy
shot (``nx=16, ny=8, tfe=8``) shorter than the real one. Both are fixed:

* the real :class:`ReactParams` geometry (``nx=128, ny=120``, 20-line trains,
  12 dummy shots so the steady state is reached);
* the contrast is the amplitude at the centre of a uniform **disc** vessel,
  ``|sum_k S(k) D(k)| / sum_k D(k)`` (see ``pyboost.diffopt.react_object_signal``),
  for two radii taken from ultrasound reference values for healthy adults
  (abstracts read only; none is MRI or 0.55T):

    R = 3.0 mm  common carotid, lumen ~6 mm  (Limbu 2006: 5.78-5.86 mm, n = 123;
                Ojaare 2021: 6.28-6.39 mm, n = 400; Nikolenko 2024: 5.5-5.7 mm)
    R = 2.2 mm  internal carotid, lumen ~4.4 mm (Ojaare 2021: 4.61-4.63 mm;
                Nikolenko 2024: up to 4.1-4.2 mm)

The old single-sample metric is printed alongside (``dc``) so the effect of the
change is visible rather than asserted.

Objectives (tissues simulated one at a time, i.e. an ideal Dixon separation)
* ``water`` = blood - muscle     (fat removed by Dixon; the REACT case)
* ``fat``   = blood - muscle - fat
* ``wall``  = blood - vessel wall

Limits: 2D disc (a tube along z in a 3D slab would weigh only kz = 0), tissue
values from ``TISSUE_PROPERTIES`` (untraced), block inversion (MRzero cannot
simulate the adiabatic pulse; on the scanner the effective TI of the sech pulse
is shorter by roughly half its duration).

Usage
-----
    python scripts/optimize_react_ti.py [--step 10] [--shot-interval 3.0] [--ablation]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MRzeroCore as mr0
from pyboost import ReactParams, scanner_055T, build_react_sequence
from pyboost.diffopt import (import_react_for_optimization, set_react_ti,
                             react_signals, locate_react_inversion)
from pyboost.params import null_time_after_t2prep
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES

TISSUE_NAMES = ("blood", "wall", "muscle", "fat")
RADII = {"3.0": 3.0e-3, "2.2": 2.2e-3}
PUBLISHED = {7.8: "original REACT, 1.5T (Erdem 2025)",
             12.2: "MTC-REACT shortest, 1.5T (Erdem 2025)",
             70.0: "Isaak 2021 Table 1, 1.5T"}


def voxel(name: str):
    t = TISSUES[name]
    # Small voxel: MRzero applies the voxel's sinc envelope along the readout,
    # which would zero the outer samples of every line (see react_object_signal).
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=t["PD"], T1=t["T1"], T2=t["T2"],
        T2dash=0.03, D=0.0, voxel_size=1e-4,
    )


def measure(seq0, objs, nx):
    """{'dc': {tissue: |s|}, 'R3.0': {...}, 'R2.2': {...}} from one run per tissue."""
    out = {"dc": {}, **{f"R{k}": {} for k in RADII}}
    with torch.no_grad():
        for n, obj in objs.items():
            dc, areas = react_signals(seq0, obj, nx, list(RADII.values()))
            out["dc"][n] = float(dc.abs())
            for k, r in RADII.items():
                out[f"R{k}"][n] = float(areas[r].abs())
    return out


def objectives(s):
    return {"water": s["blood"] - s["muscle"],
            "fat": s["blood"] - s["muscle"] - s["fat"],
            "wall": s["blood"] - s["wall"]}


def import_variant(p, system, use_t2prep, use_inversion, path="/tmp/react_variant.seq"):
    p = replace(p, inversion_kind="block")
    seq = build_react_sequence(p, system, use_t2prep=use_t2prep,
                               use_inversion=use_inversion)
    seq.write(path)
    return mr0.Sequence.import_file(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot-interval", type=float,
                    default=ReactParams().shot_interval,
                    help="recovery between preparations [s]")
    ap.add_argument("--lo", type=float, default=5.0, help="sweep start [ms]")
    ap.add_argument("--hi", type=float, default=200.0, help="sweep end [ms]")
    ap.add_argument("--step", type=float, default=10.0, help="sweep step [ms]")
    ap.add_argument("--dummy-shots", type=int, default=12)
    ap.add_argument("--ablation", action="store_true",
                    help="also isolate T2-prep and inversion (no prep, T2-prep "
                         "only, IR only, and T2-prep + IR at 12/84/155 ms)")
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    p = ReactParams(dummy_shots=args.dummy_shots, shot_interval=args.shot_interval)
    seq0, inv_idx, info = import_react_for_optimization(p, system)
    objs = {n: voxel(n) for n in TISSUE_NAMES}

    base, pos = info[inv_idx[0]]
    min_ti = float(base.sum() - base[pos]) * 1e3
    grid = sorted(set(np.arange(args.lo, args.hi + 1e-9, args.step).tolist())
                  | set(PUBLISHED))
    print(f"shot interval {p.shot_interval:.1f} s, T2-prep {p.t2prep_duration*1e3:.0f} ms, "
          f"{p.t2prep_refocus} refocusing, {p.n_shots} shots of "
          f"{-(-p.n_encodes // p.n_shots)} lines, {p.dummy_shots} dummy shots")
    print(f"shortest TI this implementation can play: {min_ti:.1f} ms (inversion "
          f"pulse + spoiler); rows below it are not simulated\n")
    print("analytic nulling time after the T2-prep (T1*ln(1+E2)):")
    for n in TISSUE_NAMES:
        t = TISSUES[n]
        print(f"  {n:7s} T1={t['T1']*1e3:5.0f} ms  T2={t['T2']*1e3:4.0f} ms  -> "
              f"{null_time_after_t2prep(t['T1'], t['T2'], p.t2prep_duration)*1e3:6.1f} ms")
    print()

    metrics = ["dc"] + [f"R{k}" for k in RADII]
    rows = []
    head = (f"{'TI [ms]':>8}  " + "  ".join(f"{m:>5}:{'water':>8}{'fat':>8}{'wall':>8}"
                                           for m in metrics) + "  note")
    print(head)
    print("-" * len(head))
    for ti in grid:
        note = PUBLISHED.get(round(ti, 1), "")
        if ti < min_ti:
            print(f"{ti:>8.1f}  -- shorter than the inversion pulse + spoiler --  {note}")
            continue
        set_react_ti(seq0, inv_idx, info, torch.tensor(ti * 1e-3))
        m = measure(seq0, objs, p.nx)
        o = {k: objectives(v) for k, v in m.items()}
        rows.append((ti, m, o))
        print(f"{ti:>8.1f}  " + "  ".join(
            f"{k:>5}:{o[k]['water']:>8.4f}{o[k]['fat']:>8.4f}{o[k]['wall']:>8.4f}"
            for k in metrics) + f"  {note}", flush=True)

    print("\noptimum TI per metric and objective "
          f"(grid {args.step:g} ms):")
    best = {}
    for mk in metrics:
        for ob in ("water", "fat", "wall"):
            ti, _, o = max(rows, key=lambda r: r[2][mk][ob])
            v = o[mk][ob]
            within = [r[0] for r in rows if r[2][mk][ob] >= 0.95 * v]
            best[(mk, ob)] = (ti, v)
            print(f"  {mk:>5} {ob:6s}: TI = {ti:6.1f} ms (value {v:.4f}); "
                  f"within 95% for {min(within):.0f}-{max(within):.0f} ms")

    cur = ReactParams().resolved_ti() * 1e3
    set_react_ti(seq0, inv_idx, info, torch.tensor(cur * 1e-3))
    m = measure(seq0, objs, p.nx)
    o = {k: objectives(v) for k, v in m.items()}
    print(f"\ncurrent default TI = {cur:.1f} ms, as a share of each optimum:")
    for mk in metrics:
        print("  " + f"{mk:>5}: " + "  ".join(
            f"{ob} {o[mk][ob]/best[(mk, ob)][1]*100:5.1f}% (opt {best[(mk, ob)][0]:.0f} ms)"
            for ob in ("water", "fat", "wall")))

    if args.ablation:
        print("\nablation: blood - muscle, by preparation")
        print(f"{'preparation':<26}" + "".join(f"{m:>10}" for m in metrics))
        cases = [("no preparation", False, False, None),
                 ("T2-prep only", True, False, None),
                 ("IR only, TI 155 ms", False, True, 155.0),
                 ("T2-prep + IR, TI 12 ms", True, True, 12.0),
                 ("T2-prep + IR, TI 84 ms", True, True, 84.2),
                 ("T2-prep + IR, TI 155 ms", True, True, 155.0)]
        for label, t2, ir, ti in cases:
            s0 = import_variant(p, system, t2, ir)
            if ir:
                idx, inf = locate_react_inversion(s0, p.flip_angle)
                set_react_ti(s0, idx, inf, torch.tensor(ti * 1e-3))
            m = measure(s0, {n: objs[n] for n in ("blood", "muscle")}, p.nx)
            print(f"{label:<26}" + "".join(
                f"{m[mk]['blood'] - m[mk]['muscle']:>10.4f}" for mk in metrics),
                flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
