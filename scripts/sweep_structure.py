#!/usr/bin/env python
"""Discrete sweeps of structural (integer) MRA parameters at 0.55T.

`iNAV_lines` and the number of imaging segments are *integer* counts that change
the sequence structure (number of repetitions), so they have no gradient and are
swept discretely -- unlike the flip / T2-prep TE, which are continuous and
optimized differentiably elsewhere.

* **iNAV_lines** -- length of the bSSFP start-up ramp; more catalyzation drives a
  cleaner steady state at the cost of dead time before acquisition.
* **im_segments** -- k-space lines per heartbeat. Fewer segments -> more shots
  (heartbeats, longer scan) but a shorter readout window; more segments -> fewer
  shots but a longer window (fat recovers, transient decays before k-space edge).

Usage
-----
    python scripts/sweep_structure.py [--out structure_sweep.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MRzeroCore as mr0
from pyboost import build_mra_sequence, BoostParams, scanner_055T
from pyboost.phantom import TISSUE_PROPERTIES

TISSUES = {k: TISSUE_PROPERTIES[k] for k in ("blood", "muscle")}


def central_signal(seq0, tp, per):
    obj = mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                 T2=tp["T2"], T2dash=0.03, D=0.0, voxel_size=5e-3)
    signal, kspace = mr0.util.simulate(seq0, obj)
    s = signal.detach().cpu().numpy().reshape(-1)
    k = kspace.detach().cpu().numpy()
    idx = int(np.argmin(np.abs(k[:, 0]) + np.abs(k[:, 1])))
    return abs(s[idx])


def contrast(system, **kw):
    p = BoostParams(nx=16, dummy_heart_beats=1, centric=True,
                    t2prep_duration=0.06, im_flip_angle=(90.0, 80.0), **kw)
    seq = build_mra_sequence(p, system, use_t2prep=True, use_fatsat=True,
                             add_trigger=False)
    seq.write("/tmp/struct.seq")
    seq0 = mr0.Sequence.import_file("/tmp/struct.seq")
    per = len(seq0)  # single global search over the whole acquisition
    b = central_signal(seq0, TISSUES["blood"], per)
    m = central_signal(seq0, TISSUES["muscle"], per)
    return b, m, p.n_shots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="structure_sweep.png")
    args = ap.parse_args()
    system = scanner_055T()

    # --- iNAV_lines sweep (ny fixed, single shot) -----------------------
    inav_vals = [0, 2, 4, 6, 8, 10, 12]
    inav_c = []
    for k in inav_vals:
        b, m, _ = contrast(system, ny=16, im_segments=16, inav_lines=k)
        inav_c.append(b - m)
        print(f"iNAV_lines={k:2d}  blood-muscle contrast={b-m:+.3f}")

    # --- im_segments sweep (ny=24, divisors) ----------------------------
    ny = 24
    seg_vals = [4, 6, 8, 12, 24]
    seg_c, seg_shots = [], []
    for s in seg_vals:
        b, m, n_shots = contrast(system, ny=ny, im_segments=s, inav_lines=6)
        seg_c.append(b - m); seg_shots.append(n_shots)
        print(f"im_segments={s:2d}  n_shots={n_shots}  contrast={b-m:+.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(inav_vals, inav_c, "-o", color="#2980b9")
    ax1.set_xlabel("iNAV_lines (start-up ramp length)")
    ax1.set_ylabel("blood - muscle contrast")
    ax1.set_title("iNAV ramp length (discrete)")
    ax2.plot(seg_vals, seg_c, "-o", color="#c0392b")
    for x, y, ns in zip(seg_vals, seg_c, seg_shots):
        ax2.annotate(f"{ns} shots", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="0.4")
    ax2.set_xlabel(f"im_segments (ny={ny})")
    ax2.set_ylabel("blood - muscle contrast")
    ax2.set_title("k-space segments per heartbeat (discrete)")
    fig.suptitle("Structural (integer) MRA parameter sweeps @ 0.55T")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
