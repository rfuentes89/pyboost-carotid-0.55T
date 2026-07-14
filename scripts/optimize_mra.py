#!/usr/bin/env python
"""Sweep the T2-prep echo time to optimize bright-blood MRA contrast at 0.55T.

Blood keeps signal through a T2-prep (long T2 ~263 ms) while muscle (short T2
~55 ms) decays, so a longer TE improves the blood/muscle *ratio* — but blood also
loses signal, so the absolute blood-to-muscle *contrast* S_blood - S_muscle (the
CNR driver) peaks at an intermediate TE. This script finds that operating point.

Relaxation values come from ``pyboost.phantom.TISSUE_PROPERTIES`` (single source
of truth). Muscle at 0.55T is T1=450 ms / T2=55 ms.

Usage
-----
    python scripts/optimize_mra.py [--out mra_t2prep_sweep.png]
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

TISSUES = {k: TISSUE_PROPERTIES[k] for k in ("blood", "wall", "muscle", "fat")}
COLORS = {"blood": "#c0392b", "wall": "#2980b9", "muscle": "#27ae60",
          "fat": "#e67e22"}


def center_mag(signal, kspace, per):
    s = signal.detach().cpu().numpy().reshape(-1)[:per]
    k = kspace.detach().cpu().numpy()[:per]
    idx = int(np.argmin(np.abs(k[:, 0]) + np.abs(k[:, 1])))
    return abs(s[idx])


def tissue_signals(seq0, p):
    per = p.im_segments * p.nx
    out = {}
    for name, tp in TISSUES.items():
        obj = mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"],
                                     T1=tp["T1"], T2=tp["T2"], T2dash=0.03,
                                     D=0.0, voxel_size=5e-3)
        signal, kspace = mr0.util.simulate(seq0, obj)
        out[name] = center_mag(signal, kspace, per)
    return out


def build(te, system):
    # Single shot (im_segments == ny) + centric so the central line reads the
    # prepared bright-blood contrast.
    p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                    centric=True, t2prep_duration=te)
    seq = build_mra_sequence(p, system, use_t2prep=te > 0, use_fatsat=True,
                             add_trigger=False)
    seq.write("/tmp/mra_opt.seq")
    return mr0.Sequence.import_file("/tmp/mra_opt.seq"), p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mra_t2prep_sweep.png")
    args = ap.parse_args()
    system = scanner_055T()

    te_ms = np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150])
    sig = {n: [] for n in TISSUES}
    for te in te_ms:
        seq0, p = build(te * 1e-3, system)
        s = tissue_signals(seq0, p)
        for n in TISSUES:
            sig[n].append(s[n])
        print(f"T2prep TE={te:4d} ms  " + "  ".join(
            f"{n}={s[n]:.3f}" for n in TISSUES) +
            f"  blood/muscle={s['blood']/max(s['muscle'],1e-9):.2f}")
    for n in TISSUES:
        sig[n] = np.array(sig[n])

    cnr = sig["blood"] - sig["muscle"]            # absolute blood-muscle contrast
    ratio = sig["blood"] / np.maximum(sig["muscle"], 1e-9)
    best = int(np.argmax(cnr))
    te_opt = te_ms[best]
    print(f"\nOptimal T2prep TE ~ {te_opt} ms:  blood-muscle contrast="
          f"{cnr[best]:.3f}  blood/muscle={ratio[best]:.2f}  "
          f"blood/fat={sig['blood'][best]/max(sig['fat'][best],1e-9):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for n in TISSUES:
        ax1.plot(te_ms, sig[n], "-o", ms=4, color=COLORS[n], label=n)
    ax1.axvline(te_opt, color="0.4", ls="--", lw=1)
    ax1.set_xlabel("T2-prep TE [ms]"); ax1.set_ylabel("bright-blood signal")
    ax1.set_title("Signal vs T2-prep TE per tissue"); ax1.legend()
    ax2.plot(te_ms, cnr, "-o", color="#8e44ad", label="blood - muscle (CNR)")
    ax2.plot(te_ms, ratio / ratio.max() * cnr.max(), "-s", ms=4, color="0.5",
             label="blood/muscle (scaled)")
    ax2.axvline(te_opt, color="0.4", ls="--", lw=1,
                label=f"optimal TE ~ {te_opt} ms")
    ax2.set_xlabel("T2-prep TE [ms]"); ax2.set_ylabel("contrast")
    ax2.set_title("Blood-to-muscle contrast peaks at intermediate TE")
    ax2.legend()
    fig.suptitle("Bright-blood MRA T2-prep optimization @ 0.55T "
                 "(muscle T1=450/T2=55 ms)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
