#!/usr/bin/env python
"""Carotid image WITHOUT vs WITH blood flow (complete-washout model).

Demonstrates the point the static simulation could not: with a static phantom the
lumen and the vessel wall behave almost identically (blood T1 1122 ms vs wall
750 ms are too close for an inversion to separate them), whereas as soon as the
blood is allowed to flow -- i.e. the lumen is refilled with spins that never saw
the preparation -- the lumen decouples from the wall.

Flow is modelled in the complete through-plane washout limit by
``pyboost.flow.simulate_with_flow``: stationary tissues run the full sequence,
blood runs a preparation-stripped copy of it, and the two k-spaces are added.

Note on the direction of the effect: with fresh blood arriving at thermal
equilibrium (blood coming from outside the prepared region) this is *inflow
enhancement* -- the lumen gets BRIGHTER than the wall. Turning that into a dark
lumen (true black-blood) additionally needs upstream tagging, i.e. a DIR module
that leaves inflowing blood inverted; that is future work.

Usage
-----
    python scripts/image_flow_blackblood.py [--out flow_vs_static.png]
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
from pyboost import build_boost_sequence, BoostParams, scanner_055T, simulate_with_flow
from pyboost.phantom import carotid_phantom_maps, to_mrzero_phantom, CarotidGeometry
from scripts.image_carotid_phantom import grid_and_reconstruct, _downsample_labels


def black_blood(bright, reference):
    """Phase-sensitive combination of the two BOOST contrasts."""
    ref_mag = np.abs(reference)
    psir = np.real(bright * np.conj(reference)) / (ref_mag + 1e-3 * ref_mag.max())
    return np.abs(psir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="flow_vs_static.png")
    ap.add_argument("--matrix", type=int, default=48)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--ti", type=float, default=0.09, help="inversion time [s]")
    args = ap.parse_args()

    fov = 0.06
    maps = carotid_phantom_maps(CarotidGeometry(fov=fov, matrix=96))
    system = scanner_055T()
    n = args.matrix
    p = BoostParams(fov=fov, nx=n, ny=n, im_segments=args.segments,
                    dummy_heart_beats=1, centric=True, inversion_kind="block",
                    ir_inversion_time=args.ti)
    seq = build_boost_sequence(p, system, add_trigger=False)
    seq.write("/tmp/flow_bb.seq")
    print(f"sequence {n}x{n}, TI {args.ti*1e3:.0f} ms, centric, block inversion")

    # --- baseline: static phantom (no flow) -----------------------------
    print("simulating static (no flow) ...")
    obj = to_mrzero_phantom(maps, fov)
    seq0 = mr0.Sequence.import_file("/tmp/flow_bb.seq")
    sig_s, k_s = mr0.util.simulate(seq0, obj)
    bright_s, ref_s = grid_and_reconstruct(sig_s, k_s, p, fov, p.slice_thickness)

    # --- flowing blood (complete washout) -------------------------------
    print("simulating with flow (complete washout) ...")
    sig_f, k_f, _, _ = simulate_with_flow("/tmp/flow_bb.seq", maps, fov, p)
    bright_f, ref_f = grid_and_reconstruct(sig_f, k_f, p, fov, p.slice_thickness)

    imgs = {"static": black_blood(bright_s, ref_s).T,
            "flow": black_blood(bright_f, ref_f).T}
    bright = {"static": np.abs(bright_s).T, "flow": np.abs(bright_f).T}

    lab = _downsample_labels(maps["label"], n)
    def stats(img):
        m = lambda l: img[lab == l].mean() if np.any(lab == l) else float("nan")
        return m(1), m(2)                                 # lumen, wall

    print(f"\n{'case':8s} {'lumen':>10s} {'wall':>10s} {'lumen/wall':>12s}")
    for case in ("static", "flow"):
        lu, wa = stats(bright[case])
        print(f"{case:8s} {lu:10.4g} {wa:10.4g} {lu/max(wa,1e-9):12.2f}")

    # --- figure -----------------------------------------------------------
    vmax = max(bright["static"].max(), bright["flow"].max())
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(maps["label"], cmap="viridis"); ax[0].set_title("phantom (tissues)")
    for a, case in zip(ax[1:], ("static", "flow")):
        lu, wa = stats(bright[case])
        a.imshow(bright[case], cmap="gray", vmax=vmax)
        a.set_title(f"{'no flow (static)' if case=='static' else 'with flow (washout)'}\n"
                    f"lumen/wall = {lu/max(wa,1e-9):.2f}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Carotid @ 0.55T — blood flow decouples the lumen from the wall "
                 "(T2prep+IR contrast)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
