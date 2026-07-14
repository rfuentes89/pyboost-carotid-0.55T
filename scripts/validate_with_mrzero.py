#!/usr/bin/env python
"""Cross-validate the executable .seq against the Koma reference contrast.

Hybrid loop: PyPulseq builds the sequence -> the .seq is imported into
MRzero-Core -> a 0.55T carotid tissue model is simulated -> the two BOOST
contrasts and their phase-sensitive subtraction are compared per tissue.

This reproduces the *contrast* experiment of the Koma reference (RR_sim.jl),
which simulated the magnetization of each tissue without spatial encoding. Here
we read the central-k-space signal of each contrast, which is proportional to
the prepared magnetization -- so the numbers are directly comparable in spirit
to Koma's ``signal_1hb`` / ``signal_2hb`` and their subtraction (RR_sim.jl:462).

Tissue relaxation values are the 0.55T literature values used by the Koma
``carotid_phantom`` (RR_sim.jl:196-207); muscle is added as the dominant
surrounding tissue.

Usage
-----
    python scripts/validate_with_mrzero.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MRzeroCore as mr0
from pyboost import build_boost_sequence, BoostParams, scanner_055T
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES

# 0.55T tissue model (PD, T1[s], T2[s]) -- single source of truth in
# pyboost/phantom.py. Blood / wall from RR_sim.jl:199-202; muscle at 0.55T.


def contrast_center_signal(signal, kspace, n_shots, im_segments, nx):
    """Central-k-space |signal| for each of the two contrasts.

    Returns (bright_blood, reference). The acquisition order is, per shot,
    contrast-0 (im_segments*nx samples) then contrast-1 (im_segments*nx samples).
    Within each contrast block the sample nearest k=0 is the echo peak.
    """
    sig = signal.detach().cpu().numpy().reshape(-1)
    k = kspace.detach().cpu().numpy()
    per_contrast = im_segments * nx
    out = []
    for contrast in (0, 1):
        vals = []
        for shot in range(n_shots):
            start = (shot * 2 + contrast) * per_contrast
            block = slice(start, start + per_contrast)
            kb = k[block]
            radius = np.abs(kb[:, 0]) + np.abs(kb[:, 1])  # |kx|+|ky|
            vals.append(np.abs(sig[block][np.argmin(radius)]))
        out.append(float(np.mean(vals)))
    return out[0], out[1]


def main() -> int:
    system = scanner_055T()
    # Moderate matrix keeps the simulation quick while exercising the real
    # segmented structure (2 contrasts x n_shots heartbeats).
    p = BoostParams(nx=32, ny=32, im_segments=16, dummy_heart_beats=2)
    seq = build_boost_sequence(p, system, add_trigger=False)
    seq_path = "/tmp/boost_validate.seq"
    seq.write(seq_path)
    seq0 = mr0.Sequence.import_file(seq_path)
    print(f"Imported {seq_path}: {len(seq0)} repetitions, "
          f"n_shots={p.n_shots}, im_segments={p.im_segments}\n")

    header = f"{'tissue':8s}{'bright-blood':>14s}{'reference':>12s}{'|BB=c0-c1|':>12s}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, tp in TISSUES.items():
        obj = mr0.CustomVoxelPhantom(
            pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"], T2=tp["T2"],
            T2dash=0.03, D=0.0, voxel_size=0.005,
        )
        signal, kspace = mr0.util.simulate(seq0, obj)
        c0, c1 = contrast_center_signal(signal, kspace, p.n_shots,
                                        p.im_segments, p.nx)
        bb = abs(c0 - c1)
        results[name] = (c0, c1, bb)
        print(f"{name:8s}{c0:>14.4f}{c1:>12.4f}{bb:>12.4f}")

    # Physics checks that the prep pulses actually do their job at 0.55T:
    #  (1) Fat saturation: fat is far dimmer in the reference contrast (which
    #      carries the FatSat pulse) than in the T2prep+IR contrast.
    #  (2) Adiabatic inversion: the inversion contrast (c0) darkens long-T1
    #      blood relative to the non-inverted reference (c1) -- at TI=90 ms and
    #      blood T1=1122 ms the inverted magnetization is still strongly
    #      negative, the mechanism BOOST exploits for black-blood.
    print()
    fat_c0, fat_c1, _ = results["fat"]
    bld_c0, bld_c1, _ = results["blood"]
    fatsat_ratio = fat_c0 / max(fat_c1, 1e-6)
    print(f"FatSat suppression (fat c0/c1) = {fatsat_ratio:.2f}x "
          "(>1 means fat is saturated in the reference contrast)")
    print(f"Inversion darkening (blood c0/c1) = {bld_c0 / max(bld_c1, 1e-6):.2f} "
          "(<1 means the IR darkens blood)")
    checks = {
        "fat saturated in reference contrast": fat_c1 < fat_c0,
        "inversion darkens blood": bld_c0 < bld_c1,
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = all(checks.values())
    print("\nPrep-pulse contrast is physically consistent with the 0.55T BOOST "
          "reference." if ok else
          "\nWARNING: unexpected contrast -- inspect prep timing/TI.")
    print("For a quantitative match, run the Koma reference (RR_sim.jl) and "
          "compare against 'Simulaciones RR optimo/BB.png'.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
