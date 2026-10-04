#!/usr/bin/env python
"""Reconstruct a simulated REACT acquisition of the carotid phantom.

What this script can and cannot establish
-----------------------------------------
``phantom.py`` represents fat as a **B0 offset** (-79.6 Hz at 0.55T), because
MRzero has no multi-species model. To a two-point Dixon that is *the same data*
as water sitting off-resonance, so this phantom **cannot validate species
separation** -- an earlier version of this script reported "99.5% swapped voxels"
and that was not an algorithm failure, it was a question the phantom cannot
answer (it also used tfe_factor=64 instead of the design value, so fat was not
even nulled). The separation is validated on synthetic images with independently
known water, fat and field map: ``tests/test_dixon.py``.

What the simulation *can* show, and what is checked here:

1. **The reordering reconstructs.** Echo images come out as a sharp carotid
   cross-section, so echo demultiplexing and the centric order are right.
2. **REACT contrast on echo 1.** Blood brighter than muscle, fat suppressed by
   the inversion, at the design shot structure (tfe_factor=22).
3. **The reconstruction preserves complex phase between echoes.** Phase is the
   entire Dixon measurement, so this is the property that matters. It is tested
   with each tissue simulated *alone*: the signal is a sum over independent
   isochromats, so images superpose, and a tissue on its own must show exactly the
   designed inter-echo phase -- 0 deg for water-like tissue, 162.5 deg for the fat
   offset. Measured on the *combined* phantom instead, the weak tissue is
   contaminated by truncation ringing from its bright neighbours (muscle, the
   weakest, reads about -40 deg with a 55 deg spread); that is point-spread
   leakage, not a reconstruction error, and is printed for information only.

The Dixon output is shown for illustration only. The fat ring comes out labelled
as fat because the |psi|-near-zero prior prefers it, not because the phantom
proves anything about separation.

Usage
-----
    python scripts/recon_react_phantom.py [--out react_recon.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MRzeroCore as mr0
from pyboost import (build_react_sequence, ReactParams, scanner_055T,
                     kernel_report, fat_frequency)
from pyboost.dixon import conditioning, separate
from pyboost.phantom import (carotid_phantom_maps, to_mrzero_phantom,
                             CarotidGeometry, LABELS)
from pyboost.recon import kspace_from_signal, images_from_kspace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=int, default=64)
    ap.add_argument("--fov", type=float, default=60.0, help="FOV [mm]")
    # The carotid phantom is a 60 mm zoomed view, so delta_k is ~3x what a 200 mm
    # neck FOV gives, the readout-gradient area grows with it, and the slew-limited
    # rewinder no longer fits inside the 5.67 ms Dixon spacing at 300 Hz/pixel.
    # _build_kernel refuses rather than mistime the echoes, so bandwidth is raised.
    ap.add_argument("--bandwidth", type=float, default=400.0, help="Hz/pixel")
    ap.add_argument("--out", default=None, help="optional PNG")
    args = ap.parse_args()

    n, fov = args.matrix, args.fov * 1e-3
    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    # Design shot structure (tfe_factor and shot_interval are ReactParams defaults).
    p = ReactParams(nx=n, ny=n, fov=fov, readout_bandwidth=args.bandwidth,
                    dummy_shots=2, inversion_kind="block")
    rep = kernel_report(system, p)
    df = fat_frequency(system)
    cond = conditioning(rep["te1"], rep["te2"], df)
    designed_phase = 360.0 * df * rep["delta_te"]

    print(f"TE1 {rep['te1']*1e3:.2f} ms, TE2 {rep['te2']*1e3:.2f} ms, "
          f"dTE {rep['delta_te']*1e3:.3f} ms, TR {rep['tr']*1e3:.2f} ms")
    print(f"tfe_factor {p.tfe_factor} -> {p.n_shots} shots, shot interval "
          f"{p.shot_interval:.1f} s, TI {p.resolved_ti()*1e3:.1f} ms")
    print(f"designed water-fat phase {designed_phase:+.1f} deg "
          f"(noise x{cond.noise_amplification:.4f} vs 180 deg)\n")

    geom = CarotidGeometry(fov=fov, matrix=n, apply_fat_offset=True)
    maps = carotid_phantom_maps(geom)
    obj = to_mrzero_phantom(maps, fov)
    seq = build_react_sequence(p, system)
    seq.write("/tmp/react_recon.seq")
    seq0 = mr0.Sequence.import_file("/tmp/react_recon.seq")
    signal, _ = mr0.util.simulate(seq0, obj)

    k1, k2 = kspace_from_signal(signal, p)
    s1, s2 = (a[0] for a in images_from_kspace(k1, k2))

    label = maps["label"]
    regions = {name: label == LABELS[name]
               for name in ("blood", "wall", "muscle", "fat")}

    # --- 2. contrast on echo 1 (combined phantom) --------------------------
    mag = {k: float(np.abs(s1[m]).mean()) for k, m in regions.items()}
    # Informational: inter-echo phase on the combined phantom, leakage included.
    phase = {k: float(np.rad2deg(np.angle(np.sum(s2[m] * np.conj(s1[m])))))
             for k, m in regions.items()}

    print(f"{'region':8}{'N':>6}{'|echo1|':>10}{'phase, combined [deg]':>24}")
    print("-" * 48)
    for k, m in regions.items():
        print(f"{k:8}{m.sum():>6}{mag[k]:>10.4f}{phase[k]:>24.1f}")
    print("(combined-phantom phase is contaminated by neighbours' truncation "
          "ringing)\n")

    # --- 3. phase preservation: each tissue alone --------------------------
    def interecho_phase(name):
        sub = dict(maps)
        sub["PD"] = np.where(label == LABELS[name], maps["PD"], 0.0)
        sig, _ = mr0.util.simulate(seq0, to_mrzero_phantom(sub, fov))
        a1, a2 = (a[0] for a in images_from_kspace(*kspace_from_signal(sig, p)))
        m = regions[name]
        return float(np.rad2deg(np.angle(np.sum(a2[m] * np.conj(a1[m])))))

    alone = {k: interecho_phase(k) for k in ("blood", "wall", "muscle", "fat")}
    print("each tissue simulated alone (no neighbours):")
    for k, v in alone.items():
        print(f"  {k:7} inter-echo phase {v:+7.2f} deg")

    checks = {
        "blood brighter than muscle": mag["blood"] > mag["muscle"],
        "blood brighter than fat": mag["blood"] > mag["fat"],
        "water-like tissue alone stays in phase (<1 deg)":
            all(abs(alone[k]) < 1.0 for k in ("blood", "wall", "muscle")),
        "fat alone rotates by the designed angle (<1 deg)":
            abs(abs(alone["fat"]) - abs(designed_phase)) < 1.0,
    }
    print(f"\nblood/muscle {mag['blood']/mag['muscle']:.2f}x, "
          f"blood/fat {mag['blood']/max(mag['fat'], 1e-9):.1f}x")
    for label_, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label_}")

    water, fat, psi = separate(s1, s2, rep["te1"], rep["te2"], df)
    print("\nDixon output (illustrative only -- see the module docstring):")
    for k in ("blood", "muscle", "fat"):
        m = regions[k]
        print(f"  {k:7} mean W {water[m].mean():.4f}   mean F {fat[m].mean():.4f}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        for a, im, t in ((ax[0, 0], np.abs(s1), "echo 1"),
                         (ax[0, 1], np.abs(s2), "echo 2"),
                         (ax[0, 2], label, "phantom labels"),
                         (ax[1, 0], water, "water (illustrative)"),
                         (ax[1, 1], fat, "fat (illustrative)"),
                         (ax[1, 2], np.angle(s2 * np.conj(s1)), "inter-echo phase")):
            h = a.imshow(im, cmap="twilight" if "phase" in t else "gray")
            a.set_title(t)
            a.axis("off")
            fig.colorbar(h, ax=a, fraction=0.046)
        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print(f"wrote {args.out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
