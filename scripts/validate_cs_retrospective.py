#!/usr/bin/env python
"""Retrospective test of the R-fold undersampling + CS + Dixon chain (no MRzero).

What this is, and is not
------------------------
A synthetic two-echo 3D object with a carotid-like bifurcation is Fourier
transformed, the real sampling pattern of ``pyboost.sampling`` is applied, and the
result is reconstructed with ``pyboost.cs`` and separated with ``pyboost.dixon``.
It tests the reconstruction chain on data whose truth is known. It does **not**
test the sequence (T1 recovery along each shot, echo timing, eddy currents), which
the k-space here does not model, and it is single-coil.

Object, at the protocol's 0.96 mm voxel (64^3 = 61 mm cube): elliptical muscle body,
subcutaneous fat ring, and a vessel running along z that splits into two branches.
Radii in voxels: common 3.1 (R = 3.0 mm), internal 2.3 (2.2 mm), external 1.6
(1.5 mm); the first two are the radii used for the TI scoring, the third is an
illustrative thin branch. Amplitudes (blood 1, muscle 0.25, fat 0.6, all positive)
are illustrative, not measured. Echo times come from the 3D protocol preset.

lambda is picked on one object and evaluated on others, so the numbers below are
not tuned on the data they are reported on.

Usage: python scripts/validate_cs_retrospective.py [--n 64] [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboost import scanner_055T, fat_frequency, react_3d_208x208x88, kernel_report
from pyboost.cs import cs_reconstruct
from pyboost.dixon import fat_phasors, separate
from pyboost.recon import images_from_kspace
from pyboost.sampling import sampling_mask

AMP = dict(blood=1.0, muscle=0.25, fat=0.6)


def make_object(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(float)
    c = (n - 1) / 2
    body = ((x - c) / (n * 0.42)) ** 2 + ((y - c) / (n * 0.34)) ** 2 < 1
    inner = ((x - c) / (n * 0.34)) ** 2 + ((y - c) / (n * 0.26)) ** 2 < 1
    fat = body & ~inner
    muscle = inner
    # vessel: common trunk, then two branches diverging after z0
    z0 = n * (0.45 + 0.05 * rng.standard_normal())
    cx, cy = c + 4 + 2 * rng.standard_normal(), c - 3 + 2 * rng.standard_normal()
    spread = n * 0.20
    vessel = np.zeros_like(body)
    radii = {}
    for zi in range(n):
        t = max(0.0, (zi - z0) / (n - z0))
        for rad, dx in ((3.1 if zi < z0 else 2.3, +spread * t),
                        (1.6, -0.6 * spread * t if zi >= z0 else None)):
            if dx is None:
                continue
            m = np.hypot(x[zi] - (cx + dx), y[zi] - cy) < rad
            vessel[zi] |= m
    vessel &= inner
    water = np.where(vessel, AMP["blood"], np.where(muscle, AMP["muscle"], 0.0))
    fatamp = np.where(fat, AMP["fat"], 0.0)
    smooth = np.sin(2 * np.pi * (x + 0.5 * y + 0.3 * z) / (1.7 * n))
    psi = 12.0 * smooth + 4.0 * rng.standard_normal() * 0
    return water, fatamp, psi, vessel, muscle, fat


def forward(w, f, psi, te1, te2, df, phi0=0.4):
    c1, c2 = fat_phasors(te1, te2, df)
    s1 = (w + f * c1) * np.exp(2j * np.pi * psi * te1 + 1j * phi0)
    s2 = (w + f * c2) * np.exp(2j * np.pi * psi * te2 + 1j * phi0)
    return s1, s2


def kspace(img):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))


def nrmse(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def swap_fraction(water, fat, w_t, f_t):
    fm, wm = f_t > 0, w_t > 0
    wrong = np.sum(water[fm] > fat[fm]) + np.sum(fat[wm] > water[wm])
    return float(wrong / max(1, (fm.sum() + wm.sum())))


def evaluate(obj, r, snr, lam, te1, te2, df, noise_seed=0, max_iter=100,
             seed_pattern=0):
    water, fatamp, psi, vessel, muscle, fat = obj
    n = water.shape[0]
    s1, s2 = forward(water, fatamp, psi, te1, te2, df)
    k1, k2 = kspace(s1), kspace(s2)
    sigma_img = AMP["blood"] / snr if snr else 0.0
    rng = np.random.default_rng(noise_seed)
    if sigma_img:
        sk = sigma_img * np.sqrt(k1.size)
        for k in (k1, k2):
            k += sk * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
    mask = sampling_mask(n, n, acceleration=r, seed=seed_pattern)
    ku1, ku2 = k1 * mask[:, :, None], k2 * mask[:, :, None]
    out = {}
    full = images_from_kspace(k1, k2)
    zf = images_from_kspace(ku1, ku2)
    cs = (cs_reconstruct(ku1, mask, lam=lam, max_iter=max_iter),
          cs_reconstruct(ku2, mask, lam=lam, max_iter=max_iter))
    body = muscle | fat | vessel
    w_full = separate(*full, te1, te2, df)[0]
    for name, (e1, e2) in (("full", full), ("zero-filled", zf), ("CS", cs)):
        w, f, _ = separate(e1, e2, te1, te2, df)
        out[name] = dict(
            echo1=(float("nan") if name == "full"
                   else nrmse(np.abs(e1), np.abs(full[0]), body)),
            water=nrmse(w, water, body),
            water_vs_full=(float("nan") if name == "full"
                           else nrmse(w, w_full, body)),
            lumen=float(w[vessel].mean() / max(w[muscle & ~vessel].mean(), 1e-9)),
            swaps=swap_fraction(w, f, water, fatamp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--quick", action="store_true", help="fewer settings")
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    rep = kernel_report(system, react_3d_208x208x88(5.0))
    te1, te2, df = rep["te1"], rep["te2"], fat_frequency(system)
    print(f"echo times {te1*1e3:.2f}/{te2*1e3:.2f} ms (dTE {rep['delta_te']*1e3:.2f}), "
          f"n = {args.n}^3 at 0.96 mm, truth lumen/muscle = "
          f"{AMP['blood']/AMP['muscle']:.1f}\n")

    # --- 1. lambda on a tuning object (seed 0), R = 5, SNR 30 ---------------
    tune = make_object(args.n, seed=0)
    lams = (0.001, 0.003, 0.01, 0.03, 0.1) if not args.quick else (0.003, 0.03)
    print("lambda sweep on the tuning object (R = 5, blood SNR 30):")
    print(f"{'lambda':>8}{'water vs full':>15}{'lumen/muscle':>14}{'swaps':>9}{'time [s]':>10}")
    best = None
    for lam in lams:
        t = time.time()
        r = evaluate(tune, 5.0, 30.0, lam, te1, te2, df, max_iter=args.max_iter)["CS"]
        print(f"{lam:>8g}{r['water_vs_full']:>15.4f}{r['lumen']:>14.2f}{r['swaps']:>9.3f}{time.time()-t:>10.1f}",
              flush=True)
        if best is None or r["water_vs_full"] < best[1]:
            best = (lam, r["water_vs_full"])
    lam = best[0]
    print(f"-> lambda = {lam:g} (lowest water error against the fully sampled Dixon on the tuning object)\n")

    # --- 2. held-out objects: R and noise -----------------------------------
    rs = (3.0, 5.0, 8.0) if not args.quick else (5.0,)
    snrs = (30.0, 15.0) if not args.quick else (30.0,)
    held = [make_object(args.n, seed=s) for s in (1, 2)]
    print(f"held-out objects (seeds 1, 2), lambda = {lam:g}; mean of the two:")
    print(f"{'R':>4}{'SNR':>5}  {'recon':<12}{'echo1 vs full':>14}{'water vs full':>14}"
          f"{'water vs truth':>15}{'lumen/muscle':>14}{'swaps':>8}")
    for r in rs:
        for snr in snrs:
            res = [evaluate(o, r, snr, lam, te1, te2, df, noise_seed=i,
                            max_iter=args.max_iter) for i, o in enumerate(held)]
            for name in ("full", "zero-filled", "CS"):
                m = {k: float(np.mean([x[name][k] for x in res]))
                     for k in ("echo1", "water", "water_vs_full", "lumen", "swaps")}
                e1 = "-" if np.isnan(m["echo1"]) else f"{m['echo1']:.4f}"
                wf = "-" if np.isnan(m["water_vs_full"]) else f"{m['water_vs_full']:.4f}"
                print(f"{r:>4g}{snr:>5g}  {name:<12}{e1:>14}{wf:>14}{m['water']:>15.4f}"
                      f"{m['lumen']:>14.2f}{m['swaps']:>8.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
