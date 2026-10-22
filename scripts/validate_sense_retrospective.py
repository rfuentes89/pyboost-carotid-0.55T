#!/usr/bin/env python
"""Retrospective test of SENSE + Dixon on a SYNTHETIC 20-channel array.

What this is, and is not
------------------------
A synthetic two-echo 3D neck-sized object is multiplied by the sensitivities of a
**synthetic** 20-loop array (``pyboost.coils``: invented geometry, NOT the user's
head/neck coil), Fourier transformed with independent noise per channel,
undersampled and reconstructed. It tests that the SENSE chain is implemented
correctly -- unfolding, geometry factor, phase handling, self-calibrated map
estimation -- on data whose truth is known. It does not give the real coil's
g-factor, and the k-space does not model the sequence dynamics.

Two kinds of undersampling, and why the main test is the regular one:

* **Regular** (every ry-th ky line and rz-th kz line, a fully sampled calibration
  scan separate): classical SENSE, solved in closed form per aliased voxel
  (``pyboost.sense.unfold_regular``). Exact, no convergence question, exact g-factor.
  This is the SENSE that the published REACT protocols describe (Isaak 2021: 3.2).
* **Random variable-density** (the pattern of ``pyboost.sampling``, meant for CS):
  needs an iterative solver. It is NOT validated here: with sigpy's unpreconditioned
  solvers it converges too slowly to judge (see ``pyboost.sense.sense_reconstruct``),
  and ``--random`` only reproduces that measurement.

Scaling. The protocol's 208 x 208 x 88 grid at 0.96 mm is too large for a 20-channel
bench, so the grid here is 64 x 64 x 32 over the *same physical field of view*
(200 x 200 x 84.5 mm, voxels 3.1 x 3.1 x 2.6 mm). SENSE conditioning depends on the
coil variation relative to the aliasing distance FOV/R, which this preserves; SNR per
voxel and resolution do not carry over. Vessels are therefore enlarged (common 12 mm,
internal 8.8 mm, external 6 mm diameter, 2x the radii used for the TI scoring) so they
are resolved. Amplitudes (blood 1, muscle 0.25, fat 0.6) are illustrative.

Usage: python scripts/validate_sense_retrospective.py [--snr 30] [--random]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboost import scanner_055T, fat_frequency, react_3d_208x208x88, kernel_report
from pyboost.coils import synthetic_array
from pyboost.cs import cs_reconstruct
from pyboost.dixon import fat_phasors, separate
from pyboost.sampling import sampling_mask
from pyboost.sense import (estimate_maps, sense_reconstruct, coil_combine_rss,
                           regular_mask, unfold_regular)

NZ, NY, NX = 32, 64, 64
FOV_XY, SLAB = 0.200, 0.0845
VOX = (SLAB / NZ, FOV_XY / NY, FOV_XY / NX)
AMP = dict(blood=1.0, muscle=0.25, fat=0.6)
AX = (-3, -2, -1)


def make_object(seed: int):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:NZ, 0:NY, 0:NX].astype(float)
    xm, ym = (x - (NX - 1) / 2) * VOX[2], (y - (NY - 1) / 2) * VOX[1]
    body = (xm / 0.055) ** 2 + (ym / 0.045) ** 2 < 1
    inner = (xm / 0.047) ** 2 + (ym / 0.037) ** 2 < 1
    fat, muscle = body & ~inner, inner
    z0 = NZ * (0.45 + 0.05 * rng.standard_normal())
    cx = 0.012 + 0.004 * rng.standard_normal()
    cy = -0.008 + 0.004 * rng.standard_normal()
    vessel = np.zeros(body.shape, bool)
    for zi in range(NZ):
        t = max(0.0, (zi - z0) / (NZ - z0))
        rad_main = 0.006 if zi < z0 else 0.0044
        vessel[zi] |= np.hypot(xm[zi] - (cx + 0.016 * t), ym[zi] - cy) < rad_main
        if zi >= z0:
            vessel[zi] |= np.hypot(xm[zi] - (cx - 0.010 * t), ym[zi] - cy) < 0.003
    vessel &= inner
    water = np.where(vessel, AMP["blood"], np.where(muscle, AMP["muscle"], 0.0))
    fatamp = np.where(fat, AMP["fat"], 0.0)
    psi = 12.0 * np.sin(2 * np.pi * (x + 0.5 * y + 0.3 * z) / (1.7 * NX))
    return water, fatamp, psi, vessel, muscle, fat, body


def two_echoes(w, f, psi, te1, te2, df, phi0=0.4):
    c1, c2 = fat_phasors(te1, te2, df)
    s1 = (w + f * c1) * np.exp(2j * np.pi * psi * te1 + 1j * phi0)
    s2 = (w + f * c2) * np.exp(2j * np.pi * psi * te2 + 1j * phi0)
    return s1, s2


def to_k(img, maps=None):
    a = img[None] * (maps if maps is not None else 1.0)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a, axes=AX), axes=AX), axes=AX)


def nrmse(a, b, m):
    return float(np.linalg.norm((a - b)[m]) / np.linalg.norm(b[m]))


def swaps(water, fat, w_t, f_t):
    fm, wm = f_t > 0, w_t > 0
    return float((np.sum(water[fm] > fat[fm]) + np.sum(fat[wm] > water[wm]))
                 / max(1, fm.sum() + wm.sum()))


def adjoint_combine(ksp, maps):
    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp, axes=AX), axes=AX),
                          axes=AX)
    num = (np.conj(maps) * img).sum(0)
    return num / np.maximum((np.abs(maps) ** 2).sum(0), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=30.0,
                    help="blood SNR of the fully sampled image at the median coil RSS")
    ap.add_argument("--random", action="store_true",
                    help="also run the slow iterative reconstructions on the random "
                         "pattern (not converged, see the docstring)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    rep = kernel_report(system, react_3d_208x208x88(5.0))
    te1, te2, df = rep["te1"], rep["te2"], fat_frequency(system)

    maps = synthetic_array((NZ, NY, NX), VOX)
    rss_map = np.sqrt((np.abs(maps) ** 2).sum(0))
    water, fatamp, psi, vessel, muscle, fat, body = make_object(1)
    print(f"synthetic array: {maps.shape[0]} channels; RSS inside the neck "
          f"{rss_map[body].min():.2f}-{rss_map[body].max():.2f} "
          f"(median {np.median(rss_map[body]):.2f}); grid {NZ}x{NY}x{NX} over "
          f"200x200x84.5 mm; echo times {te1*1e3:.2f}/{te2*1e3:.2f} ms")

    n_vox = NZ * NY * NX
    s1, s2 = two_echoes(water, fatamp, psi, te1, te2, df)
    clean1, clean2 = to_k(s1, maps), to_k(s2, maps)

    def run(snr):
        sigma_k = (0.0 if snr <= 0 else (AMP["blood"] / snr)
                   * float(np.median(rss_map[body])) * np.sqrt(n_vox))
        rng = np.random.default_rng(0)

        def noise():
            return sigma_k * (rng.standard_normal(maps.shape)
                              + 1j * rng.standard_normal(maps.shape)) / np.sqrt(2)

        k1, k2 = clean1 + noise(), clean2 + noise()
        full = [unfold_regular(k, maps, 1, 1)[0] for k in (k1, k2)]
        w_full, f_full, _ = separate(full[0], full[1], te1, te2, df)
        label = "noise-free" if snr <= 0 else f"blood SNR {snr:g}"
        print(f"\n=== {label} ===")
        print(f"fully sampled (R = 1) SENSE combination -> Dixon: water vs truth "
              f"{nrmse(w_full, water, body):.3f}, lumen/muscle "
              f"{w_full[vessel].mean()/w_full[muscle & ~vessel].mean():.2f}, swaps "
              f"{swaps(w_full, f_full, water, fatamp):.3f}")
        r1, r2 = coil_combine_rss(k1), coil_combine_rss(k2)
        w_r, f_r, _ = separate(r1, r2, te1, te2, df)
        print(f"RSS combination (phase discarded) -> Dixon: water vs truth "
              f"{nrmse(w_r, water, body):.3f}, swaps {swaps(w_r, f_r, water, fatamp):.3f}"
              f"   <- why phase must be kept")

        # self-calibrated maps: ESPIRiT on a separate fully sampled centre block
        cal = np.zeros_like(k1)
        c0, c1, c2 = NZ // 2, NY // 2, NX // 2
        cal[:, c0 - 8:c0 + 8, c1 - 8:c1 + 8, c2 - 8:c2 + 8] = \
            k1[:, c0 - 8:c0 + 8, c1 - 8:c1 + 8, c2 - 8:c2 + 8]
        est = estimate_maps(cal, calib_width=16, kernel_width=6)

        print(f"\n{'ry x rz':>8}{'R':>4}{'maps':>10}{'echo1 vs full':>14}{'water vs full':>14}"
              f"{'vs truth':>10}{'lumen/mus':>10}{'swaps':>7}{'g mean':>8}{'g med':>7}{'g max':>7}")
        for ry, rz in ((2, 1), (2, 2), (4, 2)):
            mask = regular_mask((NZ, NY), ry, rz)
            m4 = mask[None, :, :, None]
            for name, mp in (("oracle", maps), ("ESPIRiT", est)):
                e1, g = unfold_regular(k1 * m4, mp, ry, rz)
                e2, _ = unfold_regular(k2 * m4, mp, ry, rz)
                if name == "ESPIRiT":
                    # ESPIRiT maps are unit-norm, so y = rss * S_unit * x and the
                    # unfolded image is x * rss (a coil-RSS intensity bias). Divide it
                    # out with the oracle RSS to isolate the unfolding error.
                    e1, e2 = e1 / rss_map, e2 / rss_map
                w, f, _ = separate(e1, e2, te1, te2, df)
                print(f"{ry}x{rz:>5}{ry*rz:>4}{name:>10}"
                      f"{nrmse(np.abs(e1), np.abs(full[0]), body):>14.4f}"
                      f"{nrmse(w, w_full, body):>14.4f}{nrmse(w, water, body):>10.3f}"
                      f"{w[vessel].mean()/max(w[muscle & ~vessel].mean(),1e-9):>10.2f}"
                      f"{swaps(w, f, water, fatamp):>7.3f}"
                      f"{g[body].mean():>8.2f}{np.median(g[body]):>7.2f}{g[body].max():>7.2f}",
                      flush=True)
        return noise, k1

    run(0.0)
    noise, k1 = run(args.snr)

    # Monte-Carlo check of the analytic g-factor (linear unfolding, so exact in expectation)
    sigma_k = (AMP["blood"] / args.snr) * float(np.median(rss_map[body])) * np.sqrt(n_vox)
    rng = np.random.default_rng(1)
    print("\nMonte-Carlo check of g (24 noise-only replicas, oracle maps):")
    print(f"{'ry x rz':>8}{'g analytic (mean)':>20}{'g measured (mean)':>20}")
    reps = [sigma_k * (rng.standard_normal(maps.shape)
                       + 1j * rng.standard_normal(maps.shape)) / np.sqrt(2)
            for _ in range(24)]
    std_full = np.stack([unfold_regular(n_, maps, 1, 1)[0] for n_ in reps]).std(0)
    for ry, rz in ((2, 1), (2, 2), (4, 2)):
        mask = regular_mask((NZ, NY), ry, rz)
        m4 = mask[None, :, :, None]
        stack, g = [], None
        for n_ in reps:
            im, g = unfold_regular(n_ * m4, maps, ry, rz)
            stack.append(im)
        measured = np.stack(stack).std(0) / (std_full * np.sqrt(ry * rz))
        print(f"{ry}x{rz:>5}{g[body].mean():>20.2f}{measured[body].mean():>20.2f}",
              flush=True)

    if args.random:
        print("\nrandom variable-density pattern, iterative solvers (NOT converged):")
        mask = sampling_mask(NY, NZ, acceleration=5.0)
        m4 = mask[None, :, :, None]
        k1c = clean1 * m4
        for it in (60, 150, 300):
            e = sense_reconstruct(k1c, mask, maps, lam=1e-4, max_iter=it)
            print(f"  CG-SENSE, noise-free, {it:4d} iterations: echo-1 error vs truth "
                  f"{nrmse(e, s1, body):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
