#!/usr/bin/env python
"""Build the 0.55T REACT carotid angiography sequence, check it, write .seq.

Usage
-----
    python scripts/write_react_055T.py [--out react_055T.seq] [--nz 1]

Runs from the repo root with the project venv active (see README_pypulseq.md).

The gradient defaults below are the *derated* limits of the target MAGNETOM
Free.Max (23 mT/m, 25 T/m/s), which sit under the 26 mT/m / 45 T/m/s nameplate.
That derating matters: at 25 T/m/s essentially every gradient in this sequence
is slew-limited rather than amplitude-limited, so raising --max-grad buys almost
nothing while raising --max-slew shortens the TR directly.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboost import (build_react_sequence, ReactParams, scanner_055T,
                     kernel_report, fat_frequency)
from pyboost.phantom import TISSUE_PROPERTIES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="react_055T.seq", help="output .seq path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=120)
    ap.add_argument("--nz", type=int, default=1,
                    help="partitions; 1 = single-slice 2D, >1 = 3D slab")
    ap.add_argument("--slab", type=float, default=5.0,
                    help="slice/slab thickness [mm]")
    ap.add_argument("--fov", type=float, default=200.0, help="in-plane FOV [mm]")
    ap.add_argument("--bandwidth", type=float, default=300.0, help="Hz/pixel")
    ap.add_argument("--tfe", type=int, default=22, help="encodes per shot")
    ap.add_argument("--ti", type=float, default=None,
                    help="inversion time [ms]; default derives it from the "
                         "0.55T relaxation table")
    ap.add_argument("--inversion", choices=("block", "adiabatic"), default="adiabatic",
                    help="adiabatic for the scanner, block for MRzero validation")
    ap.add_argument("--max-grad", type=float, default=23.0, help="mT/m")
    ap.add_argument("--max-slew", type=float, default=25.0, help="T/m/s")
    args = ap.parse_args()

    system = scanner_055T(max_grad=args.max_grad, max_slew=args.max_slew,
                          rf_ringdown_time=20e-6)
    params = ReactParams(
        nx=args.nx, ny=args.ny, nz=args.nz,
        fov=args.fov * 1e-3, slice_thickness=args.slab * 1e-3,
        readout_bandwidth=args.bandwidth, tfe_factor=args.tfe,
        ti=None if args.ti is None else args.ti * 1e-3,
        inversion_kind=args.inversion,
    )

    # 0. Where the derived 0.55T numbers come from ---------------------------
    df = fat_frequency(system)
    rep = kernel_report(system, params)
    ti = params.resolved_ti()
    print("[derivation] every number below is computed, not copied from a "
          "1.5T/3T protocol")
    print(f"   fat-water shift at {system.B0} T   : {df:+.1f} Hz "
          f"(vs -220 Hz at 1.5T)")
    print(f"   Dixon spacing (162.6 deg phase)  : {rep['delta_te']*1e3:.3f} ms "
          f"-> water-fat phase {rep['water_fat_phase_deg']:.1f} deg")
    print(f"   TE1 / TE2                        : {rep['te1']*1e3:.2f} / "
          f"{rep['te2']*1e3:.2f} ms")
    print(f"   TR                               : {rep['tr']*1e3:.2f} ms")
    print(f"   readout bandwidth                : {rep['bandwidth_hz_px']:.0f} Hz/px "
          f"-> chemical shift {rep['chemical_shift_px']:.2f} px")
    if args.ti is None:
        t = TISSUE_PROPERTIES[params.ti_null_tissue]
        print(f"   TI (nulls {params.ti_null_tissue}: T1={t['T1']*1e3:.0f} ms, "
              f"T2={t['T2']*1e3:.0f} ms, after a "
              f"{params.t2prep_duration*1e3:.0f} ms T2-prep)"
              f"{'':<3}: {ti*1e3:.1f} ms")
    else:
        print(f"   TI (user-supplied)               : {ti*1e3:.1f} ms")
    print(f"   shot: {params.tfe_factor} TRs = "
          f"{params.tfe_factor*rep['tr']*1e3:.0f} ms, {params.n_shots} shots")

    seq = build_react_sequence(params, system)

    # 1. Timing --------------------------------------------------------------
    ok, errors = seq.check_timing()
    print(f"\n[timing]  check_timing: {'OK' if ok else f'{len(errors)} ERRORS'}")
    if not ok:
        for e in errors[:10]:
            print("   ", e)
        return 1

    # 2. Both echoes must share a ky line or Dixon separation is invalid ------
    ka = seq.calculate_kspace()[0]
    n = params.nx
    n_tr = ka.shape[1] // (2 * n)
    mismatch = max(abs(float(ka[1, 2 * i * n]) - float(ka[1, (2 * i + 1) * n]))
                   for i in range(n_tr))
    print(f"[dixon]   echo pairs share ky to {mismatch:.3e} 1/m "
          f"({'OK' if mismatch < 1e-6 else 'MISMATCH'})")
    if mismatch >= 1e-6:
        return 1

    # 3. Combined (vector) slew across axes ----------------------------------
    # Pulseq enforces max_slew per axis, but PNS responds to the vector sum of
    # gradients played concurrently. Since the derating here is a PNS margin,
    # report the combined figure explicitly rather than let it hide behind the
    # per-axis numbers in test_report.
    try:
        wave = seq.waveforms_and_times()[0]          # one (time, amplitude) pair per axis
        dt = system.grad_raster_time
        t = np.arange(0.0, max(w[0].max() for w in wave), dt)
        g = np.stack([np.interp(t, w[0], w[1], left=0.0, right=0.0) for w in wave])
        vec_slew = float(np.max(np.linalg.norm(np.diff(g, axis=1) / dt, axis=0)))
        vec_slew /= system.gamma                      # Hz/m/s -> T/m/s
        vec_grad = float(np.max(np.linalg.norm(g, axis=0))) / system.gamma * 1e3
        within = vec_slew <= args.max_slew * 1.001
        print(f"[slew]    peak vector gradient {vec_grad:.1f} mT/m, "
              f"peak vector slew {vec_slew:.1f} T/m/s "
              f"(per-axis limits {args.max_grad:.0f} mT/m, {args.max_slew:.0f} T/m/s)")
        if not within:
            print(f"          NOTE: the vector slew exceeds the per-axis limit, "
                  f"which is normal when\n          axes ramp concurrently (up to "
                  f"sqrt(3)x in the worst case) and is legal as far\n          as the "
                  f"scanner is concerned. But if {args.max_slew:.0f} T/m/s was chosen "
                  f"as a PNS margin,\n          that margin is what PNS actually sees "
                  f"-- lower --max-slew to about "
                  f"{args.max_slew**2/vec_slew:.0f} to hold it.")
    except Exception as exc:  # pragma: no cover - waveform export is optional
        print(f"[slew]    skipped ({exc})")

    # 4. Test report (TE/TR, flip angles, k-space, duration) -----------------
    print("\n[report]")
    print(seq.test_report())
    if args.inversion == "adiabatic":
        print("NOTE: the extra flip angle listed above (~289 deg) is the "
              "hyperbolic-secant\n      inversion. test_report derives flip from "
              "the RF integral, which is\n      meaningless for an adiabatic "
              "sweep -- the pulse still inverts.")

    # 5. SAR (best effort; at 0.55T there is large headroom) -----------------
    try:
        from pypulseq.SAR.SAR_calc import calc_SAR
        sar = calc_SAR(seq)
        peak = float(np.max(sar)) if np.ndim(sar) else float(sar)
        print(f"[SAR]     peak whole-body estimate: {peak:.3f} W/kg "
              f"(SAR scales with B0^2, so 0.55T sits at ~13% of the 1.5T value)")
    except Exception as exc:  # pragma: no cover - depends on optional data
        print(f"[SAR]     skipped ({exc})")

    # 6. Write ---------------------------------------------------------------
    dur, n_blocks, _ = seq.duration()
    print(f"\n[write]   {n_blocks} blocks, {dur:.1f} s total "
          f"({dur/60:.1f} min, {params.n_shots} shots + "
          f"{params.dummy_shots} dummy)")
    seq.write(args.out)
    print(f"[write]   wrote {args.out}")
    print("[write]   target interpreter: Pulseq v1.4.2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
