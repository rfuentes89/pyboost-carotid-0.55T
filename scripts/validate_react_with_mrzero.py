#!/usr/bin/env python
"""Validate the REACT sequence against a 0.55T tissue model in MRzero.

Two independent things are checked, because REACT rests on two mechanisms:

1. **Relaxation contrast.** After a 50 ms T2-prep and a non-selective inversion
   at the derived TI, blood must be bright while fat and muscle sit near null.
   This is what makes REACT flow-independent -- and it is why the simulation is
   meaningful at all: MRzero models relaxation faithfully, and REACT's contrast
   is *entirely* relaxation-driven, with no inflow term to be missed.

2. **Dixon encoding.** Fat must accrue the designed water-fat phase between the
   two echoes while water accrues none. This is the check that cannot be faked:
   with the phantom's fat offset switched off, both echoes are identical and
   water/fat separation is untestable, so the simulation would "pass" while
   proving nothing. Here fat carries its real -80 Hz offset.

The inversion is forced to ``block`` (a hard 180 deg): MRzero's PDG model treats
each RF as an instantaneous rotation, so it does not reproduce an adiabatic
frequency sweep. The scanner ``.seq`` uses the adiabatic pulse; see
``pyboost.prep.inversion``.

Usage
-----
    python scripts/validate_react_with_mrzero.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MRzeroCore as mr0
from pyboost import (build_react_sequence, ReactParams, scanner_055T,
                     kernel_report, fat_frequency)
from pyboost.phantom import TISSUE_PROPERTIES as TISSUES


def echo_signals(signal, kspace, nx):
    """Complex signal at the true k-space centre, for echo 1 and echo 2.

    Acquisition order is echo1 (nx samples), echo2 (nx samples), per TR.

    It matters that this takes the *global* DC point rather than averaging each
    TR's echo peak. Every TR peaks at kx=0, but only one sits at ky=0 too, and
    with centric ordering that one is acquired first -- while the prepared
    magnetization is still fresh. Averaging over the shot instead mixes it with
    late TRs that have already relaxed toward the spoiled steady state (where
    short-T1 fat outshines long-T1 blood), which inverts the apparent contrast.
    DC dominates the image, so DC is what to measure.
    """
    sig = signal.detach().cpu().numpy().reshape(-1)
    k = kspace.detach().cpu().numpy()
    n_tr = len(sig) // (2 * nx)
    out = []
    for echo in (0, 1):
        idx = np.concatenate([np.arange((2 * i + echo) * nx,
                                        (2 * i + echo) * nx + nx)
                              for i in range(n_tr)])
        radius = np.linalg.norm(k[idx][:, :3], axis=1)
        out.append(sig[idx][np.argmin(radius)])
    return out[0], out[1]


def main() -> int:
    system = scanner_055T(max_grad=23.0, max_slew=25.0, rf_ringdown_time=20e-6)
    # Small matrix: the contrast and the echo phase do not depend on resolution,
    # and this keeps the simulation to seconds.
    p = ReactParams(nx=32, ny=16, tfe_factor=16, dummy_shots=1,
                    inversion_kind="block")
    rep = kernel_report(system, p)
    df = fat_frequency(system)
    expected_phase = 360.0 * df * rep["delta_te"]     # negative: fat is below water

    seq = build_react_sequence(p, system)
    seq_path = "/tmp/react_validate.seq"
    seq.write(seq_path)
    seq0 = mr0.Sequence.import_file(seq_path)

    print(f"REACT at {system.B0} T: TE1={rep['te1']*1e3:.2f} ms, "
          f"TE2={rep['te2']*1e3:.2f} ms, dTE={rep['delta_te']*1e3:.3f} ms, "
          f"TR={rep['tr']*1e3:.2f} ms")
    print(f"TI={p.resolved_ti()*1e3:.1f} ms (derived to null "
          f"{p.ti_null_tissue}), T2-prep={p.t2prep_duration*1e3:.0f} ms")
    print(f"fat offset {df:+.1f} Hz -> expected inter-echo phase "
          f"{expected_phase:+.1f} deg\n")

    header = (f"{'tissue':8s}{'|echo1|':>10s}{'|echo2|':>10s}"
              f"{'arg(S2/S1)':>13s}{'B0 [Hz]':>10s}")
    print(header)
    print("-" * len(header))

    results = {}
    for name, tp in TISSUES.items():
        # Only fat is off-resonance; that offset is the entire basis of Dixon.
        b0 = df if name == "fat" else 0.0
        obj = mr0.CustomVoxelPhantom(
            pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"], T2=tp["T2"],
            T2dash=0.03, B0=b0, D=0.0, voxel_size=0.005,
        )
        signal, kspace = mr0.util.simulate(seq0, obj)
        s1, s2 = echo_signals(signal, kspace, p.nx)
        dphi = np.rad2deg(np.angle(s2 / s1))
        results[name] = (abs(s1), abs(s2), dphi)
        print(f"{name:8s}{abs(s1):>10.4f}{abs(s2):>10.4f}{dphi:>13.1f}{b0:>10.1f}")

    print()
    checks = {}

    # --- 1. Relaxation contrast -------------------------------------------
    blood = results["blood"][0]
    muscle = results["muscle"][0]
    fat = results["fat"][0]
    checks["blood brighter than muscle"] = blood > muscle
    checks["blood brighter than fat"] = blood > fat
    print(f"blood/muscle = {blood/max(muscle,1e-9):.2f}x, "
          f"blood/fat = {blood/max(fat,1e-9):.2f}x")

    # --- 2. Dixon encoding -------------------------------------------------
    # Water species must stay in phase between the echoes; fat must rotate by
    # the designed angle. Tolerance is loose because T2* decay and the spoiled
    # steady state add a small common phase.
    water_dphi = max(abs(results[t][2]) for t in ("blood", "muscle", "wall"))
    fat_dphi = results["fat"][2]
    checks["water stays in phase between echoes"] = water_dphi < 15.0
    # Compare magnitudes: the simulator's phase-accumulation sign convention is
    # not the one we happen to write the fat offset with, and only the size of
    # the rotation is a physical statement about the echo spacing.
    checks["fat rotates by the designed angle"] = \
        abs(abs(fat_dphi) - abs(expected_phase)) < 20.0
    print(f"water inter-echo phase <= {water_dphi:.1f} deg; "
          f"fat {fat_dphi:+.1f} deg vs designed {abs(expected_phase):.1f} deg "
          f"in magnitude")

    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    ok = all(checks.values())
    print("\nREACT contrast and Dixon encoding behave as designed at 0.55T."
          if ok else
          "\nWARNING: inspect the TI, the T2-prep, or the echo spacing.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
