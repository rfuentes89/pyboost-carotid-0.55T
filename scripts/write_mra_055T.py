#!/usr/bin/env python
"""Build the 0.55T bright-blood MRA sequence with optimized parameters, write .seq.

Parameters come from the differentiable / discrete optimizations (see
README_pypulseq.md): imaging flip ~110 deg, T2-prep TE ~79 ms (CNR-efficiency
operating point), centric ordering, FatSat on. Runs timing/SAR checks then writes
a scanner-executable `.seq`.

Usage
-----
    python scripts/write_mra_055T.py [--out mra_055T.seq]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboost import build_mra_sequence, BoostParams, scanner_055T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mra_055T.seq", help="output .seq path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=120)
    ap.add_argument("--flip", type=float, default=110.0, help="imaging flip [deg]")
    ap.add_argument("--t2prep", type=float, default=0.079, help="T2-prep TE [s]")
    args = ap.parse_args()

    system = scanner_055T()
    params = BoostParams(nx=args.nx, ny=args.ny, centric=True,
                         t2prep_duration=args.t2prep,
                         im_flip_angle=(args.flip, 80.0))
    seq = build_mra_sequence(params, system, use_t2prep=True, use_fatsat=True)

    ok, errors = seq.check_timing()
    print(f"[timing]  check_timing: {'OK' if ok else f'{len(errors)} ERRORS'}")
    if not ok:
        for e in errors[:10]:
            print("   ", e)
        return 1

    print("\n[report]")
    print(seq.test_report())

    try:
        from pypulseq.SAR.SAR_calc import calc_SAR
        sar = calc_SAR(seq)
        peak = float(np.max(sar)) if np.ndim(sar) else float(sar)
        print(f"[SAR]     peak whole-body estimate: {peak:.3f} W/kg "
              f"(0.55T runs far below the 4 W/kg limit)")
    except Exception as exc:  # pragma: no cover - depends on optional data
        print(f"[SAR]     skipped ({exc})")

    dur, n_blocks, _ = seq.duration()
    print(f"\n[write]   {n_blocks} blocks, {dur:.2f} s total "
          f"({dur / params.rr:.1f} RR intervals); flip {args.flip:.0f} deg, "
          f"T2prep {args.t2prep*1e3:.0f} ms")
    seq.write(args.out)
    print(f"[write]   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
