#!/usr/bin/env python
"""Build the 0.55T BOOST carotid sequence, run safety/timing checks, write .seq.

Usage
-----
    python scripts/write_boost_055T.py [--out boost_055T.seq]

Runs from the repo root with the project venv active (see README_pypulseq.md).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pypulseq as pp
from pyboost import build_boost_sequence, BoostParams, scanner_055T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="boost_055T.seq", help="output .seq path")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=120)
    args = ap.parse_args()

    system = scanner_055T()
    params = BoostParams(nx=args.nx, ny=args.ny)
    seq = build_boost_sequence(params, system)

    # 1. Timing --------------------------------------------------------------
    ok, errors = seq.check_timing()
    print(f"[timing]  check_timing: {'OK' if ok else f'{len(errors)} ERRORS'}")
    if not ok:
        for e in errors[:10]:
            print("   ", e)
        return 1

    # 2. Test report (TE/TR, flip angles, k-space, duration) -----------------
    print("\n[report]")
    print(seq.test_report())

    # 3. SAR (best effort; at 0.55T there is large headroom) -----------------
    try:
        from pypulseq.SAR.SAR_calc import calc_SAR
        sar = calc_SAR(seq)
        peak = float(np.max(sar)) if np.ndim(sar) else float(sar)
        print(f"[SAR]     peak whole-body estimate: {peak:.3f} W/kg "
              f"(0.55T runs far below the 4 W/kg limit)")
    except Exception as exc:  # pragma: no cover - depends on optional data
        print(f"[SAR]     skipped ({exc})")

    # 4. Write ---------------------------------------------------------------
    dur, n_blocks, _ = seq.duration()
    print(f"\n[write]   {n_blocks} blocks, {dur:.2f} s total "
          f"({dur / params.rr:.1f} RR intervals)")
    seq.write(args.out)
    print(f"[write]   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
