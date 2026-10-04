#!/usr/bin/env python
"""How much do REACT's null times move if the 0.55T relaxation table is wrong?

``TISSUE_PROPERTIES`` comes from a reference script of the group's Koma work, not
from a primary 0.55T measurement that has been checked here. The inversion time
that nulls a tissue after the T2-prep is ``TI = T1 * ln(1 + exp(-TE_prep/T2))``
(pyboost.params.null_time_after_t2prep), so it is linear in T1 and the question
is just how large the plausible T1 error is. This script tabulates it; it uses no
simulator and adds no new tissue values -- the alternatives below are scalings of
the table's own entries, labelled as such.

Two indirect statements bound the scaling (both read as abstracts only):
* Khodarahmi 2024: T1 of non-fluid tissues at 0.55T is ~47% of the 1.5T value.
  That would put fat near 125-135 ms instead of the table's 183 ms.
* Campbell-Washburn 2019: averaged over tissues T1 is 32% shorter and T2 26%
  longer than at 1.5T (per-tissue table not read).

Usage: python scripts/react_ti_sensitivity.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboost.params import null_time_after_t2prep
from pyboost.phantom import TISSUE_PROPERTIES as T

TE_PREP = 50e-3


def mz(t1: float, t2: float, ti: float) -> float:
    """Longitudinal magnetization (units of M0) after T2-prep, IR and TI."""
    e2 = math.exp(-TE_PREP / t2)
    return 1.0 - (1.0 + e2) * math.exp(-ti / t1)


def main() -> int:
    print(f"T2-prep {TE_PREP*1e3:.0f} ms. Null time TI = T1 ln(1+E2) [ms]\n")
    print(f"{'tissue':<8}{'T1 [ms]':>9}{'T2 [ms]':>9}{'null TI':>10}")
    for n in ("fat", "muscle", "wall", "blood"):
        t = T[n]
        print(f"{n:<8}{t['T1']*1e3:>9.0f}{t['T2']*1e3:>9.0f}"
              f"{null_time_after_t2prep(t['T1'], t['T2'], TE_PREP)*1e3:>10.1f}")

    print("\nFat null vs fat T1 (T2 fixed at the table's 93 ms):")
    for t1 in (0.125, 0.135, 0.150, 0.183):
        tag = "  <- table" if t1 == 0.183 else ""
        print(f"  T1 {t1*1e3:>4.0f} ms -> TI {null_time_after_t2prep(t1, T['fat']['T2'], TE_PREP)*1e3:>5.1f} ms{tag}")

    print("\nMuscle null vs muscle T1 and T2 (the water-objective optimum sits here):")
    print(f"  {'':>10}" + "".join(f"T2 {t2*1e3:>3.0f} ms " for t2 in (0.040, 0.055, 0.070)))
    for t1 in (0.36, 0.45, 0.54):
        row = "".join(f"{null_time_after_t2prep(t1, t2, TE_PREP)*1e3:>10.1f} " for t2 in (0.040, 0.055, 0.070))
        tag = " <- table T1" if t1 == 0.45 else ""
        print(f"  T1 {t1*1e3:>4.0f} ms{row}{tag}")

    print("\nBlood-muscle contrast vs TI if the muscle T1 is off by -20/0/+20% "
          "(full recovery, no readout):")
    print(f"  {'TI [ms]':>8}{'-20%':>9}{'table':>9}{'+20%':>9}")
    for ti in (0.012, 0.0842, 0.120, 0.155, 0.190):
        vals = []
        for f in (0.8, 1.0, 1.2):
            b = abs(mz(T['blood']['T1'], T['blood']['T2'], ti))
            m = abs(mz(T['muscle']['T1'] * f, T['muscle']['T2'], ti))
            vals.append(b - m)
        print(f"  {ti*1e3:>8.1f}" + "".join(f"{v:>9.3f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
