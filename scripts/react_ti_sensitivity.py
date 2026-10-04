#!/usr/bin/env python
"""How much do REACT's null times move within the measured 0.55T relaxation spread?

Source of the tissue values
---------------------------
Campbell-Washburn AE et al., Radiology 2019, table of T1/T2/T2* at 0.55T and 1.5T
(supplied by the user as an image). Used here, mean +- SD at 0.55T:

    arterial blood   T1 1122 +- 85 ms   T2 263 +- 27 ms
    fat              T1  187 +- 10 ms   T2  93 +- 16 ms
    myocardium       T1  701 +- 24 ms   T2  58 +-  6 ms

``TISSUE_PROPERTIES`` matches blood exactly and uses 183 ms for fat T1 (within the
SD of the table's 187; kept as is). **Skeletal muscle and vessel wall are not in
that table**: muscle 450/55 ms and wall 750/90 ms come from the Koma reference
script with no primary source. Myocardium is printed below only as a labelled
proxy bound for muscle, not as a substitute: it is a different tissue.

An earlier version of this script tested a fat T1 of 125-135 ms (from reading a
"47% of the 1.5T value" statement in an abstract). The table refutes it: measured
fat T1 is 187 ms, 55-65% of its 1.5T range (288-343 ms).

The null time after the T2-prep is ``TI = T1 ln(1 + exp(-TE_prep/T2))``
(``pyboost.params.null_time_after_t2prep``); no simulator is used here.

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

# Campbell-Washburn 2019, 0.55T: (T1, sd, T2, sd) in seconds.
TABLE = {
    "blood":      (1.122, 0.085, 0.263, 0.027),
    "fat":        (0.187, 0.010, 0.093, 0.016),
    "myocardium": (0.701, 0.024, 0.058, 0.006),
}


def null_ms(t1, t2):
    return null_time_after_t2prep(t1, t2, TE_PREP) * 1e3


def main() -> int:
    print(f"T2-prep {TE_PREP*1e3:.0f} ms. Null time TI = T1 ln(1+E2) [ms]\n")
    print(f"{'tissue':<12}{'T1 [ms]':>9}{'T2 [ms]':>9}{'null TI':>10}  source")
    for n in ("fat", "muscle", "wall", "blood"):
        t = T[n]
        src = {"blood": "matches Campbell-Washburn 2019",
               "fat": "T2 matches; T1 183 vs 187 +- 10 in the table",
               "muscle": "NOT in the table; Koma script, no primary source",
               "wall": "NOT in the table; Koma script, no primary source"}[n]
        print(f"{n:<12}{t['T1']*1e3:>9.0f}{t['T2']*1e3:>9.0f}"
              f"{null_ms(t['T1'], t['T2']):>10.1f}  {src}")

    print("\nNull time from the table's own mean +- SD (T1 and T2 varied together "
          "to the extremes):")
    print(f"{'tissue':<12}{'mean':>8}{'T1 +-SD only':>20}{'T2 +-SD only':>20}{'extremes':>20}")
    for n, (t1, s1, t2, s2) in TABLE.items():
        mean = null_ms(t1, t2)
        a = (null_ms(t1 - s1, t2), null_ms(t1 + s1, t2))
        b = (null_ms(t1, t2 - s2), null_ms(t1, t2 + s2))
        corners = [null_ms(t1 + i * s1, t2 + j * s2) for i in (-1, 1) for j in (-1, 1)]
        print(f"{n:<12}{mean:>8.1f}{a[0]:>10.1f}-{a[1]:<9.1f}{b[0]:>10.1f}-{b[1]:<9.1f}"
              f"{min(corners):>10.1f}-{max(corners):<9.1f}")

    print("\nMuscle null, the water-image optimum: the table has no skeletal muscle.")
    print(f"  code value 450/55 ms (no source)      -> {null_ms(0.450, 0.055):6.1f} ms")
    print(f"  myocardium 701/58 ms (proxy bound)    -> {null_ms(0.701, 0.058):6.1f} ms")
    print("  muscle null over T1 +-20% and T2 40-70 ms around the code value:")
    print(f"  {'':>10}" + "".join(f"T2 {t2*1e3:>3.0f} ms " for t2 in (0.040, 0.055, 0.070)))
    for t1 in (0.36, 0.45, 0.54):
        print(f"  T1 {t1*1e3:>4.0f} ms" + "".join(f"{null_ms(t1, t2):>10.1f} "
                                                  for t2 in (0.040, 0.055, 0.070)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
