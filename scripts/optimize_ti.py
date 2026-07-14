#!/usr/bin/env python
"""Sweep the inversion time (TI) to find the black-blood operating point.

Why phase-sensitive. A plain magnitude subtraction |c0 - c1| cannot null blood:
blood has the longest T1, so any TI at which its inversion has recovered has also
recovered every shorter-T1 tissue. Real BOOST black-blood is *phase sensitive* --
it recovers the sign of the inverted magnetization. We reproduce that by
projecting the bright-blood contrast (T2prep + IR, c0) onto the phase of the
reference contrast (c1):

    psir(tissue) = Re(c0 * conj(c1)) / |c1|

Blood, still inverted (negative Mz) at a short TI, comes out negative (dark);
tissue that has recovered past its null comes out positive (bright). The optimal
TI maximizes the wall-to-lumen separation while driving blood through zero.

Usage
-----
    python scripts/optimize_ti.py [--out ti_sweep.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MRzeroCore as mr0
from pyboost import build_boost_sequence, BoostParams, scanner_055T

# 0.55T tissues (RR_sim.jl:199-207 + muscle). Fat is not the target here.
TISSUES = {
    "blood":  dict(PD=0.70, T1=1.122, T2=0.263),
    "wall":   dict(PD=0.60, T1=0.750, T2=0.090),
    "muscle": dict(PD=0.70, T1=0.600, T2=0.045),
}
COLORS = {"blood": "#c0392b", "wall": "#2980b9", "muscle": "#27ae60"}


def center_complex(signal, kspace, nx, im_seg, inav_lines):
    """Complex signal of the FIRST acquired line for each contrast (c0, c1).

    With centric ordering the first imaging line is ky~0 (the DC line), sampled
    right after the preparation -- so its central-readout sample tracks the
    prepared longitudinal magnetization (and its sign) before the bSSFP transient
    decays. The alternating (0, pi) RF phase of that line (index 0) is undone.
    """
    sig = signal.detach().cpu().numpy().reshape(-1)
    k = kspace.detach().cpu().numpy()
    per = im_seg * nx
    out = []
    for contrast in (0, 1):
        b = slice(contrast * per, (contrast + 1) * per)
        first_line = slice(0, nx)                       # first acquired line (ky~0)
        kb, sb = k[b][first_line], sig[b][first_line]
        idx = int(np.argmin(np.abs(kb[:, 0])))          # central readout sample (kx~0)
        phase = np.exp(-1j * np.pi * ((inav_lines + 0) % 2))
        out.append(sb[idx] * phase)
    return out[0], out[1]


def simulate_tissue(seq0, tp, p):
    obj = mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"], T2=tp["T2"],
        T2dash=0.03, D=0.0, voxel_size=0.005)
    signal, kspace = mr0.util.simulate(seq0, obj)
    c0, c1 = center_complex(signal, kspace, p.nx, p.im_segments, p.inav_lines)
    return c0, c1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ti_sweep.png")
    args = ap.parse_args()

    ti_ms = np.arange(60, 900, 40)                     # inversion times to try
    psir = {name: [] for name in TISSUES}              # signed black-blood signal
    used_ti = []

    for ti in ti_ms:
        # RR generous so a long TI still fits the heartbeat.
        p = BoostParams(nx=16, ny=16, im_segments=16, dummy_heart_beats=1,
                        rr=2.2, ir_inversion_time=ti * 1e-3, centric=True)
        try:
            seq = build_boost_sequence(p, scanner_055T(), add_trigger=False)
        except ValueError:
            continue                                    # TI doesn't fit -> skip
        seq.write("/tmp/ti.seq")
        seq0 = mr0.Sequence.import_file("/tmp/ti.seq")
        used_ti.append(ti)
        for name, tp in TISSUES.items():
            c0, c1 = simulate_tissue(seq0, tp, p)
            psir[name].append(np.real(c0 * np.conj(c1)) / (abs(c1) + 1e-9))
        print(f"TI={ti:4d} ms  " + "  ".join(
            f"{n}={psir[n][-1]:+.3f}" for n in TISSUES))

    used_ti = np.array(used_ti)
    for n in TISSUES:
        psir[n] = np.array(psir[n])

    # Black-blood operating point: null the blood (|signal| -> 0). In the
    # phase-sensitive image the displayed intensity is |signal|, so blood goes
    # dark exactly at its inversion null, while the shorter-T1 wall and muscle
    # (already past their own nulls, hence opposite sign) keep a large magnitude
    # and stay bright.
    best = int(np.argmin(np.abs(psir["blood"])))
    ti_opt = used_ti[best]
    b, w, m = (psir[n][best] for n in ("blood", "wall", "muscle"))
    print(f"\nBlood-null TI ~ {ti_opt} ms:  blood={b:+.3f} (|{abs(b):.3f}| dark)  "
          f"wall={w:+.3f} (|{abs(w):.3f}| bright)  muscle={m:+.3f}")
    contrast = abs(w) - abs(b)
    print(f"wall-to-lumen black-blood contrast |wall|-|blood| = {contrast:+.3f}")

    # --- figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in TISSUES:                                # signed recovery curves
        ax1.plot(used_ti, psir[name], "-o", ms=4, color=COLORS[name], label=name)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axvline(ti_opt, color="0.4", ls="--", lw=1)
    ax1.set_xlabel("inversion time TI [ms]")
    ax1.set_ylabel("phase-sensitive signal  Re(c0·c1*)/|c1|")
    ax1.set_title("Inversion recovery per tissue (zero crossing = null)")
    ax1.legend()
    for name in TISSUES:                                # displayed black-blood
        ax2.plot(used_ti, np.abs(psir[name]), "-o", ms=4, color=COLORS[name],
                 label=name)
    ax2.axvline(ti_opt, color="0.4", ls="--", lw=1,
                label=f"blood-null TI ~ {ti_opt} ms")
    ax2.set_xlabel("inversion time TI [ms]")
    ax2.set_ylabel("black-blood image intensity  |signal|")
    ax2.set_title("Black-blood magnitude — blood dark at its null, wall bright")
    ax2.legend()
    fig.suptitle("BOOST TI optimization @ 0.55T (block inversion, MRzero)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
