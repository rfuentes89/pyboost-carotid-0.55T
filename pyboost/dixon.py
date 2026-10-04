"""Two-point Dixon water/fat separation for REACT's flexible echo times.

Why the classic formula does not apply
--------------------------------------
Textbook two-point Dixon assumes one echo exactly in phase and one exactly
opposed, so ``W = (S_in + S_opp)/2`` and ``F = (S_in - S_opp)/2`` on magnitudes.
REACT's echoes at 0.55T sit at about -120 and +78 deg of water-fat phase, and the
information lives in the *complex* signal; taking magnitudes first discards it.

Provenance -- read this before citing
-------------------------------------
The echo-time *concept* (two-point Dixon with freely chosen echo times, which is
what REACT's mDIXON readout relies on) is that of

    Eggers H, Brendel B, Duijndam A, Herigault G.
    "Dual-echo Dixon imaging with flexible choice of echo times."
    Magn Reson Med 2011;65(1):96-107.  doi:10.1002/mrm.22578

Only that paper's *abstract* was read while writing this module. **The
closed-form per-voxel derivation below is original to this code** and is
verified numerically by ``tests/test_dixon.py``; it is not a transcription of
the paper's algorithm and should not be cited as one.

Signal model
------------
Per voxel, with ``df`` the fat offset [Hz] and ``psi`` the field offset [Hz]::

    S_n = (W + F * exp(2j*pi*df*TE_n)) * exp(2j*pi*psi*TE_n) * exp(1j*phi0)

Two complex measurements (four real numbers) against four unknowns
``W, F, psi, phi0``: exactly determined, but with a discrete ambiguity.
:func:`separate_water_fat` returns *both* solutions per voxel.

Both solutions fit the data exactly
-----------------------------------
This is the honest statement of the two-point problem, and it holds at every
echo spacing: a single voxel can never choose between its two candidates, so the
choice has to be made *spatially* (see :func:`resolve_field_map`). In particular
180 deg is not a special degenerate angle -- the genuine degeneracy is ``W == F``,
where the two solutions coincide, and it depends on the mixture, not on the echo
times. The echo-time choice does matter for noise: see :func:`conditioning`.
REACT's 162.6 deg is kept for traceability to the published 1.5T protocol
(Isaak 2021), at a measured cost of about 1.2% more noise than 180 deg.

.. warning::
   **Never coil-combine by root-sum-of-squares before this.** RSS discards
   phase, and phase is the entire measurement. Combine with phase preserved
   (adaptive/Walsh or sensitivity maps), or separate per channel.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np


def fat_phasors(te1: float, te2: float, fat_freq: float) -> Tuple[complex, complex]:
    """Fat phase relative to water at each echo: ``exp(2j*pi*df*TE)``."""
    return (np.exp(2j * np.pi * fat_freq * te1),
            np.exp(2j * np.pi * fat_freq * te2))


class Conditioning(NamedTuple):
    """How well-posed the separation is for a given pair of echo times."""
    phase_deg: float            # water-fat phase evolved between the echoes
    det: float                  # |det A|, at most 2 (at 180 deg)
    noise_amplification: float  # relative to the 180 deg optimum
    alias_period_hz: float      # field-map ambiguity period, 1/dTE


def conditioning(te1: float, te2: float, fat_freq: float) -> Conditioning:
    """Noise properties of a two-point acquisition.

    With the field map demodulated, the problem is
    ``[[1, c1], [1, c2]] @ [W, F] = [s1, s2]``, so noise scales as ``1/|det A|``
    with ``|det A| = |c2 - c1| = 2*|sin(dphi/2)|``, maximal (2) at 180 deg.
    """
    c1, c2 = fat_phasors(te1, te2, fat_freq)
    dte = te2 - te1
    det = float(abs(c2 - c1))
    return Conditioning(
        phase_deg=float(np.rad2deg(np.angle(c2 / c1))),
        det=det,
        noise_amplification=float(2.0 / det) if det > 0 else float("inf"),
        alias_period_hz=float(1.0 / dte),
    )


class Candidates(NamedTuple):
    """The two per-voxel solutions of the two-point problem.

    Arrays are stacked on a leading axis of length 2: index 0 is the
    water-dominant solution, index 1 the fat-dominant one.
    """
    water: np.ndarray       # (2, ...) non-negative amplitudes
    fat: np.ndarray         # (2, ...)
    psi: np.ndarray         # (2, ...) field offset [Hz], defined modulo alias_hz
    residual: np.ndarray    # (2, ...) magnitude misfit; ~0 for both, by construction
    alias_hz: float         # field-map ambiguity period, 1/dTE
    magnitude: np.ndarray   # |s1|, so a resolver can ignore background


def separate_water_fat(s1: np.ndarray, s2: np.ndarray, te1: float, te2: float,
                       fat_freq: float) -> Candidates:
    """Both candidate water/fat solutions per voxel, in closed form.

    With ``p1 = exp(i(2*pi*psi*te1 + phi0))`` and ``b = exp(2j*pi*psi*dTE)``::

        s1 = (W + F*c1) * p1
        s2 = (W + F*c2) * p1 * b          =>   r = s2/s1 = b*(W + F*c2)/(W + F*c1)

    Because ``|b| = 1`` the *magnitude* of the ratio does not involve the field
    map. With ``t = F/W``::

        |r|^2 = |1 + t*c2|^2 / |1 + t*c1|^2
        (1 - |r|^2) t^2 + 2 (Re c2 - |r|^2 Re c1) t + (1 - |r|^2) = 0

    The leading and constant coefficients are equal, so **the two roots multiply
    to exactly 1**: they are ``t`` and ``1/t`` -- the water-dominant and
    fat-dominant solutions fall out of the algebra rather than being imposed.

    Each root is carried as an unnormalised pair ``(u, v)`` meaning
    ``u*water + v*fat``, with ``t = v/u``. Using the numerically stable
    quadratic formula, root A is ``(a, q)`` and root B is ``(q, a)`` with
    ``a = 1 - |r|^2`` and ``q = -(b + sign(b)*sqrt(disc))/2``; no division is
    needed, so ``|r| = 1`` (pure water or pure fat, where ``a = 0``) needs no
    special case. Amplitudes then follow from ``|s1| = |u + v*c1| * scale``,
    and the field map from the phase left over, ``b = r / rho``.

    ``W`` and ``F`` are returned as non-negative amplitudes; the common phase
    ``phi0`` is absorbed. If noise drives a root negative the pair is taken in
    absolute value.
    """
    s1 = np.asarray(s1, dtype=complex)
    s2 = np.asarray(s2, dtype=complex)
    if s1.shape != s2.shape:
        raise ValueError(f"echo shapes differ: {s1.shape} vs {s2.shape}")
    c1, c2 = fat_phasors(te1, te2, fat_freq)
    dte = te2 - te1

    has_signal = np.abs(s1) > 0
    r = np.where(has_signal, s2 / np.where(has_signal, s1, 1.0), 1.0)
    r2 = np.abs(r) ** 2

    a = 1.0 - r2
    b = 2.0 * (c2.real - r2 * c1.real)
    sq = np.sqrt(np.maximum(b * b - 4.0 * a * a, 0.0))   # noise can push disc < 0
    q = -0.5 * (b + np.where(b >= 0, 1.0, -1.0) * sq)

    waters, fats, psis, residuals = [], [], [], []
    for u, v in ((a, q), (q, a)):
        u, v = np.abs(u), np.abs(v)
        d1 = u + v * c1
        d2 = u + v * c2
        m1 = np.abs(d1)
        ok = m1 > 0
        rho = np.where(ok, d2 / np.where(ok, d1, 1.0), 1.0)
        bph = np.where(np.abs(rho) > 0, r / np.where(np.abs(rho) > 0, rho, 1.0), 1.0)
        psi = np.angle(bph) / (2.0 * np.pi * dte)

        scale = np.where(ok, np.abs(s1) / np.where(ok, m1, 1.0), 0.0)
        w, f = u * scale, v * scale
        res = ((np.abs(w + f * c1) - np.abs(s1)) ** 2
               + (np.abs(w + f * c2) - np.abs(s2)) ** 2)
        waters.append(w)
        fats.append(f)
        psis.append(psi)
        residuals.append(res)

    water, fat, psi, res = (np.stack(x) for x in (waters, fats, psis, residuals))
    swap = water[0] < fat[0]            # index 0 must be the water-dominant one

    def order(x):
        return np.stack([np.where(swap, x[1], x[0]), np.where(swap, x[0], x[1])])

    return Candidates(water=order(water), fat=order(fat), psi=order(psi),
                      residual=order(res), alias_hz=float(1.0 / dte),
                      magnitude=np.abs(s1))


def synthesize_in_opposed(water: np.ndarray, fat: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """In-phase and opposed-phase images built from W and F.

    Not cosmetic for REACT: the clinical literature uses them to *recognise*
    water/fat swaps. Pennig et al. (Clin Neuroradiol 2021) saw swap artifacts in
    10 of 35 patients at 3T and used the in-phase image to confirm each was an
    artifact rather than a real signal void.
    """
    return water + fat, np.abs(water - fat)
