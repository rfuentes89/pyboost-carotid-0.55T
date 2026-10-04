"""Two-point Dixon water/fat separation for REACT's flexible echo times.

Why the classic formula does not apply
--------------------------------------
Textbook two-point Dixon assumes one echo exactly in phase and one exactly
opposed, so ``W = (S_in + S_opp)/2`` and ``F = (S_in - S_opp)/2`` on magnitudes.
REACT's echoes at 0.55T sit at about -120 and +78 deg of water-fat phase, and the
information lives in the *complex* signal; taking magnitudes first discards it.

Provenance -- read this before citing
-------------------------------------
Only *abstracts* were read for the papers below; none of them was read in full.
The closed-form per-voxel derivation in this module is original to this code and
is verified numerically by ``tests/test_dixon.py``. It is not a transcription of
any of these papers and should not be cited as one.

* Eggers H, Brendel B, Duijndam A, Herigault G. "Dual-echo Dixon imaging with
  flexible choice of echo times." Magn Reson Med 2011;65(1):96-107.
  doi:10.1002/mrm.22578. Cited for the concept REACT's mDIXON readout relies on:
  a two-point method freed from the opposed-phase echo-time restriction. Its
  abstract also reports that a more accurate fat spectral model substantially
  reduces variation in the fat suppression -- a limit of the single-peak model
  used here.
* Berglund J et al. "Two-point Dixon method with flexible echo times." Magn Reson
  Med 2011. Its abstract describes the same structure as this module: phase
  errors, mostly from static field inhomogeneity, must be removed before the
  least-squares water/fat estimate; the resulting ambiguity is resolved by a
  global optimization (message passing, versus the simpler region growing used
  here); noise in the estimates matches the Cramer-Rao bounds; and the signal
  model accounts for spectral broadening of the fat peak (not modelled here).
* Eggers H et al. "Chemical shift encoding-based water-fat separation
  methods." J Magn Reson Imaging 2014. A review covering the selection of key
  parameters and typical artifacts; the place to look before changing echo times.

Signal model
------------
Per voxel, with ``df`` the fat offset [Hz] and ``psi`` the field offset [Hz]::

    S_n = (W + F * exp(2j*pi*df*TE_n)) * exp(2j*pi*psi*TE_n) * exp(1j*phi0)

with ``W`` and ``F`` real. They may have opposite signs (REACT's inversion can
leave water negative while fat, past its null, is positive); see ``signed`` in
:func:`separate_water_fat`. Two complex measurements (four real numbers) against four unknowns
``W, F, psi, phi0``: exactly determined, but with a discrete ambiguity.
:func:`separate_water_fat` returns *both* solutions per voxel.

Both solutions fit the data exactly
-----------------------------------
This is the honest statement of the two-point problem, and it holds at every
echo spacing (for ``W`` and ``F`` of the same sign, or of either sign with
``signed=True``): a single voxel can never choose between its two candidates, so the
choice has to be made *spatially* (see :func:`resolve_field_map`). In particular
180 deg is not a special degenerate angle -- the genuine degeneracy is ``W == F``,
(or ``W == -F``) where the two solutions coincide, and it depends on the mixture,
not on the echo times. The echo-time choice does matter for noise: see :func:`conditioning`.
REACT's 162.6 deg is kept for traceability to the published 1.5T protocol
(Isaak 2021), at a measured cost of about 1.2% more noise than 180 deg.

.. warning::
   **Never coil-combine by root-sum-of-squares before this.** RSS discards
   phase, and phase is the entire measurement. Combine with phase preserved
   (adaptive/Walsh or sensitivity maps), or separate per channel.
"""

from __future__ import annotations

import heapq
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
    water: np.ndarray       # (2, ...) |W|, non-negative
    fat: np.ndarray         # (2, ...) |F|, non-negative
    psi: np.ndarray         # (2, ...) field offset [Hz], defined modulo alias_hz
    residual: np.ndarray    # (2, ...) magnitude misfit; ~0 for both, by construction
    alias_hz: float         # field-map ambiguity period, 1/dTE
    magnitude: np.ndarray   # |s1|, so a resolver can ignore background
    opposed: np.ndarray | None = None   # (2, ...) True where W and F have opposite sign


def separate_water_fat(s1: np.ndarray, s2: np.ndarray, te1: float, te2: float,
                       fat_freq: float, signed: bool = False) -> Candidates:
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

    ``signed=False`` (default) takes ``abs()`` of the roots, i.e. assumes ``W`` and
    ``F`` have the same sign. When they do not (``W=-1, F=0.5``) that returns a wrong
    field map (-45 Hz error there) and a large residual.

    ``signed=True`` keeps the signs: ``W`` and ``F`` are real and **may have opposite
    signs**, which REACT's inversion produces (water negative, fat past its null
    positive). They are returned as the non-negative amplitudes ``|W|`` and ``|F|``
    (the water image REACT shows) together with ``opposed`` (``W*F < 0``). The
    overall sign ``(W, F) -> (-W, -F)`` is absorbed in the unknown phase ``phi0`` and
    cannot be recovered; only the relative sign can. The two roots have the same
    sign, so the data fixes it and it adds no ambiguity.

    **Why it is not the default.** ``signed=True`` is exact when the two-point model
    holds, but it is less robust than ``abs()`` when it does not. The model has no
    inter-echo decay (T2*), and on the MRzero carotid phantom (T2' = 30 ms, so
    ``|s2/s1| ~ 0.77`` in pure tissue) the extra freedom is spent on explaining that
    decay with a small opposite-sign species, and at the default TI, where fat sits
    at its null and its voxels are near-degenerate mixtures, whole regions came out
    swapped (blood classified correctly in 1.7% of voxels against 100% with
    ``abs()``). At TI = 155 ms, where water and fat really do have opposite signs,
    the two agree on which species dominates. See ``docs/react_literature.md``, O12.

    Degenerate when ``t = +-1``, i.e. ``W == F`` or ``W == -F``: the two roots
    coincide. A voxel with ``W + F*c1 ~ 0`` has almost no signal at echo 1 and
    falls under the mask of :func:`resolve_field_map`.
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
    opps = []
    for u, v in ((a, q), (q, a)):
        if not signed:
            u, v = np.abs(u), np.abs(v)
        # Signs are kept: W and F are real but may have opposite signs (REACT's
        # inversion leaves water negative while fat, past its null, is positive).
        # The two roots of the quadratic multiply to +1, so both candidates have
        # the same sign of F/W and the data decides it -- no extra ambiguity.
        d1 = u + v * c1
        d2 = u + v * c2
        m1 = np.abs(d1)
        ok = m1 > 0
        rho = np.where(ok, d2 / np.where(ok, d1, 1.0), 1.0)
        bph = np.where(np.abs(rho) > 0, r / np.where(np.abs(rho) > 0, rho, 1.0), 1.0)
        psi = np.angle(bph) / (2.0 * np.pi * dte)

        scale = np.where(ok, np.abs(s1) / np.where(ok, m1, 1.0), 0.0)
        w, f = u * scale, v * scale          # signed; the global sign is not recoverable
        res = ((np.abs(w + f * c1) - np.abs(s1)) ** 2
               + (np.abs(w + f * c2) - np.abs(s2)) ** 2)
        # Meaningful only when both species are present; a pure-water or pure-fat
        # voxel has one of them at round-off level with an arbitrary sign.
        both = np.minimum(np.abs(w), np.abs(f)) > 1e-9 * np.maximum(np.abs(w), np.abs(f))
        opps.append((w * f < 0) & both)
        waters.append(np.abs(w))
        fats.append(np.abs(f))
        psis.append(psi)
        residuals.append(res)

    water, fat, psi, res, opp = (np.stack(x) for x in
                                 (waters, fats, psis, residuals, opps))
    swap = water[0] < fat[0]            # index 0 must be the water-dominant one

    def order(x):
        return np.stack([np.where(swap, x[1], x[0]), np.where(swap, x[0], x[1])])

    return Candidates(water=order(water), fat=order(fat), psi=order(psi),
                      residual=order(res), alias_hz=float(1.0 / dte),
                      magnitude=np.abs(s1), opposed=order(opp))


def synthesize_in_opposed(water: np.ndarray, fat: np.ndarray,
                          opposed: np.ndarray | None = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """In-phase and opposed-phase images built from |W|, |F| and their relative sign.

    Not cosmetic for REACT: the clinical literature uses them to *recognise*
    water/fat swaps. Pennig et al. (Clin Neuroradiol 2021) saw swap artifacts in
    10 of 35 patients at 3T and used the in-phase image to confirm each was an
    artifact rather than a real signal void.

    With ``opposed`` true (W and F of opposite sign) the roles swap: the echo
    where the species add is then ``|W| - |F|`` in magnitude, and the one where
    they cancel is ``|W| + |F|``. ``opposed=None`` means same sign everywhere.
    """
    same = water + fat
    diff = np.abs(water - fat)
    if opposed is None:
        return same, diff
    return np.where(opposed, diff, same), np.where(opposed, same, diff)


def _wrap(delta: float, alias: float) -> float:
    """|delta| folded into one alias period: the field map is only defined mod 1/dTE."""
    return abs((delta + alias / 2.0) % alias - alias / 2.0)


def _orient(pick: np.ndarray, psi: np.ndarray, alias: float, members: list) -> None:
    """Fix the one global ambiguity of a grown component, in place.

    Smoothness fixes every voxel *relative to the seed*, but cannot say whether
    the whole component is right or swapped: flipping every voxel to its other
    candidate shifts the field map by about the fat-water shift and is just as
    smooth. The component is therefore oriented by the only prior available --
    the field map is centred near zero after shimming -- measured as the
    **median** of ``|psi|`` over the whole component for both labelings, and the
    smaller one wins.

    A *global* statistic is deliberate. Deciding from one seed voxel is unsound:
    a pure-fat voxel at ``psi = +80 Hz`` is literally the same data as pure water
    at 0 Hz, so any single voxel can sit exactly where the prior is blind. The
    median over many voxels survives a localised excursion of the field map.
    """
    idx = tuple(np.array(members).T)
    k = pick[idx].astype(int)
    here = np.array([_wrap(psi[(int(kk),) + tuple(i)], alias)
                     for kk, i in zip(k, members)])
    other = np.array([_wrap(psi[(1 - int(kk),) + tuple(i)], alias)
                      for kk, i in zip(k, members)])
    if np.median(other) < np.median(here):
        pick[idx] = ~pick[idx]


def estimate_noise(magnitude: np.ndarray, k: float = 3.0,
                   n_iter: int = 20) -> float:
    """Noise sigma from the background of a magnitude image (Rayleigh).

    Background magnitude is Rayleigh with ``E[x^2] = 2*sigma^2``. The estimate
    iterates: take the voxels below ``k*sigma``, compute ``sqrt(mean(x^2)/2)``,
    and divide out the bias that truncating at ``k*sigma`` introduces,
    ``E[x^2 | x < k*sigma] = 2*sigma^2 * (1 - (k^2/2) e^{-k^2/2} / (1 - e^{-k^2/2}))``.

    It starts from the median of the darkest quarter of the image. That start is
    only a first guess -- it is biased high whenever less than all of the image
    is background (by 1.5x at 47% background, measured) -- but because it errs
    high, every background voxel is included on the first pass and the iteration
    converges onto the true value without depending on the background fraction.

    Needs *some* pure-background voxels. An image filled with tissue has none and
    returns an overestimate; pass ``noise_sigma`` explicitly in that case.
    """
    flat = magnitude.reshape(-1).astype(float)
    sigma = float(np.median(np.sort(flat)[: max(1, flat.size // 4)]) / 0.516)
    e = np.exp(-k * k / 2.0)
    bias = 1.0 - (k * k / 2.0) * e / (1.0 - e)
    for _ in range(n_iter):
        bg = flat[flat < k * sigma]
        if bg.size < 8:
            break
        new = float(np.sqrt(np.mean(bg ** 2) / (2.0 * bias)))
        if abs(new - sigma) <= 1e-9 * max(sigma, 1e-300):
            sigma = new
            break
        sigma = new
    return sigma


def resolve_field_map(cand: Candidates, threshold: float = 0.05,
                      noise_sigma: float | None = None,
                      noise_k: float = 4.0) -> np.ndarray:
    """Choose one candidate per voxel by region growing; ``True`` = fat-dominant.

    A voxel cannot choose between its two candidates (both fit exactly), but the
    field map is physically smooth, so a *wrong* choice shows up as a jump in
    ``psi`` across the water/fat boundary. For a pure-species voxel the wrong
    candidate is the right one displaced by exactly the fat-water shift
    (~79.6 Hz), which is why the decision has to be made between regions.

    Algorithm: starting from a seed, repeatedly resolve the unresolved voxel whose
    best candidate has the smallest ``psi`` jump from an already-resolved
    neighbour (a priority queue, so the most confident voxels are decided first
    and noisy ones last). ``psi`` jumps are measured modulo the alias period.

    **Assumption that real data must satisfy.** Smoothness cannot fix the
    *global* ambiguity: swapping every voxel and shifting the whole field map by
    the fat-water shift is perfectly smooth. It is broken with a prior -- the
    field map is assumed centred near zero after shimming, tested as the median
    of ``|psi|`` over each connected component (see :func:`_orient`). If a scan
    violates that, e.g. an unshimmed offset near the fat-water shift, water and
    fat come out swapped as a whole; ``tests/test_dixon.py`` documents this limit
    rather than hiding it. A field map that spans more than one alias period
    carries no such information at all, since its wrapped distribution is
    uniform whatever the offset.

    **Disconnected regions are resolved independently**, each seeded at its
    brightest voxel with the same prior. An isolated component therefore gets no
    help from its neighbours: a lone fat island is classified by its own
    ``|psi|`` alone, and is only as reliable as that prior.

    Voxels below the mask threshold are background: they are not resolved and
    default to the water-dominant candidate (their amplitudes are ~0 anyway). The
    threshold is the larger of ``threshold * max(|s1|)`` and ``noise_k * sigma``.
    The noise term is not optional polish: with only a relative threshold,
    background noise passes the mask once sigma approaches it, bridges separate
    regions through voxels whose ``psi`` is random, and lets the noise decide how
    a whole component is oriented -- measured as ~46% swapped voxels on 1 in 12
    noise draws at SNR 33. ``sigma`` is estimated by :func:`estimate_noise`
    unless ``noise_sigma`` is given.
    """
    mag = cand.magnitude
    pick = np.zeros(mag.shape, dtype=bool)
    sigma = estimate_noise(mag) if noise_sigma is None else noise_sigma
    cut = max(threshold * mag.max(), noise_k * sigma)
    mask = mag > cut if mag.max() > 0 else np.zeros_like(mag, bool)
    if not mask.any():
        return pick

    psi, alias = cand.psi, cand.alias_hz
    resolved = np.zeros(mag.shape, dtype=bool)
    shape = mag.shape

    def neighbours(idx):
        for axis in range(len(shape)):
            for step in (-1, 1):
                j = list(idx)
                j[axis] += step
                if 0 <= j[axis] < shape[axis]:
                    yield tuple(j)

    members: list = []

    def settle(idx, k):
        pick[idx] = bool(k)
        resolved[idx] = True
        members.append(idx)
        ref = psi[(k,) + idx]
        for nb in neighbours(idx):
            if mask[nb] and not resolved[nb]:
                costs = [_wrap(psi[(c,) + nb] - ref, alias) for c in (0, 1)]
                c = int(np.argmin(costs))
                heapq.heappush(heap, (costs[c], nb, c))

    heap: list = []
    # Growth only crosses masked-in voxels, so a mask with several connected
    # components needs one seed per component. Seeding only once silently leaves
    # every other component at the water-dominant default -- right for a water
    # region, wrong for a fat one -- so a test with one lucky seed can pass while
    # the algorithm is broken.
    while True:
        todo = mask & ~resolved
        if not todo.any():
            break
        members.clear()
        seed = np.unravel_index(np.argmax(np.where(todo, mag, -1.0)), shape)
        settle(seed, 0)                  # provisional; orientation is fixed below
        while heap:
            _, idx, c = heapq.heappop(heap)
            if resolved[idx]:
                continue
            settle(idx, c)
        _orient(pick, psi, alias, members)
    return pick


def select(cand: Candidates, pick: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """Apply a candidate choice, returning ``(water, fat, psi)``."""
    return (np.where(pick, cand.water[1], cand.water[0]),
            np.where(pick, cand.fat[1], cand.fat[0]),
            np.where(pick, cand.psi[1], cand.psi[0]))


def select_opposed(cand: Candidates, pick: np.ndarray) -> np.ndarray:
    """Relative-sign flag of the chosen candidate (``True`` = W and F opposite)."""
    return np.where(pick, cand.opposed[1], cand.opposed[0])


def separate(s1: np.ndarray, s2: np.ndarray, te1: float, te2: float,
             fat_freq: float, signed: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    """Candidates -> spatial resolution -> ``(water, fat, psi)``."""
    cand = separate_water_fat(s1, s2, te1, te2, fat_freq, signed=signed)
    return select(cand, resolve_field_map(cand, **kwargs))
