# Carotid MRI pulse sequences at 0.55T

Design, simulation and optimization of carotid pulse sequences for **low-field
(0.55T)** MRI — both **BOOST** black-blood vessel-wall imaging and **bright-blood
MRA** — spanning a KomaMRI reference and a scanner-executable PyPulseq
implementation with differentiable optimization.

> *Simulación de sangre negra y angiografía para la carótida a 0.55T.*

## Two components

| | Language | What it is |
| :-- | :-- | :-- |
| **KomaMRI reference** | Julia | `RR_sim.jl`, `FA_sim.jl` — the original BOOST *contrast* simulation (1D isochromat model, full Bloch): heart-rate and flip-angle sweeps for black-blood. |
| **`pyboost`** | Python | A [PyPulseq](https://github.com/imr-framework/pypulseq) port that produces a real, **scanner-executable `.seq`**, plus a 2D carotid phantom, [MRzero](https://github.com/MRsources/MRzero-Core) validation, and gradient-based (differentiable) sequence optimization. |

The Julia scripts establish the physics and contrast; `pyboost` adds the spatial
encoding a scanner needs and the optimization tooling. **See
[`README_pypulseq.md`](README_pypulseq.md) for the full Python documentation.**

## What it does

- **Executable BOOST sequence** — balanced segmented bSSFP readout, iNAV ramp,
  T2-prep + adiabatic/block inversion + fat-sat, ECG-gated dual contrast; exports
  a `.seq` that passes PyPulseq timing/SAR checks (TE 3.51 ms, TR 7 ms).
- **2D carotid phantom** — concentric lumen + wall + muscle + fat with 0.55T
  relaxation values, fed to MRzero.
- **Bright-blood MRA** — T2-prep + fat-sat + bSSFP; lumen bright, background
  suppressed (blood/muscle ≈ 2.6, relaxation-driven).
- **Differentiable optimization** — of the imaging flip and T2-prep TE, by
  gradient descent *through the real MRzero simulation*.

## Quick start

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/write_boost_055T.py --out boost_055T.seq   # build + check + export
python scripts/image_mra_phantom.py --out mra_carotid.png # bright-blood MRA image
python -m pytest tests/ -q                                # 26 tests
```

Full command list and the physics/optimization write-up: **[`README_pypulseq.md`](README_pypulseq.md)**.

## Key findings (0.55T)

- **Black-blood needs flow, not T1.** Blood (T1≈1122 ms) and vessel wall
  (T1≈750 ms) are too close in T1, so inversion darkens the *whole vessel*, not
  the lumen selectively. True black-blood relies on flow (blood leaving the
  slice), which no available simulator models — simulation validates the
  prep/relaxation physics, not the flow-void contrast.
- **Bright-blood MRA is fully simulatable** (relaxation-based): the long-T2 blood
  survives the T2-prep while muscle decays.
- **MRzero can't simulate adiabatic pulses** (its PDG treats RF as instantaneous
  rotations, so a hyperbolic-secant *saturates* instead of inverting); the
  KomaMRI Bloch reference does. Use a block inversion for MRzero, adiabatic for
  the scanner/Koma.
- **Differentiate through the simulator, not a surrogate** — a simplified Bloch
  model predicted the wrong optimal flip; differentiating through MRzero itself
  gives the trustworthy answer (flip ≈ 100–110°).

## Repository map

```
RR_sim.jl, FA_sim.jl        KomaMRI reference (Julia)
pyboost/                    PyPulseq package: system, params, prep, readout,
                            boost, mra, phantom, diffopt
scripts/                    build/validate/image/optimize entry points
tests/                      pytest suite
README_pypulseq.md          full Python documentation
```
