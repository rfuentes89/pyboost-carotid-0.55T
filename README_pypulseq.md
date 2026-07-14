# pyboost — scanner-executable BOOST carotid sequence (PyPulseq, 0.55T)

Python / [PyPulseq](https://github.com/imr-framework/pypulseq) port of the
KomaMRI reference simulation (`RR_sim.jl`, `FA_sim.jl`) so the **BOOST**
(bright-blood + black-blood *phase-sensitive*) carotid sequence can be exported
as a real, scanner-executable `.seq` file at **0.55T**.

The Koma scripts are a *contrast* simulation on a 1D isochromat model (one ADC
sample per TR, no spatial encoding). `pyboost` keeps the physics of every
preparation block and adds the real spatial encoding a scanner needs: a fully
balanced segmented **bSSFP** readout with frequency + phase encoding, an iNAV
start-up ramp, and the ECG-gated dual-contrast BOOST structure.

## Layout

```
pyboost/
  system.py     scanner_055T() -> pypulseq Opts (B0=0.55, Free.Max-like limits)
  params.py     BoostParams dataclass (mirrors seq_params + imaging geometry)
  prep.py       fat_sat, t2_prep, adiabatic_ir   (mirror RR_sim.jl:60-124)
  readout.py    bssfp_readout  (balanced bSSFP + iNAV ramp; the new encoding)
  boost.py      build_boost_sequence  (dual-contrast, gated assembly)
  phantom.py    concentric 2D carotid phantom (lumen+wall+muscle+fat) -> MRzero
scripts/
  write_boost_055T.py       build -> check_timing/test_report/SAR -> write .seq
  validate_with_mrzero.py   import .seq -> simulate 0.55T tissues -> compare contrast
  image_carotid_phantom.py  simulate on the 2D phantom -> reconstruct contrasts + BB
tests/          pytest unit + integration tests
```

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt          # pypulseq is required; MRzeroCore is optional
```

## Use

```bash
# Build the executable sequence and run the safety/timing checks:
python scripts/write_boost_055T.py --out boost_055T.seq

# Cross-validate the .seq contrast against the Koma reference (needs MRzeroCore):
python scripts/validate_with_mrzero.py

# Image the 2D carotid phantom and reconstruct bright / reference / black-blood:
python scripts/image_carotid_phantom.py --out carotid_boost.png

# Sweep the inversion time to find the black-blood (blood-null) operating point:
python scripts/optimize_ti.py --out ti_sweep.png

# Bright-blood MRA of the carotid phantom (lumen bright, background suppressed):
python scripts/image_mra_phantom.py --out mra_carotid.png

# Optimize the T2-prep echo time for blood-to-muscle contrast:
python scripts/optimize_mra.py --out mra_t2prep_sweep.png

# Differentiable optimization of the imaging flip angle (real MRzero gradient):
python scripts/optimize_flip.py --out flip_opt.png

# Joint differentiable optimization of flip angle AND T2-prep TE:
python scripts/optimize_joint.py --out joint_opt.png

# Run the test suite:
python -m pytest tests/ -q
```

`write_boost_055T.py` prints the PyPulseq `test_report` (TE=3.51 ms, TR=7 ms,
128×120 Cartesian, iNAV flip ramp 3.2→110°) and writes a `.seq` that passes
`check_timing`.

## Physics at 0.55T

* **Field strength** enters through `Opts(B0=0.55)` and the fat offset
  `Δf = -3.4 ppm · B0 · γ ≈ -80 Hz` (vs ≈ -220 Hz at 1.5T). The narrow low-field
  fat–water gap is why the FatSat pulse is long (~27 ms) and its bandwidth
  tight; validation confirms it still suppresses fat ~5× in the reference
  contrast.
* **Hardware limits** default to a Siemens MAGNETOM Free.Max profile
  (~26 mT/m, ~45 T/m/s). Koma's decorative `Smax=25e-3` will *not* close the
  TR=7 ms / TE=3.51 ms bSSFP timing (the slice-select ramp alone eats ~0.75 ms).
  Set `scanner_055T(max_grad=…, max_slew=…)` for your scanner.
* **Adiabatic HS inversion** (β=670, μ=5, 10.24 ms, BW≈1066 Hz) is B1-insensitive
  and affordable here — SAR sits far below the IEC limit at 0.55T.
* **bSSFP** is favourable at low field: reduced off-resonance means fewer/wider
  banding artifacts for a given TR.

## Carotid phantom & imaging (`pyboost.phantom`, `image_carotid_phantom.py`)

A concentric 2D neck cross section — circular **lumen** (blood), a vessel-**wall**
annulus, surrounding **muscle**, and a subcutaneous **fat** rind — on a pixel
grid that matches the imaging FOV, with 0.55T PD/T1/T2 per tissue. Air is
excluded to keep the voxel count down. `image_carotid_phantom.py` writes a
single `.seq`, simulates it on this phantom in MRzero, reconstructs both BOOST
contrasts (`reco_adjoint`, per-line bSSFP phase demodulated) and their
subtraction, and saves a 4-panel figure. Two findings worth keeping in mind:

* **Short acquisition window.** A single long readout (train ≫ fat T1 = 183 ms)
  lets fat recover and defeats the FatSat. The multi-shot default (`im_segments`
  lines/heartbeat, FatSat reapplied each heartbeat) keeps the window ~110 ms and
  the fat suppression visible (fat ring goes bright→dark between the two
  contrasts).
* **Fat modelled on-resonance.** Placing fat at its true −80 Hz offset also drops
  it onto the bSSFP frequency-response profile (80 Hz × 7 ms ≈ 200°/TR) and
  corrupts its readout signal, masking the FatSat effect in this single-peak
  model. The default keeps fat on-resonance (`apply_fat_offset=False`), which
  reproduces the validated ~3–6× FatSat suppression. A chemical-shift-consistent
  multi-peak fat model is a deferred refinement.

## Inversion pulse & TI optimization (`optimize_ti.py`)

**Key cross-simulator finding.** MRzero's PDG models every RF pulse as an
instantaneous rotation, so it does **not** reproduce adiabatic inversion — a
hyperbolic-secant pulse *saturates* (Mz→0) instead of inverting (Mz→−1) in the
simulation. The KomaMRI reference (full Bloch) does invert correctly. The
inversion is therefore selectable (`BoostParams.inversion_kind`):

* `"block"` (default) — a hard 180° pulse, which MRzero simulates correctly and
  which is perfectly usable at 0.55T (good B1 homogeneity, low SAR);
* `"adiabatic"` — the B1-robust HS pulse for the exported `.seq` / KomaMRI, but
  **not** for MRzero validation.

With the block inversion the pipeline reproduces textbook inversion recovery, and
`optimize_ti.py` sweeps TI to find the **blood-null** operating point. Two more
points that make the black-blood contrast actually appear:

* **Phase-sensitive readout.** A magnitude `|c0 − c1|` cannot null blood (the
  longest-T1 tissue); the script uses the signed projection
  `Re(c0·c1*)/|c1|`, so blood goes dark exactly at its inversion null while
  shorter-T1 wall/muscle (already past their nulls, opposite sign) stay bright.
* **Centric ordering** (`BoostParams.centric`) samples ky=0 right after the prep,
  before the bSSFP transient washes the prepared contrast out; with linear
  ordering the black-blood contrast is lost by the time k-space centre is reached.

At 0.55T this yields a blood-null TI of ~0.5 s for a single IR (T1_blood≈1.1 s).

**The fundamental ceiling (demonstrated by `image_carotid_phantom.py`).** Nulling
blood by inversion darkens the *whole vessel*, not the lumen selectively, because
blood (T1≈1122 ms) and vessel **wall** (T1≈750 ms) are too close in T1 — at the
blood-null TI the wall is dark too (lumen ≈ wall in every simulated contrast). The
spatial image is therefore a T1-weighted IR picture (long-T1 vessel dark vs
short-T1 muscle bright), **not** true black-blood. Real carotid black-blood
suppresses the lumen through **flow** (blood physically leaves the imaging
slice), which none of the available simulators model. **Bottom line: simulation
here validates the prep-pulse and T1/relaxation physics; the flow-void lumen-vs-
wall contrast that defines vessel-wall imaging can only be obtained on the
scanner (or with a flowing-spin simulation).**

## Bright-blood MRA (`pyboost.mra`, `image_mra_phantom.py`)

Where black-blood needs flow (not simulatable), **bright-blood MRA rests on
relaxation**, which the simulators reproduce faithfully. `build_mra_sequence`
assembles a single bright-blood contrast per heartbeat: `[inversion?] -> T2-prep
-> FatSat -> bSSFP readout`. The contrast mechanism, verified per tissue at
0.55T:

| stage | blood | muscle | blood/muscle |
| :---- | :---- | :----- | :----------- |
| bSSFP only        | 0.42 | 0.36 | 1.16 |
| + T2-prep         | 0.34 | 0.14 | **2.46** |
| + T2-prep + FatSat| 0.19 | 0.08 | **2.55** |

The **T2-prep** is the driver: blood's long T2 (~263 ms) is retained while muscle
(T2 ~55 ms) decays, so blood becomes the brightest tissue. `image_mra_phantom.py`
reconstructs a real image where the **lumen is bright** and muscle/fat are
suppressed. This is the natural, simulatable counterpart to the black-blood
ceiling above.

`optimize_mra.py` sweeps the T2-prep TE. The blood-to-muscle contrast rises with
TE and plateaus around **TE ≈ 80–120 ms** (muscle fully suppressed) in this
noiseless model. In practice a shorter TE (~40–60 ms) is used — blood signal at
TE=120 ms is only ~60% of its TE=0 value, and long T2-preps are more sensitive to
B1/B0, motion and diffusion (not modelled here). Relaxation values live in one
place (`pyboost.phantom.TISSUE_PROPERTIES`; muscle at 0.55T = T1 450 / T2 55 ms)
and the sweep/validation scripts import them.

Note: `reco_adjoint` returns the image transposed w.r.t. the phantom (x,y) grid;
the imaging scripts transpose it back so tissue statistics align with the label
map.

## Differentiable optimization through MRzero (`pyboost.diffopt`, `optimize_flip.py`)

MRzero's simulation is differentiable, but our `.seq` round-trip freezes the pulse
angles as constants on import. The trick in `pyboost.diffopt`: import the sequence
(so gradients, timing and spoilers stay correct), then **replace the imaging
pulses' `pulse.angle` with a torch tensor**. `mr0.util.simulate` is then
differentiable w.r.t. the flip angle, so it can be optimized by gradient descent
through the *real* simulator — no surrogate model.

`optimize_flip.py` runs Adam on the bSSFP flip angle (T2-prep fixed at 60 ms) to
maximize blood-to-muscle contrast. It converges into the contrast peak at
**~100–120°**, matching a brute-force MRzero sweep. The blood/muscle *ratio* is
nearly flat (~1.9) across flips, so the flip mainly trades absolute signal (SNR)
against SAR/banding — ~90–110° is a sensible practical choice.

`optimize_joint.py` extends this to a **2D** optimization of flip *and* T2-prep TE
at once: the imaging pulses' angle is one torch tensor, and the T2-prep delay
`event_time` is scaled by another (MRzero relaxes over `event_time`, so TE is
differentiable too). Adam converges to **flip ≈ 110°, TE ≈ 95–110 ms** on the 2D
contrast landscape (again, pure contrast pushes TE up; ~60 ms is the practical
SNR/robustness choice).

Notes: the imaging flip is optimized with the iNAV catalyzation ramp held fixed
(coupling the ramp to the flip flattens the optimum toward higher angles); a
compact differentiable Bloch surrogate we first tried disagreed with the full
simulation (predicting ~90°), which is exactly why the optimization differentiates
through MRzero itself rather than a model.

## Scope & caveats (this first iteration)

* **No true blood flow.** Neither Koma nor MRzero models spin inflow/outflow, the
  physical basis of black-blood contrast. The prep pulses reproduce the
  *magnetization* behaviour (validated), which is what you optimize; genuine
  flow-void fidelity requires a flowing-spin simulation or the scanner.
* **Simultaneous-axis slew.** Per-axis gradient/slew respect the limits, but the
  vector sum in the prephaser/rewinder blocks reaches ~70 T/m/s. PyPulseq checks
  per axis (what most vendors spec); verify against your system's global/PNS
  limits before running (`seq.calculate_pns(...)` with your gradient hardware
  `.asc`).
* **iNAV** is currently the flip-angle start-up ramp only (no 2D image
  navigator for motion correction).
* **Cardiac gating** is emitted as a `physio1` trigger marker with a fixed RR;
  true prospective gating (trigger *wait*) must be enabled on the scanner.

## Roadmap

1. Differentiable optimization of `im_flip_angle`, `ir_inversion_time`,
   `fatsat_flip_angle` (marked "To be optimized" in `RR_sim.jl`) via
   MRzero / Pulseq-zero — including a true phase-sensitive black-blood
   reconstruction that nulls the lumen (the current `|c0 - c1|` is a
   placeholder subtraction).
2. 2D iNAV image navigator for motion correction.
3. Chemical-shift-consistent multi-peak fat model in the phantom.
4. Quantitative Koma-vs-PyPulseq contrast comparison (run `RR_sim.jl` in Julia).
