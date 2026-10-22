# Pulseq versions: what the scanner update to 1.5.x means here

The scanner's Pulseq interpreter was first reported as v1.4.2 and then as updated to
"1.5.2" (as reported by the user). This page records what was checked, what was not,
and the evidence.

## What the sources say (read, not inferred)

* Pulseq releases page (github.com/pulseq/pulseq/releases; the page shows day and
  month without years). Releases listed: 1.4.0, 1.4.1, 1.4.2, **1.5.0, 1.5.1**. **No
  1.5.2 appears**, so nothing is known here about what a "1.5.2" interpreter changes.
  * **1.5.0**: new format revision. RF storage revised (RF centre position,
    intended use, frequency/phase offsets); arbitrary gradients revised (first and
    last points on the gradient raster edge); ADC storage revised (ppm offsets and an
    optional per-sample phase offset); a soft-delay extension; "mechanisms for
    consistent version handling of loaded files".
  * **1.5.1**: rotation extension, RF shimming extension, trigger plotting. *"This
    release requires an update of the interpreter to v1.5.1."*
  * Not stated anywhere I could read: whether an interpreter 1.5.x accepts 1.4.x files.
* PyPI, pypulseq: 1.4.2 (Jun 2024), 1.4.2.post1 (Mar 2025), 1.4.2.post2, 1.5.0 (Jan
  2026), 1.5.0.post1 (Jan 2026, MIT licence, "fixed backward compatibility").
  PyPulseq 1.5.0 is described as compatible with Pulseq interpreter sequences with
  version >= 1.5.0; features introduced in Pulseq 1.5.1 are "not yet supported".
  `Sequence.write` has a `v141_compat` option.
* A third-party bug report (MaRGE, March 2026) was seen only through a search snippet,
  not read: it describes an interpreter written for PyPulseq 1.4 failing on files from
  1.5. Treat as a hint that the formats are not interchangeable everywhere, not as a fact
  about the Siemens interpreter.

## What was verified in this repo (pypulseq 1.4.2 against 1.5.0.post1)

* Same code, both versions: REACT 2D (block and adiabatic inversion) and the 3D
  208 x 208 x 88, R = 5 protocol give **identical block counts, block durations and
  structure** (rf/gx/gy/gz/adc ids), `check_timing()` passes in both, and the 3D
  protocol lasts 507.0 s in both.
* The file header changes to `1.5.0`. `[RF]` records gain centre, ppm and phase
  offsets and a `use` letter; `[ADC]` records gain ppm/phase fields. Trapezoids,
  shapes and block layout are unchanged.
* Our pulses already carry `use`: excitation `e`, T2-prep pulses `p`, inversion `i`.
  The T2-prep refocusing pulses are tagged `preparation`, not `refocusing`; whether the
  scanner treats those differently is unknown.
* `Sequence.write(v141_compat=True)` from 1.5.0.post1 produces a file whose body is
  identical to the 1.4.2 output; only the header revision, a `TotalDuration`
  definition and the hash differ.
* MRzero 1.1.1 imports both file formats and gives the same simulated signal (maximum
  difference 8e-9, i.e. 8e-8 relative), with either pypulseq version installed.
* The full test suite passes (159 passed, 4 expected failures) with both versions.
* pypulseq 1.5 warns when an RF delay is implied by the dead time. The REACT readout
  now passes `delay=system.rf_dead_time` explicitly, which is what 1.4.2 did silently;
  output unchanged.

## What was not verified (needs the scanner)

* That the scanner's interpreter accepts files in format 1.5.0, 1.4.1 or both.
* What "1.5.2" is, and whether it differs from 1.5.1 for the features used here.
* That the centre of the adiabatic pulse (5120.5 us, now stored explicitly in 1.5 files)
  is honoured the way the simulation assumes.

## None of the new 1.5 features is used

No soft delays, rotation extension or RF shimming. Nothing in the sequence design
changes with the update; only the file format written and the pypulseq pin do.

## Suggested check on the scanner

Load the same small sequence in both layouts and see which the interpreter accepts:

    python scripts/write_react_055T.py --nx 64 --ny 32 --tfe 16 --format native --out t15.seq
    python scripts/write_react_055T.py --nx 64 --ny 32 --tfe 16 --format 1.4   --out t14.seq

If only one loads, tell me which and I will make it the default.
