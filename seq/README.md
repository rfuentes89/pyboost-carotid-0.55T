# Ready-to-run `.seq` files (0.55T carotid)

Scanner-executable Pulseq sequences, 128×120, TE 3.51 ms / TR 7 ms, ECG-gated
(RR 1.3 s). Both pass PyPulseq `check_timing`. Regenerate with the scripts noted
below.

| File | Contrast | Key parameters | Duration |
| :--- | :--- | :--- | :--- |
| `boost_carotid_055T.seq` | BOOST black-blood (dual contrast) | adiabatic inversion, T2-prep 50 ms, TI 90 ms, FatSat 180°, imaging flips 110°/80° | 18.2 s (14 RR) |
| `mra_carotid_055T.seq` | Bright-blood MRA | T2-prep **79 ms**, imaging flip **110°**, centric ordering, FatSat 180° | 9.1 s (7 RR) |

The MRA parameters are the optimization operating point (imaging flip ~110° and
the CNR-efficiency T2-prep TE ~79 ms; see `README_pypulseq.md`).

Regenerate:

```bash
python scripts/write_boost_055T.py --out seq/boost_carotid_055T.seq --inversion adiabatic
python scripts/write_mra_055T.py   --out seq/mra_carotid_055T.seq
```

Notes:

- The BOOST black-blood export uses the **adiabatic** inversion (B1-robust, right
  for the scanner). MRzero can't simulate adiabatic pulses, so use
  `--inversion block` if you want to re-validate the exported file in MRzero.
- Confirm `max_grad` / `max_slew` in `scanner_055T()` against your actual system,
  and check global/PNS limits (`seq.calculate_pns(...)` with your gradient `.asc`)
  before running on hardware — PyPulseq's `check_timing` is per-axis only.
- Cardiac gating is emitted as a `physio1` trigger marker with a fixed RR; enable
  true prospective gating (trigger *wait*) on the scanner.
