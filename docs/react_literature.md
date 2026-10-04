# REACT at 0.55T: what the literature supports, and what it does not

Status of every claim behind `pyboost/react.py` and its helpers. Compiled from PubMed
and Consensus (abstracts), plus full text where open access allowed. **Not read:** the
full text of Yoneyama 2019 (abstract only), the Pennig/Hoyer 3T table, and the
per-tissue relaxation table of Campbell-Washburn 2019. Scite was unavailable (paid
plan); ScienceDirect, PMC web and Europe PMC were blocked by the proxy.

**No REACT paper at 0.55T, or at any low field, was found.** Nothing here has a
direct reference at this field strength.

Read in full: Isaak 2021, Pennig 2020 (stroke, CHD), Gietzen 2025, Erdem 2025.
Abstract only: Yoneyama 2019, Berglund 2011, Eggers 2011, Eggers 2014, Khodarahmi 2024,
Campbell-Washburn 2019, Castillo-Passi 2024, Pedraza 2025, Tian 2023.

## Verified

| Claim | Source |
|---|---|
| T2-prep -> non-selective inversion (short TI) -> spoiled non-balanced 3D dual-echo Dixon | Yoneyama 2019 (abstract), Isaak, Pennig, Erdem |
| Flip angle 15 deg in every protocol read | Isaak, Pennig, Gietzen, Erdem |
| T2-prep 50 ms at 1.5T/3T | Isaak, Pennig (stroke), Erdem, Gietzen (table) |
| Neck protocol untriggered (3T, 2:46 min); thoracic uses ECG + navigator | Pennig (stroke); Isaak, Gietzen |
| Echo times semi-flexible, set by the scanner to the minimum | Yoneyama (abstract), Erdem |
| Each shot starts near the k-space centre ("low-high profile order") | Gietzen |
| Water/fat swaps are common (16%, 29%, 40%) and detected on in/opposed-phase | Isaak, Pennig (stroke), Gietzen |
| Two-point Dixon with flexible TEs: field-map ambiguity, noise vs Cramer-Rao bound | Berglund 2011 |
| Region growing for water/fat separation at 0.55T | Tian 2023 (ISMRM) |
| BOOST at 0.55T exists | Paredes 2025, MRM 94:1982, doi:10.1002/mrm.30611 |

## Corrected (earlier statements that were wrong or unsupported)

1. Paredes 2025 exists (was marked unverified).
2. Castillo-Passi 2024 does **not** give a 50 ms T2-prep at 0.55T in its abstract;
   the attribution was removed from `ReactParams`. The 50 ms rests on the 1.5T/3T
   protocols.
3. "T2-prep 50 ms in every variant" is false: the modified REACT (Pennig, CHD) uses
   30 ms and no inversion. Gietzen 2025 is internally inconsistent (text: 30 ms, no IR;
   table: 50 ms, 4 refocusing pulses).
4. The REACT T2-prep has **4 refocusing pulses**; it was single-refocus here.
   `prep.t2_prep(n_refocus=...)` now offers 1/2/4/8/16 (MLEV-16 phases);
   `ReactParams.t2prep_refocus` defaults to 4. Paredes 2025 compares adiabatic, MLEV4 and
   MLEV8 at 0.55T and finds MLEV8 better.
5. "TI ~ 70 ms at 1.5T" is not a single value: Erdem gives an inversion delay of 7.8 ms
   (REACT), 12.2 ms (MTC-REACT); Isaak ~70 ms.
6. 162.6 deg is not a canonical Dixon phase: Isaak 162.6, Gietzen DTE 2.11 ms (~165 deg),
   Erdem DTE 2.62 ms (~205 deg), all minimum-TE auto settings.
7. The 700-800 ms efficiency "decoy" in the shot-interval sweep was a non-steady-state
   artefact (2 dummy shots); it vanishes with 12.
8. `set_react_ti` accepted a TI shorter than the inversion pulse plus spoiler (~8-9 ms),
   which silently produced a negative recovery delay. It now raises. Consequence: a TI of
   7.8 ms (original REACT) is not playable here without shortening the spoiler.

## Open

| # | Risk | Where it stands |
|---|---|---|
| O1 | **TI and its objective.** The literature spans 7.8-70 ms; the reference point of the Philips "inversion delay" is unknown. | TI sweep (`optimize_react_ti.py`, 5 ms grid, MRzero, 3.0 s interval): water objective (blood - muscle) peaks at 155 ms (95% plateau 125-160); the fat-penalising objective at 85 ms; blood - wall is flat 80-200 ms. The current default, 84.2 ms, gives 88% / 98% / 95% of those optima. Isolated: T2-prep alone 0.065, + IR at 155 ms 0.079, at 84 ms 0.070, at 12 ms 0.057 (worse than none), no prep 0.006. The model gives short-TI inversion no benefit, which is not what the literature reports; unexplained. Default unchanged. |
| O2 | **`shot_interval`.** Published practice is 1-2 heartbeats (Erdem chose 2: CNR 15.8/18.7/21.3 at 1/2/3 HB, +78%/+38% time). | Steady-state water objective, relative to 3.0 s: 1.5 s -> 57-58% contrast, 81-82% efficiency; 2.0 s -> 76-78%, 93-96%. Efficiency plateaus at 2.5-3.0 s. 3.0 s is outside published practice. Default unchanged. |
| O3 | Acceleration (CS/SENSE 3-10x in all protocols) not implemented. | Out of scope this round. |
| O4 | **Relaxometry untraced.** `TISSUE_PROPERTIES` comes from a Koma reference script. | `react_ti_sensitivity.py`: fat T1 183 -> 125-135 ms (Khodarahmi, ~47% of 1.5T) moves the fat null from 84 to 58-62 ms; muscle null 152 ms ranges 91-215 ms over T1 +-20%, T2 40-70 ms. Needs the Campbell-Washburn per-tissue table. |
| O5 | SNR at 0.55T unknown; the Dixon separation fails at SNR <= 5, frail near 7. | Measured on synthetic data only. |
| O6 | Flow-induced swaps: DTE is 2.7x longer than at 1.5T. | My hypothesis, unverified; synthetic validation has no flow. |
| O7 | Single-peak fat model. | Eggers 2011 reports multi-peak reduces variability. |
| O8 | Monopolar readout; Pedraza 2025 uses bipolar for Dixon cMRF at 0.55T. | Defensible, not backed by a source read. |
| O9 | Vector slew peaks at 27.4 T/m/s vs 25 per axis. | Check PNS/scanner limit. |

## References

- Yoneyama M et al. Magn Reson Imaging 2019;63:137-146. doi:10.1016/j.mri.2019.08.017
- Isaak A et al. J Cardiovasc Magn Reson 2021. doi:10.1186/s12968-021-00788-3
- Pennig L et al. Clin Neuroradiol 2020. doi:10.1007/s00062-020-00963-6
- Pennig L et al. J Cardiovasc Magn Reson 2020 (CHD). doi:10.1186/s12968-019-0591-y
- Gietzen C et al. Front Cardiovasc Med 2025. doi:10.3389/fcvm.2025.1532661
- Erdem S et al. Quant Imaging Med Surg 2025. doi:10.21037/qims-24-2199
- Paredes et al. Magn Reson Med 2025;94:1982. doi:10.1002/mrm.30611
- Berglund J et al. Two-point Dixon method with flexible echo times. Magn Reson Med 2011
- Eggers H et al. Magn Reson Med 2011. doi:10.1002/mrm.22578
- Campbell-Washburn AE et al. Radiology 2019. doi:10.1148/radiol.2019190452
- Varghese J et al. Front Cardiovasc Med 2023. doi:10.3389/fcvm.2023.1120982
