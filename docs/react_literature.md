# REACT at 0.55T: what the literature supports, and what it does not

Status of every claim behind `pyboost/react.py` and its helpers. Compiled from PubMed
and Consensus (abstracts), plus full text where open access allowed. **Not read:** the
full text of Yoneyama 2019 (abstract only), the Pennig/Hoyer 3T table, and the
per-tissue relaxation table of Campbell-Washburn 2019. Scite was unavailable (paid
plan); ScienceDirect, PMC web and Europe PMC were blocked by the proxy.

**No REACT paper at 0.55T, or at any low field, was found.** Nothing here has a
direct reference at this field strength.

Read in full: Isaak 2021, Pennig 2020 (stroke, CHD), Gietzen 2025, Erdem 2025.
Abstract only (Paredes 2025 and Pedraza 2025 are used above but their read status was not recorded; treat as abstract-level; Khodarahmi's ~47% applies to non-fluid tissues and its use for fat is an extrapolation): Yoneyama 2019, Berglund 2011, Eggers 2011, Eggers 2014, Khodarahmi 2024,
Campbell-Washburn 2019, Castillo-Passi 2024, Pedraza 2025, Tian 2023.

## Verified

| Claim | Source |
|---|---|
| T2-prep -> non-selective inversion (short TI) -> spoiled non-balanced 3D dual-echo Dixon | Yoneyama 2019 (abstract), Isaak, Pennig, Erdem |
| Flip angle 15 deg in every protocol read | Isaak, Pennig, Gietzen, Erdem |
| T2-prep 50 ms at 1.5T/3T | Isaak, Pennig (stroke), Erdem, Gietzen (table) |
| Neck protocol untriggered (3T, 2:46 min); thoracic uses ECG + navigator | Pennig (stroke); Isaak, Gietzen |
| Echo times semi-flexible, set by the scanner to the minimum | Yoneyama (abstract), Erdem |
| Each shot starts near the k-space centre ("low-high profile order") | Gietzen (described there; **our `encode_order` does not do it**, see O11) |
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
| O1 | **TI and its objective.** The literature spans 7.8-70 ms; the reference point of the Philips "inversion delay" is unknown. | Re-scored over the whole scan (`optimize_react_ti.py`: real geometry, 12 dummy shots, disc vessel). Water objective (blood - muscle): optimum TI 145 ms for R = 3.0 mm (95% plateau 115-155) and **85 ms for R = 2.2 mm (65-95)**; the old single-sample metric gave 155. So the optimum depends on vessel size and there is no single TI. The current default 84.2 ms keeps 89% of the optimum at R = 3.0 and 99.9% at R = 2.2. Contrast by preparation (blood - muscle, R = 3.0 / 2.2): none 0.006 / 0.002; T2-prep alone 0.064 / 0.045; + IR 12 ms 0.059 / 0.059; + IR 84 ms 0.072 / 0.068; + IR 155 ms 0.081 / 0.039. A short-TI inversion is below T2-prep alone for the larger vessel and 32% above it for the smaller one, which narrows but does not remove the disagreement with published 7.8-70 ms delays. Default unchanged. |
| O2 | **`shot_interval`.** Published practice is 1-2 heartbeats (Erdem chose 2: CNR 15.8/18.7/21.3 at 1/2/3 HB, +78%/+38% time). | Steady state, water objective, disc vessel, relative to 3.0 s (contrast / efficiency). TI 84.2 ms: 1.5 s -> 58% / 82% (R 3.0), 60% / 85% (R 2.2); 2.0 s -> 76% / 94%, 80% / 98%. TI 155 ms: 1.5 s -> 57% / 80%, 28% / 40%; 2.0 s -> 78% / 96%, 63% / 77%. Efficiency optimum 3.0 s (R 3.0, either TI), 2.5 s (R 2.2, TI 84), 3.5 s (R 2.2, TI 155). 3.0 s is outside published practice; 2.0 s costs 2-6% efficiency at TI 84 ms for a third less time. Default unchanged. |
| O3 | Acceleration (CS/SENSE 3-10x in all protocols) not implemented. | Out of scope this round. |
| O4 | **Relaxometry untraced.** `TISSUE_PROPERTIES` comes from a Koma reference script. | `react_ti_sensitivity.py`: fat T1 183 -> 125-135 ms (Khodarahmi, ~47% of 1.5T) moves the fat null from 84 to 58-62 ms; muscle null 152 ms ranges 91-215 ms over T1 +-20%, T2 40-70 ms. Needs the Campbell-Washburn per-tissue table. |
| O5 | SNR at 0.55T unknown; the Dixon separation fails at SNR <= 5, frail near 7. | Measured on synthetic data only. |
| O6 | Flow-induced swaps: DTE is 2.7x longer than at 1.5T. | My hypothesis, unverified; synthetic validation has no flow. |
| O7 | Single-peak fat model. | Eggers 2011 reports multi-peak reduces variability. |
| O8 | Monopolar readout; Pedraza 2025 uses bipolar for Dixon cMRF at 0.55T. | Defensible, not backed by a source read. |
| O9 | Vector slew peaks at 27.4 T/m/s vs 25 per axis. | Low priority: Siemens limits slew per axis and runs its own stimulation check. |
| O10 | **The old TI / shot-interval metric saw only the first TR.** `react_dc_signal` returns the k-space-centre sample, acquired in TR 0 of the first shot, so it scored `abs(Mz(TI))`. | Replaced by `react_object_signal`, the amplitude at the centre of a uniform disc vessel over the whole scan (checked against an explicit 113-voxel disc: 0.0931 vs 0.0922). An independent reviewer's analytic model predicted that a shot-averaged metric would reverse the ranking and make a short TI best. With the disc metric that is **not** reproduced at R = 2.2-3.0 mm: the optimum moves from 155 to 145 ms (R 3.0) and 85 ms (R 2.2), and short TIs stay below the optimum. The reviewer's metric corresponds to an unweighted average over k-space (a point-like vessel, R -> 0); that limit was not run here. |
| O11 | **Segmented-centric ordering.** `shot_encodes` gave each shot a contiguous block of the centre-sorted list, so only shot 0 started at the centre (ky 60, 71, 82, ...). | Fixed: with `centric` the shots are interleaved (shot s takes entries s, s+N, s+2N, ...), so each shot starts next to the centre and climbs to high radius; the train is 20 lines for 120 encodes at `tfe_factor` 22. **This is my interpretation**: Gietzen 2025 describes the low-high order but not the division between shots, and no 0.55T paper found describes the order of a REACT-type readout (0.55T BOOST papers use a different sequence, bSSFP with variable-density spiral-like Cartesian sampling). Spogis 2025 reports Cartesian REACT, with stack-of-stars in the abdomen. |
| O12 | **Two-point Dixon assumes water and fat of the same sign.** The inversion can leave water negative and fat near zero or positive. Verified on synthetic voxels (TE 4.18/9.85 ms, +5 Hz): W=-1, F=0.5 returns water 1.52 and a field map error of -45 Hz; W=-0.3, F=1 returns -76 Hz. The fit residual is large (0.68, 0.23) so it is detectable, but nothing checks it. The "both candidates fit exactly" statement in `dixon.py` holds only for same-sign pools. | Verified; not fixed. |
| O13 | Library default `scanner_055T()` is 26 mT/m, 45 T/m/s, above the user's 23/25. REACT's own fallback now uses 23/25/20 us; BOOST/MRA still use the library default. | REACT fixed; BOOST/MRA are the user's call. |

## Vessel size used for scoring (ultrasound, healthy adults; abstracts only, none is MRI or 0.55T)

| Source | n | Common carotid | Internal carotid |
|---|---|---|---|
| Limbu 2006 | 123 | 5.78 +- 0.57 / 5.86 +- 0.66 mm (range 4.3-7.7) | - |
| Ojaare 2021 | 400 | 6.39 / 6.28 mm | 4.63 / 4.61 mm |
| Nikolenko 2024 | 865 | 5.5-5.7 mm | up to 4.1-4.2 mm |
| Kpuduwei 2021 | 104 | 6.1 +- 0.8 mm | 6.0 +- 0.8 mm (discordant) |

Radii used: R = 3.0 mm (common carotid, ~6 mm) and R = 2.2 mm (internal carotid,
~4.4 mm). The phantom's 4 mm radius ("~8 mm", no source) is larger than every value above.

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
- Limbu YR et al. Nepal Med Coll J 2006 (carotid dimensions by ultrasound)
- Ojaare MG et al. Int J Adv Med 2021 (carotid diameter, ultrasonography)
- Nikolenko V et al. Regional blood circulation and microcirculation 2024
- Kpuduwei S et al. Folia Morphol 2021 (reference luminal diameters, Nigerian adults)
- Spogis J et al. Rofo 2025 (REACT in children; Cartesian order, stack-of-stars in abdomen)
