# Evaluation results — 2026-11-06

These are local research results, not clinical validation. Candidates were chosen on validation before their new test evaluations. No candidate was switched after seeing the new test scores.

## Selected checkpoint results

| Task | Model | Test set | Primary result | Other result |
|---|---|---|---|---|
| MRI | ConvNeXt-Tiny | 306 slices / 23 patients | Patient macro-F1 0.9521 | Image macro-F1 0.9136; accuracy 0.9314 |
| Kidney CT | ConvNeXt-Tiny | 1,184 images; patients unknown | Image macro-F1 0.9656 | Balanced accuracy 0.9639; accuracy 0.9755 |
| NIH X-ray | ResNet-18 | 25,596 images / 2,797 patients | Macro AP 0.2524 | Macro AUROC 0.7876; macro-F1 0.3003 |

MRI and CT metrics are multiclass; NIH is multilabel. AUROC, AP, F1 and accuracy are different quantities and must not be substituted for one another. NIH numbers are image-level, not patient-aggregated metrics. All 14 labels had valid AP/AUROC.

## Is it better?

- **MRI: no demonstrated improvement.** ConvNeXt-Tiny won validation patient macro-F1 (1.0000 versus 0.9608), but its test patient macro-F1 is 0.9521 versus the historical ResNet-18's 0.9608. Image macro-F1 is 0.9136 versus 0.9239. Both had 93.14% image accuracy. With only 23 test patients, small differences are not evidence of reliable superiority. Retain both results and the validation-selected candidate; do not select a different model based on this test.
- **CT: a more defensible exploratory protocol, not a comparable accuracy improvement.** Grouped ConvNeXt-Tiny narrowly won validation macro-F1 (0.9663 versus EfficientNet-B0's 0.9644). Only the winner was newly tested. The original ResNet-18's 100% score came from a different random image split with confirmed leakage: 192 exact duplicate groups crossed splits. That score is invalid evidence of independent performance. The grouped split removes 517 known exact copies, retains 11,929 unique-pixel images and uses 352 proxy groups. Patient overlap remains unknown; the near-duplicate audit was capped and the repartition is not a fresh external cohort.
- **X-ray: effectively unchanged test AUROC.** Updated ResNet-18 scored 0.787607 versus historical ResNet-18's 0.787696 (difference -0.000088). The updated run adds weighted loss, macro-AP selection, validation-fitted thresholds and richer reporting. The historical run used AUROC selection, so this is a descriptive comparison of different protocols, not a controlled architecture experiment or a claim of statistically significant improvement. Pneumonia remains particularly weak (AP 0.0457, F1 0.0877).

**Release recommendation:** publish as a transparent non-clinical research/training prototype, including the mixed/negative results. Do not publish a claim that all new models outperform the baselines, that MRI detects healthy scans, or that CT is patient-independent. These internal test sets have now been examined; new tuning requires a validation protocol and further credible external evaluation.

## Per-class NIH test results

Thresholds were fitted on validation for the selected checkpoint and applied unchanged to test. They are research decision rules, not diagnoses. Weak dataset annotations and rare findings limit interpretation.

| Finding | AP | AUROC | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.3051 | 0.7474 | 0.2879 | 0.5151 | 0.3693 | 0.6743 |
| Cardiomegaly | 0.3302 | 0.8802 | 0.4157 | 0.3667 | 0.3897 | 0.6620 |
| Effusion | 0.4929 | 0.8165 | 0.4238 | 0.6617 | 0.5167 | 0.7287 |
| Infiltration | 0.3750 | 0.6801 | 0.3417 | 0.6831 | 0.4555 | 0.5357 |
| Mass | 0.2744 | 0.7902 | 0.3134 | 0.3936 | 0.3490 | 0.8271 |
| Nodule | 0.1807 | 0.7080 | 0.2690 | 0.1701 | 0.2084 | 0.8345 |
| Pneumonia | 0.0457 | 0.6783 | 0.0538 | 0.2378 | 0.0877 | 0.2794 |
| Pneumothorax | 0.3769 | 0.8395 | 0.4195 | 0.4255 | 0.4225 | 0.7902 |
| Consolidation | 0.1480 | 0.7292 | 0.1586 | 0.4281 | 0.2315 | 0.6813 |
| Edema | 0.1584 | 0.8285 | 0.1914 | 0.2973 | 0.2329 | 0.5658 |
| Emphysema | 0.3408 | 0.8848 | 0.3016 | 0.5334 | 0.3853 | 0.7598 |
| Fibrosis | 0.0839 | 0.7974 | 0.1636 | 0.0621 | 0.0900 | 0.5684 |
| Pleural_Thickening | 0.1187 | 0.7431 | 0.1184 | 0.4313 | 0.1858 | 0.4516 |
| Hernia | 0.3026 | 0.9035 | 0.7143 | 0.1744 | 0.2804 | 0.7480 |

## Provenance and verification

Aggregate JSON reports, class reports, confusion matrices, train/validation histories, checkpoint SHA256 values, manifest text hashes, split counts and the captured environment are in [reports/2026-11-06](reports/2026-11-06). Checkpoints and the manifests themselves remain local. Model cards describe the narrow tasks and preprocessing. Hashes identify local artifacts; manifest hashes do not verify image contents.

Fresh checks: 24 unit/integration tests passed; dependency checks passed; CUDA matrix multiplication passed; each selected checkpoint passed a local CLI inference smoke check; all three held-out evaluation commands finished. The legacy missing-HU-window checkpoint evaluation path is covered by a regression test. PyTorch emitted upstream pin-memory deprecation warnings during the CUDA test suite; no test failed.

No external cohort, clinical review, calibration, subgroup evaluation, confidence intervals or statistical significance testing is provided. These omissions limit claims of reliability. The release excludes all image data, patient identifiers, prediction rows and model weights.

Local UI smoke checks used Playwright because the Browser plugin was unavailable. The title and nonblank screen, MRI/CT/X-ray upload predictions, empty-upload handling, and browser runtime errors were checked. No page runtime errors occurred. Desktop and mobile screenshots were inspected. [QA evidence](reports/2026-11-06/ui_smoke_check.json) records the environment. The [demo screenshot](docs/demo-synthetic.png) uses synthetic noise; its output is not a performance result.
