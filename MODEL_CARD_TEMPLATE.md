# Model card (fill after held-out test)

## Model name, revision, date
- Architecture:
- Source-code Git commit:
- Framework / CUDA / driver:
- Checkpoint SHA256:

## Research task and scope
- Modality, anatomy, population, acquisition types:
- Exact class definitions:
- Intended research use: education / non-clinical research only
- Prohibited use: medical diagnosis, triage, or treatment decisions

## Data
- Publisher dataset title, DOI, license, dates, and citations:
- Original train/val/test partition source and subject IDs:
- Total distinct patients and slices per split, per class:
- Duplicate/leakage audit:
- Exclusion criteria / source bias:

## Preprocessing
- Modality-aware intensity scale:
- Input sizes and saved mean/std:
- Augmentations applied only in train:
- DICOM window if applicable and why:

## Training
- Initialization and hyperparameters:
- Random seed(s):
- Stopping criterion and validation strategy:
- Hardware and measured training runtime:

## Final evaluation (never choose hyperparameters using test)
- Image-level macro-F1, balanced accuracy, per-class precision/recall:
- Patient-level macro-F1 and confusion matrix:
- Confidence calibration metrics (optional):
- External validation (or explicitly not done):
- Known failure modes and subgroup limitations:

## Ethical limitations
- Predictions and scores are research outputs, NOT medical findings or calibrated risks.
- No clinical validation or regulatory authorization.
