# Candidate datasets and scope limitations

Check current terms on the publisher page before any download, training, or redistribution.
Do not commit raw data, clinical files, or patient identifiers to GitHub.

## MRI (recommended FIRST)

Jun Cheng et al., Figshare brain tumor MRI dataset, DOI 10.6084/m9.figshare.1512427.
- 3,064 T1-weighted contrast-enhanced **2D MRI slices from 233 patients**.
- 3 tumor types: meningioma, glioma, pituitary; **NO healthy/normal class**.
- Contains patient ID in the original .mat files; keep the same patient across splits.
- License on original hosting: CC BY 4.0; attribution required.
- Need actual publisher source .mat files, not a screenshot/JPG dump lacking patient IDs.

Source: https://doi.org/10.6084/m9.figshare.1512427

## X-ray (currently used: NIH ChestX-ray14)

NIH ChestX-ray14: 112,120 images from 30,805 patients, 14 findings and a separate
"No Finding" weak report-mined annotation. Official 86,524 train/validation and
25,596 test images; group-aware train/validation splitting uses supplied patient IDs.
Labels can overlap; "No Finding" is NOT confirmed healthy and pneumonia is rare.
Retain original partition lists and document NIH licensing/attribution.
Original NIH distribution is linked from https://nihcc.app.box.com/v/ChestXray-NIHCC
and https://www.kaggle.com/datasets/nih-chest-xrays/data (check provider terms).
The earlier Kermany pediatric pneumonia dataset is NOT the current training target.

## CT (third, NOT suitable for an unrestricted validated four-class result yet)

Common CT-KIDNEY Normal/Cyst/Stone/Tumor JPG dataset: images are publicly available,
but the exported release lacks patient IDs and original CT HU metadata. Therefore:
- Cannot prove patient-disjoint held-out evaluation on this data alone.
- Cannot recover true Hounsfield units from the exported JPG images.
- If used for exploration, label it a **slice-level exploratory benchmark** with
  explicit limits, not a clinically reliable kidney disorder detector.
- Seek a patient-indexed authorized DICOM dataset and task-validated ground truth before
  making robust claims. Public KiTS CT research datasets cover other specific renal tasks
  and must not be silently relabeled as the four original classes.

Example literature: https://link.springer.com/article/10.1007/s44443-026-01223-2

## Modality identification

Prefer verified DICOM metadata to determine modality when it is present.
A JPG/PNG modality classifier can be evaluated separately, but combining different
source datasets for different modalities risks learning scanner/site/format artifacts.
Report external-source testing and abstain on unsupported or uncertain input.

## DICOM / PHI

Datasets must be appropriately de-identified and licensed; never upload identifiable
patient information or unauthorized clinical scans. Even generated image arrays and
predictions stay local unless explicitly cleared for release.
