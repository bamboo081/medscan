# Model cards

Measured 2026-11-06. Source license: MIT. Dataset and weight redistribution permissions are separate. No weights or source images are included in the release.

All checkpoints were validation-selected before the new test evaluations. Test results were not used to switch candidates. Patient-disjoint local splits do not establish external clinical performance. No calibration, external evaluation, subgroup audit or confidence intervals were computed.


## MRI — convnext_tiny

Original Figshare MAT-derived 2D contrast-enhanced MRI slices; glioma/meningioma/pituitary only, no normal class. Patient IDs retained. Test cohort is only 23 patients. Historical test had already been examined. Meningioma recall is 0.7465 in this run.

- Run: `mri_convnext_tiny_20261106`; best epoch 2 of 8 completed.
- Manifest text SHA256: `f83d454e12abd03e9486bd3c7ad88bcf1ed4603f249fc88a29cf7f514dae36f4`. This does not hash image contents.
- Checkpoint SHA256: `8a45a1919f5a93a85501a651848fc843d5201a67d7de10e1c762f57e4b368d93` (local file, excluded from release).
- Classes: glioma, meningioma, pituitary.
- Input size 224; mode `mri_slice`; mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]`.
- Exported X-ray/CT: per-image 0.5/99.5 percentile scaling; MRI uses foreground percentiles. Resize uses bilinear antialiasing. Three grayscale channels are replicated.
- Training augmentation: mild brightness/contrast only. Validation and test are unaugmented. X-ray cache is float32 before augmentation and normalization.
- Initialization: torchvision ImageNet weights; AdamW; learning rate 0.0001; weight decay 0.0001; seed 42; early-stopping patience 6; CUDA AMP; gradient clipping 1.0.
- Selection: validation patient macro-F1.
- Test primary metric `patient_macro_f1`: 0.952137.
- Detailed configuration, train-only class weights, split counts, validation metrics and test metrics: [`mri_provenance.json`](reports/2026-11-06/mri_provenance.json).

- Batch size 8; maximum epochs 25; training loss is class-weighted cross entropy.


Split counts: `{"test": {"images": 306, "patients": 23, "proxy_groups": null}, "train": {"images": 2451, "patients": 187, "proxy_groups": null}, "val": {"images": 307, "patients": 23, "proxy_groups": null}}`.


## CT — convnext_tiny

Exported kidney JPEGs; cyst/normal/stone/tumor. No verified patient IDs or true CT HU. Known exact copies were removed and cached reviewed similarity pairs grouped, but filename blocks are only proxies. The old test was examined and this repartition is not independent. Residual related-slice overlap and shortcut learning remain possible.

- Run: `ct_grouped_convnext_tiny`; best epoch 5 of 11 completed.
- Manifest text SHA256: `4078d2e9a58e966db51f299f402db29a84ddeb67b898f7196711d4550afd2b7d`. This does not hash image contents.
- Checkpoint SHA256: `11e2de368d573aec97d41c40e6170c5afb8bfcc16e62fe423e7a565172d53561` (local file, excluded from release).
- Classes: cyst, normal, stone, tumor.
- Input size 224; mode `ct_exported`; mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]`.
- Exported X-ray/CT: per-image 0.5/99.5 percentile scaling; MRI uses foreground percentiles. Resize uses bilinear antialiasing. Three grayscale channels are replicated.
- Training augmentation: mild brightness/contrast only. Validation and test are unaugmented. X-ray cache is float32 before augmentation and normalization.
- Initialization: torchvision ImageNet weights; AdamW; learning rate 0.0001; weight decay 0.0001; seed 42; early-stopping patience 6; CUDA AMP; gradient clipping 1.0.
- Selection: validation image macro-F1.
- Test primary metric `macro_f1`: 0.965581.
- Detailed configuration, train-only class weights, split counts, validation metrics and test metrics: [`ct_provenance.json`](reports/2026-11-06/ct_provenance.json).

- Batch size 8; maximum epochs 25; training loss is class-weighted cross entropy.


Split counts: `{"test": {"images": 1184, "patients": null, "proxy_groups": 35}, "train": {"images": 9545, "patients": null, "proxy_groups": 282}, "val": {"images": 1200, "patients": null, "proxy_groups": 35}}`.


## XRAY — resnet18

NIH ChestX-ray14, fourteen independent weak report-derived labels. No Finding is not proof of health. Official test list retained; train/validation patient split. Metrics are image-level; patient counts are descriptive. Test labels can be noisy; pneumonia AP is only 0.0457 and recall/precision depend on validation-fitted thresholds.

- Run: `nih_resnet18_ap_cached_20261106`; best epoch 3 of 9 completed.
- Manifest text SHA256: `3af13bfe5c605945952ecda01ade942d7543b19f15a74c721f3e577b0807b7f9`. This does not hash image contents.
- Checkpoint SHA256: `de28e079dda8474a56ebd2303c5d81416714d70cfe44f8d7d64a8b9c62500fce` (local file, excluded from release).
- Classes: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.
- Input size 224; mode `xray_exported`; mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]`.
- Exported X-ray/CT: per-image 0.5/99.5 percentile scaling; MRI uses foreground percentiles. Resize uses bilinear antialiasing. Three grayscale channels are replicated.
- Training augmentation: mild brightness/contrast only. Validation and test are unaugmented. X-ray cache is float32 before augmentation and normalization.
- Initialization: torchvision ImageNet weights; AdamW; learning rate 0.0001; weight decay 0.0001; seed 42; early-stopping patience 6; CUDA AMP; gradient clipping 1.0.
- Selection: validation macro average precision.
- Test primary metric `macro_average_precision`: 0.252367.
- Detailed configuration, train-only class weights, split counts, validation metrics and test metrics: [`xray_provenance.json`](reports/2026-11-06/xray_provenance.json).

- 12 workers, batch size 32, 23 maximum epochs; stopped at 9. Runtime 16m48s including cache construction. Per-label thresholds fitted on the selected validation predictions only, saved unchanged for test.


Split counts: `{"test": {"images": 25596, "patients": 2797, "proxy_groups": null}, "train": {"images": 73916, "patients": 23806, "proxy_groups": null}, "val": {"images": 12608, "patients": 4202, "proxy_groups": null}}`.
