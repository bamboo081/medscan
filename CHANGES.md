# Evaluated research release — 2026-11-06

- Finished MRI ConvNeXt-Tiny, grouped CT EfficientNet-B0, and updated NIH ResNet-18 training; selected MRI ConvNeXt-Tiny and existing grouped CT ConvNeXt-Tiny using compatible validation metrics.
- Completed selected-checkpoint test evaluations. Added RESULTS.md, MODEL_CARDS.md, aggregate reports and artifact hashes; preserved mixed/negative comparisons and CT leakage limitations.
- Added optional exact float32 NIH preprocessing cache with fresh augmentation, cache invalidation and atomic writes. Restarted NIH with 12 workers, batch size 32 and 23 maximum epochs; early stopping finished after 9 epochs.
- Fixed root test discovery against the nested old source copy. Fixed legacy checkpoint evaluation when hu_window is absent, with regression coverage.
- Added MIT source licensing, a GitHub release/commit plan, CLI and local browser smoke checks. Datasets, predictions, weights and caches remain excluded.
- Removed redundant MRI/CT download ZIPs only after verifying every extracted non-macOS-metadata file by SHA256; original extracted inputs and split manifests were preserved.

The historical notes below describe checks performed during the previous source-only revision, not the current local release evidence.

# Code review and verification

## Audited source

Received `medscan-project-code(1).zip` (32 entries, 106 KB uncompressed source; **no actual datasets or trained weights**). Reviewed training, NIH, preprocessing, CT splitting, evaluation, inference, and tests.

## Corrected / improved

- Existing `prepare_ct.py` uses random image splitting despite previously confirmed exact cross-split duplicate leakage. New `prepare_ct_grouped.py` consumes original audit cache and creates an explicitly approximate grouped split without modifying originals or re-reading all the images.
- Default NIH training was fixed to ResNet-18, macro AUROC only, no class weighting or threshold metadata. Added architecture selection, capped training-only positive weights, validation macro AP for checkpoint selection, and per-class AP/AUROC/F1.
- `predict.py` previously used `softmax` indiscriminately and indexed missing `hu_window` on NIH checkpoints. Updated for sigmoid multilabel scores, old checkpoint compatibility, explicit modality selection, and nonclinical scope warnings.
- Added optional localhost research demo, validation-only model comparison, group leakage guard, and modern pretrained torchvision architecture options.
- Updated setup/docs for Ubuntu instead of Windows, with 8 GB VRAM-friendly initial batch sizes.

## Evidence from actual audit *metadata* (not raw scans)

- Source count: 12,446; duplicate-pixel copies removed: 517.
- New approximate CT split: 11,929 images, 352 proxy groups; 9,545 train, 1,200 val, 1,184 test.
- Known cached high-SSIM pairs grouped: 67.
- These numbers were computed from the **user-supplied audit metadata**. The new split has not been used for actual GPU training or evaluated on the complete raw CT images in this environment.
- Grouping by filename block and cached SSIM is only a *heuristic*. It is not proof of patient identity or independent medical performance. The old test set was already accessed.

## Tests run here

- `python -m pytest -q`: **20 passed** (CPU environment).
- `python -m compileall -q .`: passed.
- Syntax/CLI `--help` smoke checks: passed.
- One-epoch **synthetic** ResNet-18 CT training and evaluation: passed (CPU).
- One-epoch **synthetic** ResNet-18 NIH multi-label training and evaluation: passed (CPU).
- Old-format sample `.pt` checkpoint loading and inference: passed (CPU).

**Not tested here:** user's RTX 5060 CUDA execution, actual complete datasets, performance of new architectures on medical data, Gradio UI (optional dependency), or clinical validity. Test-set accuracy from synthetic examples has no meaning for actual research claims.
