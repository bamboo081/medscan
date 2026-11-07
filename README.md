# MedScan

A local medical-image classification research prototype with separate MRI, kidney CT, and chest X-ray models. **Research and education only; not a diagnostic tool.** Scores are uncalibrated dataset outputs.

Training and evaluation completed on an NVIDIA RTX 5060 Laptop GPU. See [RESULTS.md](RESULTS.md) for measured performance and historical comparisons, [MODEL_CARDS.md](MODEL_CARDS.md) for checkpoint provenance and limitations, and [reports/2026-11-06](reports/2026-11-06) for aggregate metrics and the captured environment.

The release includes source, tests and aggregate results. Obtain datasets separately. Images, patient IDs, prediction CSVs, cache contents and trained weights are excluded. The older nested source directory is an ignored reference copy; root source is authoritative.

## Tasks and scope

| Task | Validation-selected checkpoint | Scope |
|---|---|---|
| MRI | ConvNeXt-Tiny | Three tumor subtypes: glioma, meningioma, pituitary. No healthy class; 2D slices |
| CT | ConvNeXt-Tiny | Cyst, normal, stone, tumor on exported kidney JPEGs; exploratory proxy groups, no verified patient independence |
| X-ray | ResNet-18 | Fourteen independent NIH ChestX-ray14 findings, using sigmoid scores and validation-fitted thresholds |

No automatic modality router, 3D model, clinical calibration or external clinical validation is provided. The MRI candidate did not improve held-out macro-F1 over the historical ResNet-18 baseline. Do not describe all newer runs as better. The original CT baseline's perfect score was compromised by confirmed cross-split duplicate leakage.

## Setup

Python 3.11 was used. Create a virtual environment, install compatible PyTorch/torchvision for your hardware, then install the remaining dependencies. The measured CUDA environment used PyTorch 2.10.0 and torchvision 0.25.0 with CUDA 12.8; do not reinstall an already working GPU stack unnecessarily.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# GPU environment used for these runs:
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
# For CPU-only use, replace cu128 with cpu above.
python -m pip install -r requirements.txt
python check_gpu.py
python -m pytest -q
```

The recorded full local environment includes optional UI packages; CI uses CPU PyTorch and the main requirements. All 24 tests passed locally. CUDA availability, matrix multiplication, selected-checkpoint inference and held-out evaluation were checked. Local browser upload/prediction smoke checks passed; clinical validation was not performed.

## Prepare your own datasets

See [DATA_SOURCES.md](DATA_SOURCES.md) for sources, labels and licensing considerations. Keep source images immutable during a run. Dataset and checkpoint permissions are separate from the source-code MIT license.

MRI original MAT files retain patient IDs:

```bash
python convert_figshare_mri.py --input /path/to/original-mat-files --output data/processed/mri
python prepare_manifest.py --input data/processed/mri/all_unsplit.csv --output data/processed/mri/splits.csv
```

`prepare_nih.py` expects the NIH release in `data/archive (2)/`, including metadata, official split lists and `images_*/images/*.png`. It preserves the official test list and splits train/validation by patient.

For CT, the local exploratory manifest was generated from the original manifest and cached audit, using `prepare_ct_grouped.py`. Keep known related images together and remove exact duplicates. Do not use the old random image split for a new performance claim. The original audit cache is local and not redistributed; exact reproduction of the local grouped split requires that cache and the matching source manifest. Numeric filename blocks are not patient IDs.

## Train

Use fresh output directories. Existing training scripts do not resume optimizer state and can overwrite outputs when called directly. Selection uses validation only; seeds are 42 by default. Do not tune using the published test results.

```bash
python train.py --manifest data/processed/mri/splits.csv --mode mri_slice --arch convnext_tiny --size 224 --epochs 25 --batch-size 8 --workers 4 --augment --require-cuda --output runs/mri_convnext_tiny_20261106
python train.py --manifest data/processed/ct/splits_grouped.csv --image-level --mode ct_exported --arch convnext_tiny --size 224 --epochs 25 --batch-size 8 --workers 4 --augment --require-cuda --output runs/ct_grouped_convnext_tiny
python train_nih.py --manifest data/processed/nih/splits.csv --arch resnet18 --size 224 --epochs 23 --batch-size 32 --workers 12 --cache-dir data/cache/nih_224 --require-cuda --output runs/nih_resnet18_ap_cached_20261106
```

NIH caching is optional and lossless: a single grayscale float32 tensor after intensity scaling/resizing is stored per accessed image. Training augmentation remains fresh. First access builds the cache; later epochs reuse it. Training and validation together require about 16.5 GiB at size 224. Keys include source path, file size, nanosecond mtime, resolution and preprocessing version. Cache invalidation assumes original files remain immutable during a run; a manifest hash identifies manifest text, not image contents. Test inputs are not cached by the training command.

`compare_runs.py` compares runs only with matching task, exact manifest hash and selection metric. Grouped CT EfficientNet-B0 was also trained with batch size 16 and 25 maximum epochs, but ConvNeXt-Tiny won on validation. The historical NIH baseline used AUROC selection; the updated run uses weighted loss and macro AP selection, so their test comparison is descriptive rather than a controlled architecture experiment.

## Evaluate selected models

After selecting candidates using validation, evaluate once and retain the outputs:

```bash
python evaluate.py --checkpoint runs/mri_convnext_tiny_20261106/best.pt --manifest data/processed/mri/splits.csv --output runs/mri_convnext_tiny_20261106/test --batch-size 32 --workers 12
python evaluate.py --checkpoint runs/ct_grouped_convnext_tiny/best.pt --manifest data/processed/ct/splits_grouped.csv --output runs/ct_grouped_convnext_tiny/test --batch-size 32 --workers 12
python evaluate.py --checkpoint runs/nih_resnet18_ap_cached_20261106/best.pt --manifest data/processed/nih/splits.csv --output runs/nih_resnet18_ap_cached_20261106/test --batch-size 32 --workers 12
```

These commands generated the reported metrics. They also generate local prediction CSVs, which must remain excluded from publication. NIH evaluation reads original test images, without populating the training cache. MRI/NIH historical tests were examined previously; the CT repartition is not an independent external test. Test scores from this release must not guide further tuning.

## Local inference and demo

```bash
python predict.py --modality mri --checkpoint runs/mri_convnext_tiny_20261106/best.pt --image /path/to/original-mri.npy
python predict.py --modality ct --checkpoint runs/ct_grouped_convnext_tiny/best.pt --image /path/to/ct.jpg
python predict.py --modality xray --checkpoint runs/nih_resnet18_ap_cached_20261106/best.pt --image /path/to/xray.png
```

Use the original preprocessing and correct modality checkpoint. MRI rendered images can differ from the original NumPy input distribution. NIH findings use independent sigmoid scores, not a softmax diagnosis. Neither scores nor threshold flags are calibrated medical risks.

Optional localhost Gradio demo:

```bash
python -m pip install 'gradio>=5,<7'
python app.py --mri-checkpoint runs/mri_convnext_tiny_20261106/best.pt --ct-checkpoint runs/ct_grouped_convnext_tiny/best.pt --xray-checkpoint runs/nih_resnet18_ap_cached_20261106/best.pt
```

Visit `http://127.0.0.1:7860`. Do not upload identifiable clinical data or enable public sharing. Local browser checks covered synthetic uploads, empty-upload handling and all three modality paths; see RESULTS.md. Synthetic inputs demonstrate plumbing only, not medical performance.

A [desktop screenshot](docs/demo-synthetic.png) shows the demo using synthetic random noise. Its predicted class and scores have no medical meaning. Desktop (1280×900) and mobile (390×844) views were inspected; all three upload/prediction paths and the empty-input error were checked locally.

## License and release contents

Source code is licensed under [MIT](LICENSE). This does not relicense third-party datasets, pretrained weights or optional dependencies. Weights are not included, and redistribution permissions have not been established by this release. Aggregate results contain no image paths, patient identifiers or per-image predictions. Local data, caches, checkpoints, logs, the old nested source copy and environment directories remain ignored.
