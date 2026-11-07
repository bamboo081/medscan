# CT duplicate / slice-similarity / artifact audit

This is a **read-only**, local research audit for your image-level kidney CT baseline. It **does not** retrain, inspect the model checkpoint, or modify any dataset images/splits.

## Ubuntu setup

Copy `audit_ct.py` to the root of your existing `medscan-training-starter` project and run from that folder:

```bash
source .venv/bin/activate
python -m pip install 'numpy>=2' 'pandas>=2' 'Pillow>=11' 'scipy>=1.13' 'scikit-learn>=1.5'
python audit_ct.py \
  --manifest data/processed/ct/splits.csv \
  --output runs/ct_resnet18/audit
```

Optional smaller pass (`--max-near-pairs` caps the number of perceptual candidates compared using structural similarity; **all** manifest-referenced images are still decoded once):

```bash
python audit_ct.py --manifest data/processed/ct/splits.csv \
  --output runs/ct_resnet18/audit --max-near-pairs 100 --preview-pairs 12
```

For relative paths that aren't resolved from the manifest folder or current project directory, add `--data-root /absolute/path/to/dataset`.

## Files written

- `REPORT.md`: concise interpretation
- `summary.json`: counts, duplicate findings, probe score
- `exact_duplicate_groups.json`: identical encoded files and decoded pixels
- `exact_cross_split_pairs.csv`: exact pixel duplicate examples across splits
- `near_duplicate_candidates.csv`: highest-priority perceptual neighbors across splits with pHash/SSIM
- `exact_cross_split_examples.jpg` and `near_duplicate_examples.jpg`: contact sheets for human inspection
- `image_features.csv`: image dimensions, compression size, brightness and edge features
- `artifact_feature_by_class.csv`: training-set artifact-feature distributions per class
- `artifact_feature_importance.csv`: features used by a fast training-only shortcut-cue classifier

## What to watch for

1. Any **exact duplicate** across training, validation and test is directly problematic.
2. Visually similar different slices may come from one patient, but similarity alone is **not** patient identity proof.
3. Large class differences in image sizes, compressed byte lengths, borders or corner markers can signal shortcuts; not all such differences are artificial.
4. A high artifact-feature-only balanced accuracy is a warning sign, **not** rigorous evidence of leakage, because the training subset itself may contain related images.
5. No patient IDs means the original 1.0 macro-F1 **cannot be described as patient-independent accuracy**, regardless of audit results.

**Privacy:** Any source images and montages stay on your machine; avoid publishing unreviewed patient scans or possible burned-in patient information.
