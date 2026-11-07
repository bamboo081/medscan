"""Make an immutable patient-disjoint split from an annotated unsplit CSV.

Input: path,patient_id,label with paths relative to the input CSV.
Do not invent patient IDs from image filenames or let one patient cross splits.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from medscan.data import SPLITS, read_manifest


def split_manifest(input_path, output_path, *, seed=42, folds=10):
    source = Path(input_path).resolve()
    dest = Path(output_path).resolve()
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    if not {"path", "patient_id", "label"}.issubset(df.columns):
        raise ValueError("Input must contain path,patient_id,label")
    if "split" in df.columns:
        raise ValueError("Source includes a split column. Preserve official source partitions; do not re-split!")
    if df.empty or df[["path", "patient_id", "label"]].eq("").any().any():
        raise ValueError("Do not use missing/invented patient IDs or labels")
    if df.path.duplicated().any():
        raise ValueError("Duplicate image paths")
    if (df.groupby("patient_id").label.nunique() > 1).any():
        raise ValueError("Conflicting labels for a patient (cannot use single-label baseline)")
    if df.patient_id.nunique() < folds * 2:
        raise ValueError("Too few patients for requested folds")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = list(splitter.split(df.path, df.label, df.patient_id))
    # Two distinct group-disjoint folds for val/test, rest for training.
    df["split"] = "train"
    df.loc[assignments[0][1], "split"] = "test"
    df.loc[assignments[1][1], "split"] = "val"
    for s in SPLITS:
        if set(df.loc[df.split == s, "label"]) != set(df.label):
            raise ValueError(f"Class absent from {s}: not enough class-distinct patients for these folds")
    # Keep paths valid relative to destination, even if it is elsewhere.
    df["path"] = df.path.map(lambda p: str((source.parent / p).resolve()))
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    read_manifest(dest)
    print(df.groupby(["split", "label"]).agg(images=("path", "size"), patients=("patient_id", "nunique")))
    print(f"Saved {dest} (patient-disjoint, seed={seed}, folds={folds})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=10)
    args = p.parse_args()
    split_manifest(args.input, args.output, seed=args.seed, folds=args.folds)
