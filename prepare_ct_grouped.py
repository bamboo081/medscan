#!/usr/bin/env python3
"""Create a *new exploratory* CT manifest using the existing audit cache.

No re-reading 12,446 image files: reads audit/image_features.csv (pixel hashes)
plus the reviewed near-pair CSV, de-duplicates identical pixels, and groups
small, contiguous filename-number blocks as a *scan-series proxy*.

NOT patient-disjoint. Cannot guarantee absence of visually related slices.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = {"normal", "cyst", "stone", "tumor"}


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def filename_block(row, block_size):
    """Only a grouping HEURISTIC; numeric filenames do not prove patient identity."""
    stem = Path(row.path).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    # Keep same class and parent folder together, with small sequential blocks.
    return (row.label, str(Path(row.path).parent), (int(nums[-1]) - 1) // block_size)


def create_grouped(source, features, near, *, block_size=32, ssim=0.90, seed=42):
    if block_size < 1 or not 0 <= ssim <= 1:
        raise ValueError("Invalid block size or SSIM")
    base = pd.read_csv(source, dtype=str, keep_default_na=False)
    if not {"path", "label", "split"}.issubset(base.columns):
        raise ValueError("Original manifest needs path,label,split")
    if base.empty or base.path.duplicated().any() or base[["path", "label"]].eq("").any().any():
        raise ValueError("Bad manifest")
    if set(base.label) != LABELS:
        raise ValueError("Expected exactly four CT classes")
    audit = pd.read_csv(features, dtype={"path": str, "label": str, "pixel_sha256": str})
    if not {"path", "label", "pixel_sha256"}.issubset(audit.columns):
        raise ValueError("Audit image_features.csv needs path,label,pixel_sha256")
    if audit.path.duplicated().any() or set(audit.path) != set(base.path):
        raise ValueError("Audit and source manifest have different file lists; rerun audit first")
    df = base.merge(audit[["path", "label", "pixel_sha256"]], on="path", how="inner", validate="one_to_one", suffixes=("", "_audit"))
    if (df.label != df.label_audit).any():
        raise ValueError("Audit class labels disagree with manifest")
    df = df.drop(columns="label_audit").sort_values("path").reset_index(drop=True)
    if df.pixel_sha256.isna().any() or df.pixel_sha256.astype(str).eq("").any():
        raise ValueError("Missing pixel hashes")
    if (df.groupby("pixel_sha256").label.nunique() > 1).any():
        raise ValueError("Exact image duplicates have conflicting class labels; investigate manually")
    ds = DisjointSet(len(df))
    index = dict(zip(df.path, df.index))
    exact_seen = {}
    for row in df.itertuples():
        if row.pixel_sha256 in exact_seen:
            ds.join(row.Index, exact_seen[row.pixel_sha256])
        else:
            exact_seen[row.pixel_sha256] = row.Index
    filename_seen = {}
    for row in df.itertuples():
        key = filename_block(row, block_size)
        if key is not None:
            if key in filename_seen:
                ds.join(row.Index, filename_seen[key])
            else:
                filename_seen[key] = row.Index
    linked_near = 0
    skipped_cross_class = []
    if near is not None:
        pairs = pd.read_csv(near)
        if not {"path_a", "path_b", "ssim_128"}.issubset(pairs.columns):
            raise ValueError("Near candidate CSV columns missing")
        for row in pairs.itertuples():
            if not np.isfinite(row.ssim_128) or row.ssim_128 < ssim:
                continue
            if row.path_a not in index or row.path_b not in index:
                raise ValueError("Near candidate CSV references images absent from source")
            a, b = index[row.path_a], index[row.path_b]
            if df.at[a, "label"] != df.at[b, "label"]:
                skipped_cross_class.append([row.path_a, row.path_b, float(row.ssim_128)])
                continue
            ds.join(a, b)
            linked_near += 1
    df["component"] = [ds.find(i) for i in range(len(df))]
    # Conflicting class labels inside connected components are a metadata red flag.
    if (df.groupby("component").label.nunique() > 1).any():
        raise ValueError("Related-image component crosses disease labels; inspect numeric filename grouping")
    # Select one canonical image per *exact pixel hash*; group membership survives.
    unique = df.drop_duplicates(subset="pixel_sha256", keep="first").copy().reset_index(drop=True)
    counts = unique.groupby("component").size().sort_values(ascending=False)
    if counts.iloc[0] > len(unique) * .15:
        raise ValueError("A single similarity group exceeds 15% of data: refine grouping manually")
    if unique.groupby("label").component.nunique().min() < 10:
        raise ValueError("Too few independent *proxy groups* to make a meaningful 10-fold split")
    # 10-fold grouped stratification => ~80/10/10, respecting all proxy groups.
    result = None
    for trial in range(seed, seed + 20):
        skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=trial)
        folds = list(skf.split(unique.path, unique.label, unique.component))
        labels = np.full(len(unique), "train", dtype=object)
        labels[folds[0][1]] = "test"
        labels[folds[1][1]] = "val"
        unique["split"] = labels
        if all(set(unique.loc[unique.split == s, "label"]) == LABELS for s in ("train", "val", "test")):
            result = unique
            break
    if result is None:
        raise ValueError("Could not create a class-complete grouped split; inspect group sizes")
    result["group_id"] = result.component.map(lambda x: f"proxy_{int(x):06d}")
    if (result.groupby("group_id").split.nunique() != 1).any():
        raise AssertionError("Proxy group crosses splits")
    if (result.groupby("pixel_sha256").split.nunique() != 1).any():
        raise AssertionError("Exact pixel leakage detected")
    # Audited pairs in the prior image-level split are ALL now kept in one proxy group.
    # This checks only the capped *known* pairs, not all possible near duplicates.
    remaining = []
    for row in skipped_cross_class:
        remaining.append({"path_a": row[0], "path_b": row[1], "ssim": row[2]})
    keep_cols = ["path", "label", "split", "group_id"]
    manifest = result[keep_cols].sort_values(["split", "label", "path"]).reset_index(drop=True)
    summary = {
        "source_images": len(df), "kept_unique_pixel_images": len(result),
        "removed_exact_pixel_copies": len(df) - len(result),
        "proxy_groups": int(result.group_id.nunique()), "largest_group_images": int(counts.iloc[0]),
        "linked_reviewed_near_pairs": linked_near,
        "reviewed_cross_class_near_pairs_skipped": remaining[:30],
        "image_counts": {k: int(v) for k, v in manifest.split.value_counts().items()},
        "class_counts": {s: {str(c): int(n) for c, n in g.label.value_counts().items()}
                         for s, g in manifest.groupby("split")},
        "grouping": f"pixel SHA256 + {block_size}-number filename blocks + cached audited SSIM >= {ssim}",
        "limitations": "NOT patient disjoint; filename blocks are not patient IDs; cached near-neighbor search was capped and incomplete. Prior test accessed. No new independent test cohort. Exploratory model only.",
    }
    return manifest, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/processed/ct/splits.csv")
    p.add_argument("--audit-dir", default="runs/ct_resnet18/audit")
    p.add_argument("--output", default="data/processed/ct/splits_grouped.csv")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--near-ssim", type=float, default=.90)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    source = Path(args.manifest)
    out = Path(args.output)
    if out.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite the original manifest")
    audit = Path(args.audit_dir)
    if not (audit / "image_features.csv").is_file():
        raise FileNotFoundError("Missing audit cache; use the output from audit_ct.py")
    near = audit / "near_duplicate_candidates.csv"
    manifest, summary = create_grouped(source, audit / "image_features.csv", near if near.exists() else None,
                                       block_size=args.block_size, ssim=args.near_ssim, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing {out}; choose another --output")
    manifest.to_csv(out, index=False)
    summary["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out.with_suffix(".report.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}; heuristic scan proxy ONLY, NOT independent patients")


if __name__ == "__main__":
    main()
