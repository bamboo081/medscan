"""Audit per-class/split counts and exact content duplicates BEFORE training.

Cross-patient near-duplicate analysis and cross-hospital validation still require
additional human review; exact hashing does not catch lightly transformed copies.
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from medscan.data import read_manifest
from medscan.preprocessing import load_grayscale


def audit(manifest):
    df = read_manifest(manifest)
    print("Counts (images/patients):")
    print(df.groupby(["split", "label"]).agg(images=("path","count"), patients=("patient_id","nunique")))
    by_hash = defaultdict(list)
    shapes = defaultdict(int)
    for row in df.itertuples():
        x = load_grayscale(row.resolved_path)
        shapes[str(tuple(x.shape))] += 1
        digest = hashlib.sha256(x.tobytes() + str(x.shape).encode()).hexdigest()
        by_hash[digest].append((row.split, row.patient_id, row.label, row.path))
    duplicate_groups = [v for v in by_hash.values() if len(v) > 1]
    cross_split = [v for v in duplicate_groups if len(set(item[0] for item in v)) > 1]
    cross_patient = [v for v in duplicate_groups if len(set(item[1] for item in v)) > 1]
    print(f"Image dimensions: {dict(shapes)}")
    print(f"Exact pixel-duplicate groups: {len(duplicate_groups)}")
    print(f"Exact duplicates crossing patient IDs: {len(cross_patient)}")
    print(f"Exact duplicates crossing splits: {len(cross_split)}")
    if cross_split or cross_patient:
        print("FAIL: inspect / fix exact duplicates; no model training until leakage resolved.")
        for group in (cross_split or cross_patient)[:5]:
            print("  ", group)
        raise SystemExit(2)
    print("PASS exact-duplicate screen. Still manually audit near-duplicates, annotations, PHI, bias.")


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    a=p.parse_args()
    audit(a.manifest)
