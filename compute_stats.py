"""Optional training-only post-intensity-normalization mean/std estimation.

NEVER fit statistics to val or test; not needed for default ImageNet normalization.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from medscan.data import read_manifest, manifest_sha256
from medscan.preprocessing import load_grayscale, intensity_scale, MODES
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


def compute(manifest, mode, size, output, hu_min=None, hu_max=None, image_level=False):
    if mode == "ct_hu" and (hu_min is None or hu_max is None or hu_min >= hu_max):
        raise ValueError("Need valid HU window")
    hu_window = (hu_min, hu_max) if mode == "ct_hu" else None
    df = read_manifest(manifest, image_level=image_level)
    df = df[df.split == "train"]
    n, total, squared = 0, 0.0, 0.0
    for row in df.itertuples():
        if mode == "ct_hu" and Path(row.resolved_path).suffix.lower() != ".npy":
            raise ValueError("HU values require raw .npy inputs")
        arr = intensity_scale(load_grayscale(row.resolved_path), mode, hu_window)
        x = resize(torch.from_numpy(arr)[None], [size,size], interpolation=InterpolationMode.BILINEAR, antialias=True).numpy()
        total += float(x.sum(dtype=np.float64))
        squared += float(np.square(x, dtype=np.float64).sum())
        n += x.size
    mu = total / n
    sd = max((squared / n - mu*mu), 1e-12) ** .5
    stats = {"mean":[mu]*3, "std":[sd]*3, "mode":mode, "size":size,
             "manifest_sha256":manifest_sha256(manifest), "note":"training images only; use unchanged in val/test/inference"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--output", required=True)
    p.add_argument("--image-level", action="store_true", help="Exploratory CT manifest without patient identifiers")
    p.add_argument("--hu-min", type=float)
    p.add_argument("--hu-max", type=float)
    args = p.parse_args()
    compute(**vars(args))
