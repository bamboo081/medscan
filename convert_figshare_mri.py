"""Convert original Jun Cheng Figshare v7.3 MAT files to lossless NumPy arrays.

Read original publisher data and the documented pid/label fields. Never label
an MRI scan as 'normal' using this dataset: all 3 classes are tumors.
"""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np

# Verified original dataset encoding (not the arbitrary IDs of mirrored datasets).
LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def convert_one(mat_file: Path):
    with h5py.File(mat_file, "r") as h:
        if "cjdata" not in h:
            raise ValueError(f"No cjdata in {mat_file}")
        obj = h["cjdata"]
        image = np.asarray(obj["image"], dtype=np.float32)
        label = int(np.asarray(obj["label"]).reshape(-1)[0])
        pid_chars = np.asarray(obj["PID"]).reshape(-1)
        patient_id = "".join(chr(int(c)) for c in pid_chars).strip("\x00 ")
    if image.ndim != 2 or not patient_id or label not in LABELS:
        raise ValueError(f"Invalid image/pid/label in {mat_file.name}")
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite pixels in {mat_file.name}")
    return image, patient_id, LABELS[label]


def convert(input_dir: Path, output_dir: Path):
    files = sorted(input_dir.rglob("*.mat"))
    if not files:
        raise FileNotFoundError("No .mat images found; obtain original Figshare source")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in files:
        image, patient_id, label = convert_one(src)
        dest = output_dir / f"{src.stem}.npy"
        if dest.is_file():
            raise ValueError(f"Output name collision: {dest}")
        np.save(dest, image, allow_pickle=False)
        manifest.append({"path":str(dest.resolve()), "patient_id":patient_id, "label":label})
    csv_path = output_dir / "all_unsplit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "patient_id", "label"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Converted {len(files)} MRI slices to .npy and wrote {csv_path}")
    print("Check patient ID grouping and class counts before creating splits.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/processed/mri"))
    args = p.parse_args()
    convert(args.input, args.output)
