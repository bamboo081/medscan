import hashlib
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from .preprocessing import image_tensor

REQUIRED = {"path", "patient_id", "label", "split"}
SPLITS = {"train", "val", "test"}


def read_manifest(path: str | Path, *, image_level=False) -> pd.DataFrame:
    path = Path(path).resolve()
    df = pd.read_csv(path, dtype={"path": str, "patient_id": str, "label": str, "split": str}, keep_default_na=False)
    required = REQUIRED - {"patient_id"} if image_level else REQUIRED
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if df.empty or df[list(required)].eq("").any().any():
        raise ValueError("Manifest may not contain empty rows or required values")
    if not set(df.split).issubset(SPLITS):
        raise ValueError("split must be train/val/test")
    if set(df.split) != SPLITS:
        raise ValueError("Manifest must contain train, val, and test splits")
    if df.path.duplicated().any():
        raise ValueError("Duplicate file paths in manifest")
    if "group_id" in df.columns:
        if df.group_id.astype(str).eq("").any():
            raise ValueError("group_id cannot be blank")
        if (df.groupby("group_id").split.nunique() > 1).any():
            raise ValueError("Leakage: known related-image group crosses train/val/test")
    if not image_level:
        # Require explicit patient IDs and forbid a patient appearing across splits.
        per_pid = df.groupby("patient_id").split.nunique()
        if (per_pid > 1).any():
            raise ValueError("Patient leakage: one or more patient IDs span multiple splits")
        # A patient is not allowed to acquire contradictory single-label diagnosis in this baseline.
        # Adjust deliberately for future multi-label tasks instead of silently mixing labels.
        conflicting = df.groupby("patient_id").label.nunique()
        if (conflicting > 1).any():
            raise ValueError("A patient has multiple class labels; inspect labels before training")
    base = path.parent
    df["resolved_path"] = df.path.map(lambda p: str((base / p).resolve()) if not Path(p).is_absolute() else str(Path(p).resolve()))
    missing_files = [p for p in df.resolved_path if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} images, first: {missing_files[0]}")
    labels = set(df.label)
    for split in SPLITS:
        if set(df[df.split == split].label) != labels:
            raise ValueError(f"Split '{split}' does not contain every class; re-split by patient")
    return df


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MedicalDataset(Dataset):
    def __init__(self, df, labels, *, mode, size, augment=False, hu_window=None,
                 mean=None, std=None):
        self.records = list(df[["resolved_path", "label"]].itertuples(index=False, name=None))
        self.class_to_idx = {label: i for i, label in enumerate(labels)}
        self.kwargs = dict(mode=mode, size=size, augment=augment, hu_window=hu_window, mean=mean, std=std)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        return image_tensor(path, **self.kwargs), self.class_to_idx[label]
