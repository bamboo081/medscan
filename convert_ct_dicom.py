"""Optional CT DICOM -> HU .npy converter; clinical context determines HU window.

Do not include PHI-bearing original DICOM, private data, or derived scans in Git.
"""
import argparse
from pathlib import Path
import numpy as np


def convert(source, destination):
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install pydicom") from exc
    ds = pydicom.dcmread(str(source))
    if str(getattr(ds, "Modality", "")).upper() != "CT":
        raise ValueError("Not a CT DICOM. Only CT gets HU normalization.")
    pixels = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError("Only 2D single-frame DICOM supported in this starter")
    if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
        raise ValueError("Cannot establish Hounsfield units without slope/intercept")
    hu = pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, hu.astype(np.float32), allow_pickle=False)
    print("Saved HU-only array. Never publish original clinical data without authorization.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    convert(a.input, a.output)
