import pandas as pd
import pytest
from PIL import Image
import numpy as np
from medscan.data import read_manifest


def make_csv(tmp_path, leakage=False):
    rows=[]
    for split in ("train", "val", "test"):
        for label in ("a", "b"):
            path = tmp_path / f"{split}_{label}.png"
            Image.fromarray(np.full((16,16), 128,dtype=np.uint8)).save(path)
            rows.append({"path":path.name, "patient_id": "shared" if leakage and label == 'a' else f'{split}_{label}',
                         "label":label, "split":split})
    csv=tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    return csv


def test_manifest_valid(tmp_path):
    df = read_manifest(make_csv(tmp_path))
    assert len(df) == 6


def test_patient_leakage_rejected(tmp_path):
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(make_csv(tmp_path, leakage=True))
