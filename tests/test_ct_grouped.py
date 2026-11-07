from pathlib import Path

import pandas as pd
import pytest

from prepare_ct_grouped import create_grouped
from medscan.data import read_manifest


def test_grouped_ct_deduplicates_and_separates_blocks(tmp_path):
    base = []
    features = []
    for c in ('normal', 'cyst', 'stone', 'tumor'):
        for block in range(16):
            for j in range(2):
                num = block * 32 + j + 1
                path = tmp_path / c / f'{c}- ({num}).jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'research test fixture')
                base.append({'path': str(path), 'label': c, 'split': 'train'})
                features.append({'path': str(path), 'label': c,
                                 'pixel_sha256': f'{c}{block}{j}'})
    # second file is exactly same as an earlier image (despite different filename)
    features[5]['pixel_sha256'] = features[0]['pixel_sha256']
    src = tmp_path / 'old.csv'; feats = tmp_path / 'features.csv'
    pd.DataFrame(base).to_csv(src, index=False)
    pd.DataFrame(features).to_csv(feats, index=False)
    output, report = create_grouped(src, feats, None)
    assert report['removed_exact_pixel_copies'] == 1
    assert len(output) == len(base) - 1
    assert not output.path.duplicated().any()
    assert set(output.split) == {'train', 'val', 'test'}
    # each 32-image index block is a single proxy group; exact hash grouping never crosses
    assert (output.groupby('group_id').split.nunique() == 1).all()
    output.to_csv(tmp_path / 'new.csv', index=False)
    assert len(read_manifest(tmp_path / 'new.csv', image_level=True)) == len(output)


def test_group_id_leakage_is_rejected(tmp_path):
    from PIL import Image
    import numpy as np
    rows = []
    for split in ('train', 'val', 'test'):
        for label in ('a', 'b'):
            path = tmp_path / f'{split}_{label}.png'
            Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(path)
            rows.append({'path': str(path), 'label': label, 'split': split,
                         'group_id': 'same' if label == 'a' else f'{split}_b'})
    csv = tmp_path / 'bad.csv'; pd.DataFrame(rows).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='Leakage'):
        read_manifest(csv, image_level=True)
