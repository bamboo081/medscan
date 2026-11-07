import numpy as np
import pytest
import torch
from PIL import Image
from medscan.model import MODELS, build_model
from predict import predict


@pytest.mark.parametrize('arch', MODELS)
def test_architecture_output(arch):
    model = build_model(arch, ['a', 'b', 'c'], pretrained=False)
    # Check head shape without expensive full forward for larger models.
    if arch.startswith('resnet'):
        out_features = model.fc.out_features
    else:
        out_features = model.classifier[-1].out_features
    assert out_features == 3
    del model


def test_old_checkpoint_predicts_without_hu_key(tmp_path):
    net = build_model('resnet18', ['a', 'b'], pretrained=False)
    ckpt = tmp_path / 'old.pt'
    torch.save({'model_state': net.state_dict(), 'config': {
        'model': 'resnet18', 'classes': ['a', 'b'], 'mode': 'ct_exported',
        'size': 64, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    }}, ckpt)
    img = tmp_path / 'scan.png'
    Image.fromarray(np.arange(64*64, dtype=np.uint8).reshape(64,64)).save(img)
    out = predict(str(ckpt), str(img), modality='ct')
    assert out['task'] == 'multiclass'
    assert set(out['uncalibrated_class_scores']) == {'a', 'b'}
    with pytest.raises(ValueError, match='cannot use'):
        predict(str(ckpt), str(img), modality='mri')


def test_legacy_checkpoint_evaluation_without_hu_key(tmp_path):
    import pandas as pd
    import json
    from medscan.data import manifest_sha256
    from evaluate import evaluate

    rows = []
    for split in ('train', 'val', 'test'):
        for label in ('a', 'b'):
            path = tmp_path / f'{split}_{label}.png'
            Image.fromarray(np.arange(64*64, dtype=np.uint8).reshape(64, 64)).save(path)
            rows.append({'path': str(path), 'label': label, 'split': split,
                         'patient_id': f'{split}_{label}'})
    manifest = tmp_path / 'splits.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    ckpt = tmp_path / 'legacy.pt'
    net = build_model('resnet18', ['a', 'b'], pretrained=False)
    torch.save({'model_state': net.state_dict(), 'config': {
        'model': 'resnet18', 'classes': ['a', 'b'], 'mode': 'mri_slice',
        'size': 64, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
        'manifest_sha256': manifest_sha256(manifest),
    }}, ckpt)
    output = tmp_path / 'test'
    evaluate(ckpt, manifest, output, batch_size=2, workers=0)
    assert json.loads((output / 'test_metrics.json').read_text())['samples'] == 2
