import numpy as np
from medscan.multilabel import choose_thresholds, multilabel_metrics


def test_ap_auroc_and_validation_thresholds():
    y = np.array([[1, 0], [1, 0], [1, 1], [0, 0], [0, 1], [0, 1]])
    pred = np.array([[.9, .1], [.8, .2], [.7, .7], [.2, .1], [.1, .8], [.1, .9]])
    # 2nd class has >=3 positives but <5: threshold stays default
    thresholds = choose_thresholds(y, pred, ['a', 'b'])
    assert thresholds == [0.5, 0.5]
    m = multilabel_metrics(y, pred, ['a', 'b'], thresholds)
    assert m['macro_average_precision'] > 0.8
    assert m['macro_auroc'] > 0.8
    assert m['per_class']['b']['positives'] == 3
    assert m['per_class']['a']['f1'] == 1.0


def test_empty_class_is_reported_unavailable():
    y = np.array([[0, 1], [0, 0], [0, 1]])
    scores = np.array([[.1, .9], [.2, .3], [.9, .8]])
    m = multilabel_metrics(y, scores, ['no_positive', 'finding'])
    assert m['per_class']['no_positive']['auroc'] is None
    assert m['valid_auc_classes'] == 1


def test_nih_multilabel_no_finding_and_mixed_labels(tmp_path):
    import pandas as pd
    from train_nih import read_nih_manifest, NIHData, CLASSES
    rows = []
    for ix, split in enumerate(('train', 'val', 'test')):
        rows.append({'path': str(tmp_path / f'{split}.png'), 'patient_id': str(ix),
                     'labels': 'No Finding' if ix == 0 else 'Pneumonia|Effusion', 'split': split})
    manifest = tmp_path / 'nih.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    df = read_nih_manifest(manifest)
    data = NIHData(df, size=64)
    assert data.targets.shape == (3, 14)
    assert data.targets[0].sum() == 0
    assert data.targets[1, CLASSES.index('Pneumonia')] == 1
    assert data.targets[1, CLASSES.index('Effusion')] == 1
