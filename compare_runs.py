"""Rank compatible checkpoints by validation results only. Never read test metrics."""
import argparse
import json
from pathlib import Path


def compare(paths):
    rows = []
    for path in paths:
        run = Path(path)
        cfg = json.loads((run / 'config.json').read_text())
        hist = json.loads((run / 'history.json').read_text())
        task = cfg.get('task', 'multiclass')
        key = 'val_macro_ap' if task == 'multilabel' else ('val_macro_f1' if cfg.get('image_level') else 'val_patient_macro_f1')
        if task != 'multilabel' and not cfg.get('image_level') and any(e.get(key) is None for e in hist):
            raise ValueError(f'Missing patient-level metric for {run}')
        score = max(float(e[key]) for e in hist if e.get(key) is not None)
        rows.append({'run': str(run), 'model': cfg['model'], 'task': task,
                     'manifest_sha256': cfg['manifest_sha256'], 'metric': key,
                     'best_validation_score': score, 'checkpoint': str(run / 'best.pt')})
    signatures = {(r['task'], r['manifest_sha256'], r['metric']) for r in rows}
    if len(signatures) != 1:
        raise ValueError('Runs must use the SAME task, exact manifest and validation metric')
    rows.sort(key=lambda x: x['best_validation_score'], reverse=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    args = p.parse_args()
    print(json.dumps(compare(args.runs), indent=2))
    print('These are validation comparisons; no test-set score was used. Verify resource costs and class-level results.')
