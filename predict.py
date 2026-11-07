"""Local, explicitly selected modality inference from a trusted checkpoint.

No automatic modality diagnosis, no calibrated risk estimates, no medical use.
"""
import argparse
import json
from pathlib import Path

import torch

from medscan.model import build_model
from medscan.preprocessing import image_tensor

MODALITY_MODES = {
    'mri': {'mri_slice'},
    'ct': {'ct_exported', 'ct_hu'},
    'xray': {'xray_exported'},
}


def load_predictor(checkpoint, modality=None, device=None):
    checkpoint = Path(checkpoint)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = cp['config']
    if modality is not None and cfg['mode'] not in MODALITY_MODES[modality]:
        raise ValueError(f'{modality} cannot use a checkpoint trained with {cfg["mode"]}')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model'], cfg['classes'], pretrained=False).to(device)
    model.load_state_dict(cp['model_state'], strict=True)
    model.eval()
    return model, cfg, cp, device


def classify(model, cfg, cp, device, image_path):
    x = image_tensor(image_path, mode=cfg['mode'], size=cfg['size'], hu_window=cfg.get('hu_window'),
                     mean=cfg.get('mean'), std=cfg.get('std')).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x).float()[0]
        if cfg.get('task') == 'multilabel':
            vals = torch.sigmoid(logits).cpu().tolist()
            result = {'task': 'multilabel', 'model': cfg['model'],
                      'uncalibrated_class_scores': dict(zip(cfg['classes'], vals))}
            if 'thresholds' in cp:
                result['above_validation_threshold'] = [n for n, v, t in zip(cfg['classes'], vals, cp['thresholds']) if v >= t]
                result['warning_threshold'] = 'Thresholds are only research decision rules, not diagnoses.'
        else:
            vals = torch.softmax(logits, dim=0).cpu().tolist()
            result = {'task': 'multiclass', 'model': cfg['model'],
                      'predicted_dataset_class': cfg['classes'][int(torch.argmax(logits))],
                      'uncalibrated_class_scores': dict(zip(cfg['classes'], vals))}
    result['dataset_scope'] = cfg.get('mode')
    result['not_for_clinical_use'] = True
    result['limitations'] = 'Research demo on narrow source distributions; not diagnoses, calibrated risks or a substitute for clinical evaluation.'
    if cfg['mode'] == 'ct_exported':
        result['additional_warning'] = 'CT JPG 4-class dataset lacks patient metadata; previous image-level test had confirmed leakage.'
    if cfg['mode'] == 'mri_slice':
        result['additional_warning'] = 'MRI checkpoint only distinguishes trained tumor types; cannot rule out tumor or recognize healthy scans.'
    return result


def predict(checkpoint, image_path, modality=None):
    model, cfg, cp, device = load_predictor(checkpoint, modality=modality)
    return classify(model, cfg, cp, device, image_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--modality', choices=list(MODALITY_MODES), help='Explicit user-selected modality; no auto-detection')
    a = p.parse_args()
    print(json.dumps(predict(a.checkpoint, a.image, a.modality), indent=2))
