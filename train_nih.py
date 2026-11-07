"""Patient-disjoint NIH ChestX-ray14 multi-label training.

Official test partition remains sealed until evaluate.py. Model selection uses
macro average precision (more informative than accuracy on rare labels).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from medscan.model import MODELS, build_model
from medscan.multilabel import choose_thresholds, multilabel_metrics
from medscan.preprocessing import image_tensor, finish_image, IMAGENET_MEAN, IMAGENET_STD
from medscan.image_cache import cached_image
from medscan.data import manifest_sha256
from train import seed_everything

CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def read_nih_manifest(path):
    df = pd.read_csv(path, dtype={'patient_id': str, 'path': str, 'labels': str, 'split': str}, keep_default_na=False)
    if not {'path', 'patient_id', 'labels', 'split'}.issubset(df):
        raise ValueError('NIH manifest needs path,patient_id,labels,split')
    if df[['path', 'patient_id', 'labels', 'split']].eq('').any().any() or df.path.duplicated().any():
        raise ValueError('Invalid NIH manifest')
    if set(df.split) != {'train', 'val', 'test'} or (df.groupby('patient_id').split.nunique() > 1).any():
        raise ValueError('Invalid splits or patient overlap')
    allowed = set(CLASSES) | {'No Finding'}
    for label_str in df.labels.unique():
        labels = set(label_str.split('|'))
        if not labels <= allowed or ('No Finding' in labels and len(labels) > 1):
            raise ValueError(f'Unknown or contradictory NIH label: {label_str}')
    return df


class NIHData(Dataset):
    def __init__(self, df, size, augment=False, mean=None, std=None, cache_dir=None):
        self.paths = df.path.tolist()
        self.targets = np.array([[float(c in s.split('|')) for c in CLASSES] for s in df.labels], dtype=np.float32)
        self.size, self.augment = size, augment
        self.cache_dir = cache_dir
        self.mean = mean if mean is not None else IMAGENET_MEAN
        self.std = std if std is not None else IMAGENET_STD

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        if self.cache_dir is not None:
            image = cached_image(self.paths[i], size=self.size, cache_dir=self.cache_dir)
            return finish_image(torch.from_numpy(image), augment=self.augment,
                                mean=self.mean, std=self.std), torch.from_numpy(self.targets[i])
        return image_tensor(self.paths[i], mode='xray_exported', size=self.size,
                            augment=self.augment, mean=self.mean, std=self.std), torch.from_numpy(self.targets[i])


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x.to(device, non_blocking=True))
        predictions.append(logits.float().sigmoid().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/processed/nih/splits.csv')
    p.add_argument('--output', default='runs/nih_baseline')
    p.add_argument('--arch', choices=MODELS, default='resnet18')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pos-weight-cap', type=float, default=12.0)
    p.add_argument('--no-pos-weights', action='store_true')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--cache-dir', help='Optional lossless, on-demand preprocessing cache (train/val only)')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.size, args.patience) < 1 or args.workers < 0 or args.pos_weight_cap < 1:
        raise ValueError('Invalid training parameters')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.require_cuda and device.type != 'cuda':
        raise RuntimeError('CUDA unavailable')
    df = read_nih_manifest(args.manifest)
    if args.cache_dir:
        cache = Path(args.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        estimate = int((df.split != 'test').sum()) * (args.size * args.size * 4 + 4096)
        if shutil.disk_usage(cache).free < estimate + 5 * 1024**3:
            raise ValueError('Insufficient free disk for full cache plus a 5 GiB reserve')
        print(f'Lazy train/val cache: {cache} (up to {estimate / 1024**3:.1f} GiB; first epoch builds it)', flush=True)
    train_ds = NIHData(df[df.split == 'train'], args.size, augment=True, cache_dir=args.cache_dir)
    val_ds = NIHData(df[df.split == 'val'], args.size, cache_dir=args.cache_dir)
    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == 'cuda',
                persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **opts)
    val_loader = DataLoader(val_ds, shuffle=False, **opts)
    model = build_model(args.arch, CLASSES, pretrained=not args.no_pretrained).to(device)
    positives = train_ds.targets.sum(axis=0)
    n = len(train_ds)
    weights = np.clip((n - positives) / np.maximum(positives, 1), 1, args.pos_weight_cap)
    if not args.no_pos_weights and (positives == 0).any():
        raise ValueError('A label has no positive training examples')
    weight_tensor = None if args.no_pos_weights else torch.tensor(weights, device=device, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        'task': 'multilabel', 'model': args.arch, 'classes': CLASSES,
        'mode': 'xray_exported', 'size': args.size,
        'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'hu_window': None,
        'manifest_sha256': manifest_sha256(args.manifest), 'seed': args.seed,
        'pos_weight': None if args.no_pos_weights else weights.tolist(),
        'positive_counts_train': dict(zip(CLASSES, positives.astype(int).tolist())),
        'args': vars(args),
        'warning': 'NIH weak labels; official test data never used for selection',
    }
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    best, bad, history = -1., 0, []
    print(f'Training {args.arch} NIH multi-label on {device}: {len(train_ds)} train / {len(val_ds)} val', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(y)
        prob, y = predict_loader(model, val_loader, device)
        m = multilabel_metrics(y, prob, CLASSES)
        score = m['macro_average_precision']
        if score is None:
            raise ValueError('Validation lacks positive/negative examples')
        row = {'epoch': epoch, 'train_loss': total / len(train_ds), 'val_macro_ap': score,
               'val_macro_auroc': m['macro_auroc'], 'val_valid_classes': m['valid_auc_classes']}
        history.append(row)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch:03d} loss={row["train_loss"]:.4f} val-macro-AP={score:.4f} val-macro-AUROC={m["macro_auroc"]:.4f}', flush=True)
        if score > best + 1e-6:
            best, bad = score, 0
            # Fit decision thresholds only on validation predictions of selected model.
            # Final test will use these thresholds unchanged.
            thresholds = choose_thresholds(y, prob, CLASSES)
            torch.save({'model_state': model.state_dict(), 'config': cfg,
                        'best_epoch': epoch, 'val_metrics': m,
                        'thresholds': thresholds}, out / 'best.pt')
        else:
            bad += 1
        if bad >= args.patience:
            print('Early stopping', flush=True)
            break
    print(f'Best val macro-AP: {best:.4f}. Evaluate original test partition only after selection.')


if __name__ == '__main__':
    main()
