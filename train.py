"""Train a modality-specific 2D research classifier; never uses test during training."""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from medscan.data import MedicalDataset, manifest_sha256, read_manifest
from medscan.evaluation import evaluate_model
from medscan.model import MODELS, build_model
from medscan.preprocessing import IMAGENET_MEAN, IMAGENET_STD, MODES


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.accum < 1 or args.workers < 0:
        raise ValueError("Epochs, batch size and accumulation must be positive; workers nonnegative")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA not available. Confirm NVIDIA driver and CUDA-enabled torch installation.")
    if args.mode == "ct_hu" and (args.hu_min is None or args.hu_max is None or args.hu_min >= args.hu_max):
        raise ValueError("CT HU needs --hu-min and --hu-max (fixed, appropriate for task)")
    hu_window = [args.hu_min, args.hu_max] if args.mode == "ct_hu" else None
    if args.mode != "ct_hu" and (args.hu_min is not None or args.hu_max is not None):
        raise ValueError("HU windows only valid with ct_hu")
    if args.image_level and args.mode != "ct_exported":
        raise ValueError("Image-level mode is restricted to exploratory exported CT")
    manifest = read_manifest(args.manifest, image_level=args.image_level)
    if args.image_level and "group_id" not in manifest and not args.allow_ungrouped_exploratory:
        raise ValueError("Unprotected CT image split: use prepare_ct_grouped.py and its grouped manifest. "
                         "Override only for a clearly labeled historical baseline with --allow-ungrouped-exploratory.")
    classes = sorted(manifest.label.unique().tolist())
    if len(classes) < 2:
        raise ValueError("Need two or more classes")
    train_df = manifest[manifest.split == "train"].reset_index(drop=True)
    val_df = manifest[manifest.split == "val"].reset_index(drop=True)
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    if args.stats:
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
        if stats["manifest_sha256"] != manifest_sha256(args.manifest) or stats["mode"] != args.mode or stats["size"] != args.size:
            raise ValueError("Input stats do not match manifest, mode, or image size")
        mean, std = stats["mean"], stats["std"]
    kwargs = dict(mode=args.mode, size=args.size, hu_window=hu_window, mean=mean, std=std)
    train_ds = MedicalDataset(train_df, classes, augment=args.augment, **kwargs)
    val_ds = MedicalDataset(val_df, classes, augment=False, **kwargs)
    loader_opts = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, **loader_opts)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_opts)
    model = build_model(args.arch, classes, pretrained=not args.no_pretrained).to(device)
    counts = np.bincount([classes.index(c) for c in train_df.label], minlength=len(classes)).astype(float)
    class_weights = torch.as_tensor(len(train_df) / (len(classes) * counts),
                                    dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model": args.arch, "classes": classes, "mode": args.mode, "image_level": args.image_level,
        "proxy_grouped": bool("group_id" in manifest),
        "ct_limitation": "JPEG CT, no patient IDs, image-level exploration only" if args.image_level else None,
        "hu_window": hu_window, "size": args.size, "mean": mean, "std": std,
        "manifest_sha256": manifest_sha256(args.manifest), "seed": args.seed,
        "dataset_splits": manifest.groupby("split").size().to_dict(),
        "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "grad_accum_steps": args.accum, "lr": args.lr, "augment": args.augment,
        "pretrained": not args.no_pretrained, "torch": str(torch.__version__),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Training {args.arch} on {device} | {len(train_df)} train, {len(val_df)} val | {classes}", flush=True)
    best_f1 = -1.0
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            running_loss += float(loss.detach().item()) * len(y)
            scaler.scale(loss / args.accum).backward()
            if (batch_idx + 1) % args.accum == 0 or batch_idx + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, _, _ = evaluate_model(model, val_loader, device, classes, val_df)
        score = val_metrics["macro_f1"] if args.image_level else val_metrics["patient_macro_f1"]
        line = {"epoch":epoch,"train_loss":running_loss / len(train_df),
                "val_macro_f1":val_metrics["macro_f1"],"val_patient_macro_f1":None if args.image_level else score,
                "val_balanced_accuracy":val_metrics["balanced_accuracy"]}
        history.append(line)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch:03d}: loss={line['train_loss']:.4f} val-image-F1={line['val_macro_f1']:.4f}" + ("" if args.image_level else f" val-patient-F1={score:.4f}"), flush=True)
        if score > best_f1 + 1e-6:
            best_f1 = score
            bad_epochs = 0
            # Local .pt checkpoint: only load self-generated/trusted artifacts.
            torch.save({"model_state":model.state_dict(), "config":cfg,
                        "best_epoch":epoch, "val_metrics":val_metrics}, output / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement")
            break
    print(f"Best validation {'image' if args.image_level else 'patient'} macro-F1: {best_f1:.4f}. Final TEST only with evaluate.py.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-level", action="store_true", help="Exploratory exported CT only; no patient independence")
    p.add_argument("--allow-ungrouped-exploratory", action="store_true",
                   help="Unsafe historical compatibility: bypass grouped CT protection; never claim independent testing")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="runs/mri_baseline")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--arch", choices=MODELS, default="resnet18")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--hu-min", type=float)
    p.add_argument("--hu-max", type=float)
    p.add_argument("--stats", help="Optional train-only intensity statistics from compute_stats.py")
    train(p.parse_args())
