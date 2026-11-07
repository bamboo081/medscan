"""Run the held-out test set ONCE for the selected checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from medscan.data import MedicalDataset, read_manifest, manifest_sha256
from medscan.evaluation import evaluate_model
from medscan.model import build_model


def evaluate(checkpoint, manifest, output, batch_size=16, workers=2):
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = cp["config"]
    if manifest_sha256(manifest) != cfg["manifest_sha256"]:
        raise ValueError("Manifest has changed since training: refusing invalid test comparison")
    if cfg.get("task") == "multilabel":
        return evaluate_multilabel(cp, manifest, output, batch_size, workers)
    df = read_manifest(manifest, image_level=cfg.get("image_level", False))
    test_df = df[df.split == "test"].reset_index(drop=True)
    classes = cfg["classes"]
    ds = MedicalDataset(test_df, classes, mode=cfg["mode"], size=cfg["size"],
                        hu_window=cfg.get("hu_window"), mean=cfg["mean"], std=cfg["std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], classes, pretrained=False).to(device)
    model.load_state_dict(cp["model_state"], strict=True)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda")
    metrics, probs, true = evaluate_model(model, loader, device, classes, test_df)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = test_df[["path", "label"] if cfg.get("image_level", False) else ["path", "patient_id", "label"]].copy()
    predictions["predicted"] = [classes[i] for i in probs.argmax(axis=1)]
    for i, label in enumerate(classes):
        predictions[f"score_{label}"] = probs[:, i]
    predictions.to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps({k:v for k,v in metrics.items() if k not in {"class_report","confusion_matrix","patient_confusion_matrix"}}, indent=2))
    print("Saved confusion matrices, per-class metrics and local predictions; scores are NOT calibrated risks.")
    if cfg.get("image_level", False):
        print("WARNING: CT JPEG scores are exploratory. No verified patient independence, and earlier CT test partitions were examined.")


def evaluate_multilabel(cp, manifest, output, batch_size, workers):
    from train_nih import NIHData, CLASSES, read_nih_manifest, predict_loader
    from medscan.multilabel import multilabel_metrics
    cfg = cp["config"]
    if cfg["classes"] != CLASSES or cfg["mode"] != "xray_exported":
        raise ValueError("Unsupported multi-label checkpoint configuration")
    df = read_nih_manifest(manifest)
    test_df = df[df.split == "test"].reset_index(drop=True)
    ds = NIHData(test_df, cfg["size"], mean=cfg["mean"], std=cfg["std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], cfg["classes"], pretrained=False).to(device)
    model.load_state_dict(cp["model_state"], strict=True)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    probs, true = predict_loader(model, loader, device)
    thresholds = cp.get("thresholds", [0.5] * len(CLASSES))
    metrics = multilabel_metrics(true, probs, CLASSES, thresholds=thresholds)
    metrics["patients"] = int(test_df.patient_id.nunique())
    metrics["thresholds_from_validation"] = "thresholds" in cp
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = test_df[["path", "patient_id", "labels"]].copy()
    for i, label in enumerate(CLASSES):
        predictions[f"score_{label}"] = probs[:, i]
        predictions[f"flag_{label}"] = probs[:, i] >= thresholds[i]
    predictions.to_csv(out / "test_predictions.csv", index=False)
    summary = {k: v for k, v in metrics.items() if k != "per_class"}
    print(json.dumps(summary, indent=2))
    print("Saved per-class AP, AUROC, F1 and scores; neither probabilities nor flags are medical diagnoses.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="runs/mri_baseline/test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    evaluate(**vars(args))
