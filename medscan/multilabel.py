"""Metrics and validation-only decision thresholds for weak-label NIH findings."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
)


def choose_thresholds(y, prob, classes, default=0.5):
    """Fit thresholds ONLY on validation data, never on test data."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    thresholds = []
    for i, _ in enumerate(classes):
        if len(np.unique(y[:, i])) < 2 or y[:, i].sum() < 5:
            thresholds.append(default)
            continue
        precision, recall, t = precision_recall_curve(y[:, i], prob[:, i])
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        # Avoid degenerate thresholds at extreme edges for unstable rare classes.
        ix = int(np.argmax(f1))
        thresholds.append(float(np.clip(t[ix], 0.01, 0.99)))
    return thresholds


def multilabel_metrics(y, prob, classes, thresholds=None):
    y = np.asarray(y, dtype=np.int32)
    prob = np.asarray(prob, dtype=np.float64)
    if y.ndim != 2 or y.shape != prob.shape or y.shape[1] != len(classes):
        raise ValueError("Invalid multi-label shapes")
    if not np.isfinite(prob).all():
        raise ValueError("Nonfinite scores")
    thresholds = [0.5] * len(classes) if thresholds is None else list(thresholds)
    if len(thresholds) != len(classes):
        raise ValueError("Threshold/class count mismatch")
    per = {}
    aps, aucs = [], []
    preds = (prob >= np.asarray(thresholds)[None, :]).astype(int)
    for j, label in enumerate(classes):
        has_both = len(np.unique(y[:, j])) == 2
        ap = float(average_precision_score(y[:, j], prob[:, j])) if has_both else None
        auc = float(roc_auc_score(y[:, j], prob[:, j])) if has_both else None
        if ap is not None:
            aps.append(ap)
            aucs.append(auc)
        per[label] = {
            "positives": int(y[:, j].sum()), "auroc": auc, "average_precision": ap,
            "threshold": float(thresholds[j]),
            "precision": float(precision_score(y[:, j], preds[:, j], zero_division=0)),
            "recall": float(recall_score(y[:, j], preds[:, j], zero_division=0)),
            "f1": float(f1_score(y[:, j], preds[:, j], zero_division=0)),
        }
    return {
        "samples": int(len(y)), "macro_average_precision": float(np.mean(aps)) if aps else None,
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, preds, average="micro", zero_division=0)),
        "valid_auc_classes": len(aucs), "per_class": per,
        "warning": "Weak NIH report-mined annotations; scores are not calibrated risks, diagnoses or clinical validation.",
    }
