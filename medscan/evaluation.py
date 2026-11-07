from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report


def metrics_from_probs(probs, y_true, classes, group_ids=None):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    if len(probs) == 0 or probs.shape != (len(labels), len(classes)):
        raise ValueError("Invalid probability or label dimensions")
    preds = probs.argmax(axis=1)
    ix = list(range(len(classes)))
    out = {
        "samples": len(labels),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=ix, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=ix).tolist(),
        "class_report": classification_report(labels, preds, labels=ix, target_names=classes,
                                              zero_division=0, output_dict=True),
    }
    if group_ids is not None:
        groups = defaultdict(list)
        for i, patient in enumerate(group_ids):
            groups[str(patient)].append(i)
        if any(len(set(labels[ids])) != 1 for ids in groups.values()):
            raise ValueError("Patient with conflicting labels")
        gp = np.stack([probs[ids].mean(axis=0) for ids in groups.values()])
        gt = np.asarray([labels[ids[0]] for ids in groups.values()])
        out["patients"] = len(gt)
        out["patient_macro_f1"] = float(f1_score(gt, gp.argmax(axis=1), average="macro",
                                                    labels=ix, zero_division=0))
        out["patient_confusion_matrix"] = confusion_matrix(gt, gp.argmax(axis=1), labels=ix).tolist()
    return out


@torch.inference_mode()
def evaluate_model(model, loader, device, classes, df):
    model.eval()
    probs_list, y_list = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs_list.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        y_list.append(y.numpy())
    probs = np.concatenate(probs_list)
    true = np.concatenate(y_list)
    metrics = metrics_from_probs(probs, true, classes, df.patient_id.tolist() if "patient_id" in df else None)
    return metrics, probs, true
