"""Held-out evaluation metrics and machine-readable reports."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tensorflow import keras

from lung_cancer.dataset import discover_samples, iter_numpy
from lung_cancer.model import CLASS_NAMES, load_model


def evaluate(
    model_path: str | Path,
    data_root: str | Path,
    output_path: str | Path = "artifacts/evaluation.json",
    manifest: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate a checkpoint against an explicitly supplied labeled dataset."""
    model = load_model(model_path)
    samples = discover_samples(data_root, manifest)
    truths: list[int] = []
    probabilities: list[list[float]] = []
    for image, label in iter_numpy(samples):
        prediction = model.predict(image[np.newaxis, ...], verbose=0)[0]
        truths.append(label)
        probabilities.append(prediction.astype(float).tolist())
    y_true = np.asarray(truths)
    y_prob = np.asarray(probabilities)
    y_pred = np.argmax(y_prob, axis=1)
    metrics: dict[str, object] = {
        "sample_count": len(samples),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(3)).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(3), target_names=CLASS_NAMES, zero_division=0,
            output_dict=True,
        ),
    }
    try:
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=range(3))
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = None
        metrics["roc_auc_note"] = "ROC AUC is undefined because one or more classes are absent."
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_checkpoint_for_eval(model_path: str | Path) -> keras.Model:
    """Compatibility helper for callers needing the Keras model object."""
    return load_model(model_path)
