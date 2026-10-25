"""Training and evaluation plots."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from lung_cancer.model import CLASS_NAMES


def plot_training_history(csv_path: str | Path, output_path: str | Path) -> Path:
    """Plot training and validation loss/accuracy from Keras CSVLogger output."""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"loss", "accuracy", "val_loss", "val_accuracy"}.issubset(rows[0]):
        raise ValueError("Training history CSV is empty or lacks required metrics")
    epochs = np.arange(1, len(rows) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, train_key, valid_key, title in (
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "accuracy", "val_accuracy", "Accuracy"),
    ):
        ax.plot(epochs, [float(row[train_key]) for row in rows], label="Training")
        ax.plot(epochs, [float(row[valid_key]) for row in rows], label="Validation")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_confusion_matrix(evaluation_json: str | Path, output_path: str | Path) -> Path:
    """Render the saved confusion matrix as a labeled image."""
    metrics = json.loads(Path(evaluation_json).read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", colorbar=False
    )
    axis.set_title("Held-out classification results")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
