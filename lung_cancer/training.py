"""Training procedure for the custom CNN."""

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from lung_cancer.dataset import discover_samples, make_tf_dataset, split_samples
from lung_cancer.model import CLASS_NAMES, build_model

LOGGER = logging.getLogger(__name__)


def train(
    data_root: str | Path,
    output_dir: str | Path = "artifacts",
    manifest: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> dict[str, object]:
    """Train, validate, and persist a Keras checkpoint plus split metadata.

    Defaults not supplied by the paper are disclosed assumptions: Adam's
    Keras default learning rate (0.001), batch 32, 50 epochs, seed 42,
    70/15/15 stratified split, class weighting, and early stopping (patience 8).
    No augmentation is introduced because the paper specifies none.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tf.keras.utils.set_random_seed(seed)
    samples = discover_samples(data_root, manifest)
    train_samples, validation_samples, test_samples = split_samples(
        samples, test_fraction, validation_fraction, seed
    )
    train_ds = make_tf_dataset(train_samples, batch_size, shuffle=True, seed=seed)
    validation_ds = make_tf_dataset(validation_samples, batch_size)
    test_ds = make_tf_dataset(test_samples, batch_size)
    labels = np.asarray([item.label for item in train_samples])
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=labels)
    class_weight = {idx: float(weight) for idx, weight in enumerate(weights)}
    model = build_model()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "model.weights.h5"
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.CSVLogger(destination / "training.csv"),
    ]
    LOGGER.info(
        "Training on %d images; validating on %d; testing on %d",
        len(train_samples),
        len(validation_samples),
        len(test_samples),
    )
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    best_model = build_model(compile_model=False)
    best_model.load_weights(checkpoint)
    best_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    metadata: dict[str, object] = {
        "class_names": list(CLASS_NAMES),
        "checkpoint": checkpoint.name,
        "checkpoint_format": "keras_weights",
        "image_shape": [256, 256, 1],
        "seed": seed,
        "split_counts": {
            "train": len(train_samples),
            "validation": len(validation_samples),
            "test": len(test_samples),
        },
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_completed": len(history.epoch),
        "assumptions": {
            "batch_size": batch_size,
            "max_epochs": epochs,
            "adam_learning_rate": 0.001,
            "split": [1 - test_fraction - validation_fraction, validation_fraction, test_fraction],
            "class_weighting": "balanced",
            "early_stopping_patience": 8,
            "augmentation": "none",
        },
    }
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("Best checkpoint saved to %s; test accuracy %.4f", checkpoint, test_accuracy)
    return metadata
