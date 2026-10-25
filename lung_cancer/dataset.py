"""Dataset discovery, validation, splitting, and TensorFlow input pipeline."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from lung_cancer.model import CLASS_NAMES
from lung_cancer.preprocessing import load_and_preprocess

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}


@dataclass(frozen=True)
class Sample:
    """An image path, integer class label, and optional patient grouping key."""

    path: Path
    label: int
    patient_id: str | None = None


def discover_samples(
    data_root: str | Path, manifest: str | Path | None = None
) -> list[Sample]:
    """Read a CSV manifest or class folders named Normal/Benign/Malignant.

    A manifest requires ``image_path,label`` columns and may include
    ``patient_id``; image paths are resolved relative to the manifest's
    directory. Patient IDs enable leakage-resistant grouped train/validation/
    test splits. Without IDs, files are split stratified by class.
    """
    root = Path(data_root).expanduser().resolve()
    samples: list[Sample] = []
    label_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if manifest is not None:
        manifest_path = Path(manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"image_path", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("Manifest must contain image_path,label columns")
            for row in reader:
                raw_label = row["label"].strip()
                try:
                    label = label_index[raw_label]
                except KeyError as exc:
                    raise ValueError(f"Unsupported class label: {raw_label}") from exc
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = manifest_path.parent / image_path
                if (
                    image_path.suffix.lower() not in SUPPORTED_SUFFIXES
                    or not image_path.is_file()
                ):
                    raise ValueError(f"Missing or unsupported image in manifest: {image_path}")
                patient_id = (row.get("patient_id") or "").strip() or None
                samples.append(Sample(image_path.resolve(), label, patient_id))
    else:
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.is_dir():
                raise ValueError(f"Expected class folder: {class_dir}")
            samples.extend(
                Sample(path.resolve(), label)
                for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
    counts = np.bincount([sample.label for sample in samples], minlength=len(CLASS_NAMES))
    if not samples or np.any(counts < 8):
        raise ValueError(
            "Need at least eight readable images for every class for stratified splits"
        )
    LOGGER.info(
        "Discovered %d images: %s", len(samples), dict(zip(CLASS_NAMES, counts.tolist()))
    )
    return samples


def split_samples(
    samples: list[Sample], test_fraction: float = 0.15, validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Create stratified splits; preserve patient groups when all IDs exist."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Test plus validation fractions must be less than one")
    indices = np.arange(len(samples))
    labels = np.asarray([sample.label for sample in samples])
    patient_ids = [sample.patient_id for sample in samples]
    if all(patient_ids):
        from sklearn.model_selection import StratifiedGroupKFold

        groups = np.asarray(patient_ids)
        test_splits = max(2, round(1 / test_fraction))
        splitter = StratifiedGroupKFold(n_splits=test_splits, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(indices, labels, groups))
        relative_val = validation_fraction / (1 - test_fraction)
        train_val_labels = labels[train_val_idx]
        val_splits = max(2, round(1 / relative_val))
        val_splitter = StratifiedGroupKFold(
            n_splits=val_splits, shuffle=True, random_state=seed + 1
        )
        train_rel, val_rel = next(
            val_splitter.split(train_val_idx, train_val_labels, groups[train_val_idx])
        )
        train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
    else:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=seed, stratify=labels
        )
        relative_val = validation_fraction / (1 - test_fraction)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=seed + 1,
            stratify=labels[train_val_idx],
        )
    result = tuple([samples[int(i)] for i in part] for part in (train_idx, val_idx, test_idx))
    for name, part in zip(("train", "validation", "test"), result):
        if {sample.label for sample in part} != set(range(len(CLASS_NAMES))):
            raise ValueError(f"{name} split does not contain every class; supply more samples")
    return result  # type: ignore[return-value]


def make_tf_dataset(
    samples: list[Sample], batch_size: int = 32, shuffle: bool = False, seed: int = 42,
) -> tf.data.Dataset:
    """Build a deterministic tf.data pipeline around the canonical preprocessor."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    paths = [str(sample.path) for sample in samples]
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    def load_one(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def load_numpy(path_bytes: bytes) -> np.ndarray:
            return load_and_preprocess(path_bytes.decode("utf-8"))

        image = tf.numpy_function(load_numpy, [path], Tout=tf.float32)
        image.set_shape((256, 256, 1))
        return image, label

    return dataset.map(load_one, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def iter_numpy(samples: list[Sample]) -> Iterator[tuple[np.ndarray, int]]:
    """Yield individually preprocessed samples for evaluation and visualization."""
    for sample in samples:
        yield load_and_preprocess(sample.path), sample.label
