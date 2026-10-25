"""Inference service orchestration."""

import logging
from pathlib import Path

import numpy as np
from tensorflow import keras

from lung_cancer.preprocessing import preprocess_image
from lung_cancer.reports import ClassificationReport, create_report

LOGGER = logging.getLogger(__name__)


class InferenceService:
    """Own a validated model and expose single-image inference."""

    def __init__(self, model: keras.Model, model_version: str = "local") -> None:
        self.model = model
        self.model_version = model_version

    @classmethod
    def from_checkpoint(cls, model_path: str | Path) -> "InferenceService":
        """Load a trained model checkpoint from disk."""
        from lung_cancer.model import load_model

        path = Path(model_path)
        return cls(load_model(path), model_version=path.stem)

    def predict(self, image: np.ndarray) -> ClassificationReport:
        """Preprocess a decoded image, predict its class, and format the output."""
        tensor = preprocess_image(image)
        probabilities = self.model.predict(tensor[np.newaxis, ...], verbose=0)[0]
        LOGGER.info("Completed image classification")
        return create_report(probabilities, self.model_version)
