"""HTTP API integration tests using an injected deterministic test model."""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lung_cancer import api


class DeterministicModel:
    """Test double for API plumbing; never used as a trained model."""

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, 256, 256, 1)
        return np.asarray([[0.8, 0.1, 0.1]], dtype=np.float32)


def _png_bytes() -> bytes:
    image = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (64, 1)))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _install_service(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.weights.h5"
    checkpoint.write_bytes(b"test-only checkpoint marker")
    monkeypatch.setattr(api.settings, "model_path", checkpoint)
    monkeypatch.setattr(
        api.InferenceService,
        "from_checkpoint",
        classmethod(lambda cls, path: cls(DeterministicModel(), "test")),
    )


def test_classify_upload_returns_structured_result(monkeypatch, tmp_path) -> None:
    _install_service(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/classify", files={"file": ("slice.png", _png_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "Normal"
    assert body["probabilities"]["Normal"] == pytest.approx(0.8)


def test_classify_rejects_unsupported_extension(monkeypatch, tmp_path) -> None:
    _install_service(monkeypatch, tmp_path)
    with TestClient(api.app) as client:
        response = client.post("/api/v1/classify", files={"file": ("scan.gif", b"data")})
    assert response.status_code == 415


def test_classify_requires_configured_api_key(monkeypatch, tmp_path) -> None:
    _install_service(monkeypatch, tmp_path)
    monkeypatch.setattr(api.settings, "api_key", "test-secret")
    with TestClient(api.app) as client:
        denied = client.post(
            "/api/v1/classify", files={"file": ("slice.png", _png_bytes(), "image/png")}
        )
        accepted = client.post(
            "/api/v1/classify",
            files={"file": ("slice.png", _png_bytes(), "image/png")},
            headers={"X-API-Key": "test-secret"},
        )
    assert denied.status_code == 401
    assert accepted.status_code == 200
