"""Unit tests for the image preprocessing contract."""

import numpy as np
import pytest

from lung_cancer.preprocessing import ImageInputError, preprocess_image


def test_preprocessing_returns_fixed_normalized_tensor() -> None:
    image = np.tile(np.arange(320, dtype=np.uint8), (240, 1))
    processed = preprocess_image(image)
    assert processed.shape == (256, 256, 1)
    assert processed.dtype == np.float32
    assert np.isin(processed, (0.0, 1.0)).all()


def test_preprocessing_rejects_empty_input() -> None:
    with pytest.raises(ImageInputError):
        preprocess_image(np.empty((0, 0), dtype=np.uint8))


def test_published_geometry_rejects_other_size() -> None:
    with pytest.raises(ValueError, match="image_size=256"):
        preprocess_image(np.ones((16, 16), dtype=np.uint8), image_size=128)


def test_decode_rejects_oversized_raster_before_loading(monkeypatch) -> None:
    from lung_cancer import preprocessing

    class OversizedImage:
        width = preprocessing.MAX_DECODED_PIXELS + 1
        height = 1

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def load(self) -> None:
            raise AssertionError("oversized image should be rejected before decoding")

    monkeypatch.setattr(preprocessing.Image, "open", lambda _: OversizedImage())
    with pytest.raises(ImageInputError, match="maximum decoded pixel count"):
        preprocessing.decode_image(b"header", "large.png")
