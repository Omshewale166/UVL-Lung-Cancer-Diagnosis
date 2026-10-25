"""Research-specified CT slice preprocessing."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_DECODED_PIXELS = 40_000_000


class ImageInputError(ValueError):
    """Raised when an uploaded or on-disk image cannot be decoded safely."""


def decode_image(data: bytes, filename: str = "image") -> np.ndarray:
    """Decode PNG/JPEG or a single-frame DICOM object into a grayscale array.

    DICOM modality transforms are applied. Values are min-max scaled to uint8
    for compatibility with the paper's 8-bit OpenCV preprocessing pipeline.
    Multi-frame and volumetric studies are rejected because the paper specifies
    a 2-D image classifier.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_modality_lut

            dataset = pydicom.dcmread(BytesIO(data), force=False)
            frame_count = int(dataset.get("NumberOfFrames", 1))
            dimensions = int(dataset.get("Rows", 0)) * int(dataset.get("Columns", 0))
            if frame_count != 1:
                raise ImageInputError("Only single-frame 2-D DICOM images are supported")
            if dimensions < 1 or dimensions > MAX_DECODED_PIXELS:
                raise ImageInputError(
                    "DICOM dimensions are invalid or exceed the decode limit"
                )
            pixels = np.asarray(apply_modality_lut(dataset.pixel_array, dataset))
        except ImageInputError:
            raise
        except Exception as exc:
            raise ImageInputError(f"Unable to decode DICOM image: {exc}") from exc
        if pixels.ndim != 2:
            raise ImageInputError("Only single-frame 2-D DICOM images are supported")
        if dataset.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            pixels = np.max(pixels) - pixels
        pixels = np.nan_to_num(pixels.astype(np.float32), copy=False)
        low, high = float(pixels.min()), float(pixels.max())
        if high <= low:
            raise ImageInputError("DICOM image has no intensity variation")
        return np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)

    try:
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > MAX_DECODED_PIXELS:
                raise ImageInputError("Image exceeds the maximum decoded pixel count")
            image.load()
            rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageInputError(f"Unsupported or corrupt image file: {exc}") from exc


def preprocess_image(image: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Apply RGB conversion, resize, grayscale, blur, Otsu, normalization,
    erosion, and dilation, returning ``(image_size, image_size, 1)`` float32.

    The paper leaves morphology details unspecified; a 3x3 rectangular kernel
    and one iteration for each operation are the documented defaults here.
    """
    if image_size != 256:
        raise ValueError("The published CNN geometry requires image_size=256")
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageInputError("Image is empty")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        raise ImageInputError("Expected a 2-D grayscale or 3-channel image")
    if gray.shape != (image_size, image_size):
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    normalized = binary.astype(np.float32) / 255.0
    morph_kernel = np.ones((3, 3), dtype=np.uint8)
    binary = (normalized * 255).astype(np.uint8)
    eroded = cv2.erode(binary, morph_kernel, iterations=1)
    dilated = cv2.dilate(eroded, morph_kernel, iterations=1)
    return (dilated.astype(np.float32) / 255.0)[..., np.newaxis]


def load_and_preprocess(path: str | Path, image_size: int = 256) -> np.ndarray:
    """Read and preprocess a local image path."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        image = decode_image(image_path.read_bytes(), image_path.name)
    except OSError as exc:
        raise ImageInputError(f"Unable to read image {image_path}: {exc}") from exc
    return preprocess_image(image, image_size)
