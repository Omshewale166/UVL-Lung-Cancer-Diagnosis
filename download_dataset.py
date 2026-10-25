"""Download and arrange the verified IQ-OTH/NCCD version 4 dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from io import BytesIO
import zipfile
from pathlib import Path, PurePosixPath

DATASET_URL = "https://data.mendeley.com/public-api/zip/bhmdr45bh2/download/4"
EXPECTED_SHA256 = "CAB9CF2206F0188DA218B88D0EA6E8C9D9EA387762BA6260E87B1AD83F8A314B"
CLASS_FOLDER_MAP = {
    "Normal cases": "Normal",
    "Bengin cases": "Benign",
    "Malignant cases": "Malignant",
}
EXPECTED_COUNTS = {"Normal": 416, "Benign": 120, "Malignant": 561}
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def obtain_archive(archive_path: Path) -> None:
    """Reuse a verified archive or download and verify it before installing it."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        actual_digest = sha256_file(archive_path)
        if actual_digest != EXPECTED_SHA256:
            raise ValueError(
                f"Existing archive checksum mismatch at {archive_path}: {actual_digest}"
            )
        print(f"Using verified archive: {archive_path}")
        return

    temporary_path: Path | None = None
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "lung-cancer-diagnosis-dataset-downloader/1.0"},
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="iqothnccd-", suffix=".part", dir=archive_path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
        actual_digest = sha256_file(temporary_path)
        if actual_digest != EXPECTED_SHA256:
            raise ValueError(
                "Downloaded dataset checksum mismatch; refusing to extract it. "
                f"Expected {EXPECTED_SHA256}, got {actual_digest}."
            )
        os.replace(temporary_path, archive_path)
        temporary_path = None
        print(f"Downloaded and verified archive: {archive_path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def copy_class_images(archive_path: Path, data_root: Path) -> dict[str, int]:
    """Copy only the three labeled JPEG classes from the provider's nested ZIP."""
    counts = {label: 0 for label in EXPECTED_COUNTS}
    with zipfile.ZipFile(archive_path) as outer_archive:
        nested_archives = [
            info.filename for info in outer_archive.infolist()
            if info.filename.lower().endswith(".zip")
        ]
        if len(nested_archives) != 1:
            raise ValueError(
                "Expected exactly one nested image ZIP in the verified provider archive; "
                f"found {len(nested_archives)}."
            )
        nested_bytes = outer_archive.read(nested_archives[0])

    with zipfile.ZipFile(BytesIO(nested_bytes)) as image_archive:
        for member in image_archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) < 3:
                continue
            label = CLASS_FOLDER_MAP.get(parts[-2])
            filename = parts[-1]
            if label is None or Path(filename).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            if filename in {".", ".."} or "/" in filename or "\\" in filename:
                raise ValueError(f"Unexpected image filename in provider archive: {filename}")

            image_bytes = image_archive.read(member)
            class_directory = data_root / "lung" / label
            class_directory.mkdir(parents=True, exist_ok=True)
            destination = class_directory / filename
            if destination.exists():
                if destination.read_bytes() != image_bytes:
                    raise FileExistsError(
                        f"Refusing to overwrite different local image: {destination}"
                    )
            else:
                temporary_image = destination.with_name(destination.name + ".part")
                temporary_image.write_bytes(image_bytes)
                os.replace(temporary_image, destination)
            counts[label] += 1

    if counts != EXPECTED_COUNTS:
        raise ValueError(
            "Provider archive contents do not match the verified version 4 counts. "
            f"Expected {EXPECTED_COUNTS}, found {counts}."
        )
    return counts


def main() -> None:
    """Download, verify, and arrange the dataset beneath the selected data root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Project data directory (default: data)",
    )
    arguments = parser.parse_args()
    data_root = arguments.data_dir.expanduser().resolve()
    archive_path = data_root / "raw" / "iqothnccd-v4.zip"
    obtain_archive(archive_path)
    counts = copy_class_images(archive_path, data_root)
    print(f"Dataset installed under {data_root / 'lung'}:")
    for label, count in counts.items():
        print(f"  {label}: {count} JPEG images")
    print("Source details and the provider's count discrepancy: docs/dataset_source_iqothnccd.md")


if __name__ == "__main__":
    main()
