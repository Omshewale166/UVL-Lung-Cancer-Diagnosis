# Dataset source and provenance: IQ-OTH/NCCD

## Verified provider

- Provider: Mendeley Data
- Dataset: **The IQ-OTH/NCCD lung cancer dataset**, version 4
- DOI: [10.17632/bhmdr45bh2.4](https://doi.org/10.17632/bhmdr45bh2.4)
- Official dataset record: <https://data.mendeley.com/datasets/bhmdr45bh2/4>
- License shown on the provider record: **CC BY 4.0**. Any redistribution or publication using the data must follow the license terms and cite the dataset. See <https://creativecommons.org/licenses/by/4.0/>.
- Required citation: Alyasriy, Hamdalla; AL-Huseiny, Muayed (2023), “The IQ-OTH/NCCD lung cancer dataset”, Mendeley Data, V4, DOI: 10.17632/bhmdr45bh2.4.

## Files in this project

- Original provider download saved by `python download_dataset.py`: `data/raw/iqothnccd-v4.zip`
- The verified archive contains a nested image ZIP. The downloader reads it without extracting unrelated package materials and copies the labeled JPEGs into the project's training loader layout:
  - `data/lung/Normal/`: 416 images
  - `data/lung/Benign/`: 120 images
  - `data/lung/Malignant/`: 561 images

The verified archive observed for this release is 159,182,970 bytes with SHA-256 `CAB9CF2206F0188DA218B88D0EA6E8C9D9EA387762BA6260E87B1AD83F8A314B`.

## Count and format discrepancy

The Mendeley Data record describes 1,190 CT images/slices from 110 cases (55 normal, 15 benign, and 40 malignant). The downloaded version 4 package contains a nested image archive with **1,097 JPEG image files**: 416 normal, 120 benign, and 561 malignant. The 93-file difference is a discrepancy between the provider's description and the downloadable package. The missing files have not been fabricated or inferred.

The downloadable archive contains JPEG images, not DICOM (`.dcm`) files. The provider describes the source scans as CT images, but this package does not expose the original DICOM pixel data, acquisition metadata, or a per-image DICOM header. Do not represent the local JPEG files as original DICOM studies.

## Important training limitation

The downloaded package does not include a reliable patient/case identifier for every image. The current project loader may therefore split images rather than patients/cases into training, validation, and test groups. Images from one patient or scan may then appear in multiple splits, causing data leakage and overly optimistic evaluation results. Treat metrics from that split as exploratory, and do not present them as patient-independent clinical performance. A patient-level evaluation requires a verified case-to-image mapping or another dataset release that provides one.

## Data handling

Run `python download_dataset.py` from the project root to obtain the archive, verify its SHA-256, and arrange the three image directories. The script fails closed on a checksum or expected class-count mismatch. Image data under `data/` is excluded from Git by the project's ignore rules. Keep this provenance note and the required dataset citation with any report, model release, or redistribution of derived data. Verify the provider's current license and record before redistributing the archive.
