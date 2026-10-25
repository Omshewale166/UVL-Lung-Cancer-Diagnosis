# Lung CT Slice Classifier

Research software implementing the paper's two-dimensional, three-class custom CNN pipeline. It provides preprocessing, dataset loading, training, held-out evaluation, single-slice inference, a FastAPI endpoint, and a small browser UI.

This repository does not include a trained checkpoint or the source dataset. Training and meaningful classification require labeled data. The paper's reported 90.8% accuracy is a reported result, not a result reproduced by this code. This project is not a medical device and must not be used to diagnose, rule out, or treat disease.

## What the paper specifies

- Classes: `Normal`, `Benign`, `Malignant`.
- Input: 256 × 256 single CT slices with one grayscale channel.
- Preprocessing sequence: BGR-to-RGB color correction, resize, grayscale, Gaussian blur (5 × 5), Otsu binary threshold, divide by 255, erosion, then dilation.
- CNN: valid 3 × 3 Conv2D with 64 filters, ReLU, 2 × 2 max pool; a second valid 3 × 3 64-filter convolution and 2 × 2 max pool; flatten; 16-unit dense layer; 3-unit softmax.
- Training objective: sparse categorical cross-entropy, Adam, accuracy.
- Inputs named in the paper: JPG, PNG, and DICOM. This implementation accepts PNG/JPEG and single-frame 2-D DICOM. It does not classify an entire 3-D volume.

The architecture matches the layer shapes and parameter counts in the paper: 254 × 254 × 64, 127 × 127 × 64, 125 × 125 × 64, 62 × 62 × 64, 246,016 flattened values, and dense-layer parameter counts 3,936,272 and 51. The total is 3,973,891 trainable parameters.

## Explicit implementation assumptions

The paper omits several details, so these defaults are recorded in training metadata and can be overridden from the CLI where applicable:

| Missing detail | Implemented assumption |
| --- | --- |
| Conv2D padding and kernel | Valid padding, 3 × 3 kernels, as implied by the published output shapes and parameter counts |
| Convolution activation | ReLU, matching the stated Conv2D + ReLU blocks |
| Dense hidden activation | ReLU for the 16-unit dense layer |
| Image dimension | 256 × 256, the example dimension in the paper and required by its layer table |
| Morphology kernel/iterations | 3 × 3 rectangular kernel, one erosion followed by one dilation |
| Adam learning rate | Keras Adam default, 0.001 |
| Batch size and epoch limit | 32 and 50 |
| Train/validation/test split | Stratified 70/15/15 with seed 42; patient-grouped when manifest patient IDs are all present |
| Imbalance and stopping | Balanced class weights and early stopping with patience 8 on validation loss |
| Augmentation | None; the paper does not specify an augmentation procedure for its proposed model |
| DICOM intensity conversion | Apply the modality LUT, invert MONOCHROME1, then min-max scale the single slice to 8-bit; no CT window is specified in the paper |
| Report output | Transparent summary of predicted class and probabilities. No lesion size, location, shape, or recommendations are invented |

The paper says GPT-based report generation is a future stage, while elsewhere implying it already generated reports. It provides no trained language model, report corpus, feature-to-language projection, fusion objective, or evaluation protocol. Accordingly, this implementation does not claim to implement or validate a fine-tuned GPT vision-language model. A deterministic report formatter describes only the CNN output. A structured, image-grounded report would require the missing annotations and a specified, trained model.

## Architecture

```text
PNG / JPEG / single-frame DICOM
               |
       decode + preprocess
               |
  256 x 256 x 1 float32 tensor
               |
      custom Keras CNN
               |
  three class probabilities
               |
 conservative result formatter
          /            \
       CLI        FastAPI + browser UI
```

The service is stateless: uploaded image bytes are processed in memory and not saved. No database is necessary for the specified single-image classification workflow. Production hosting should use TLS, access controls, audited storage policies, and institutional privacy/security review. An optional `LUNG_API_KEY` can protect the API route; it is not a substitute for a full identity provider or deployment controls.

## Folder tree

```text
.
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── config.py
├── docs/
│   └── research_mapping.md
├── install_dependencies.py
├── main.py
├── requirements.txt
├── setup.py
├── lung_cancer/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── logging_config.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── reports.py
│   ├── training.py
│   ├── visualization.py
│   └── static/index.html
└── tests/
    ├── test_api.py
    ├── test_model.py
    ├── test_preprocessing.py
    └── test_reports.py
```

## Installation

Python 3.10 or 3.11 is recommended. TensorFlow 2.16.1 is pinned. On Windows, use WSL2 or Docker for TensorFlow CPU support.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Alternatively, run the included dependency installer from the activated environment:

```bash
python install_dependencies.py
```

It installs packages listed in `requirements.txt` for the Python interpreter used to run the script. Use `python install_dependencies.py --dry-run` to ask pip to resolve dependencies without installing them.

Docker:

```bash
docker build -t lung-cancer-classifier .
docker run --rm -p 8000:8000 -v ./artifacts:/app/artifacts lung-cancer-classifier
```

## Dataset layout

For class-folder datasets, use exact folder names and include image files recursively:

```text
data/lung/
├── Normal/
├── Benign/
└── Malignant/
```

Accepted suffixes are `.jpg`, `.jpeg`, `.png`, `.dcm`, and `.dicom`. Each class needs at least eight files and enough distinct patient groups for all split parts. If patient identities are available, prefer a CSV manifest so slices from a patient remain in one split:

```csv
image_path,label,patient_id
images/a001.png,Normal,patient-001
images/b014.dcm,Benign,patient-014
```

Manifest paths are relative to the CSV location unless absolute. The supported labels are case-sensitive. Without `patient_id`, splitting is stratified at the image level and cannot prevent cross-slice patient leakage; report patient-level metrics only if your split design supports them.

## Training

```bash
lseed 42
# Or with patient-ung-cancer train --data data/lung --output artifacts --epochs 50 --batch-size 32 --aware manifest:
lung-cancer train --data . --manifest data/manifest.csv --output artifacts
```

Training saves `artifacts/model.keras`, `training.csv`, and `training_metadata.json`. The checkpoint selected by minimum validation loss is evaluated once on the held-out test split; metrics are recorded in metadata. A GPU is optional. Images are decoded and preprocessed through the same function used for inference.

Training is not a substitute for a valid experimental protocol. Keep patient-level separation, inspect class counts and acquisition differences, and use an independent external test cohort for generalization claims. Reproduce the paper's reported 90.8% only if its exact data, split, preprocessing code, and checkpoint become available.

## Inference

```bash
lung-cancer predict data/lung/Malignant/example.png --model artifacts/model.keras
```

The result contains the predicted class, all three probabilities, confidence, timestamp, model version, a conservative report string, and a research-use disclaimer. A probability is the model's score, not a calibrated clinical certainty.

## Evaluation and visualization

Supply a labeled held-out directory or manifest that was not used in training:

```bash
lung-cancer evaluate --model artifacts/model.keras --data data/test --output artifacts/evaluation.json
lung-cancer plot training --input artifacts/training.csv --output artifacts/training.png
lung-cancer plot confusion --input artifacts/evaluation.json --output artifacts/confusion.png
```

Evaluation reports accuracy, macro F1, per-class precision/recall/F1/support, confusion matrix, and macro one-vs-rest ROC AUC when all classes are present. Results are written as JSON.

## API and browser interface

```bash
LUNG_MODEL_PATH=artifacts/model.keras lung-cancer serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. OpenAPI documentation is at `/docs`; readiness is at `/health`.

`POST /api/v1/classify` accepts multipart form field `file` with one supported image. Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/classify -F "file=@slice.png"
```

Response fields: `predicted_class`, `probabilities`, `confidence`, `report_text`, `generated_at`, `model_version`, and `disclaimer`. Errors use HTTP 400 (invalid image), 401 (configured API key missing/invalid), 413 (upload too large), 415 (unsupported extension), or 503 (checkpoint unavailable).

Example response:

```json
{
  "predicted_class": "Normal",
  "probabilities": {"Normal": 0.81, "Benign": 0.12, "Malignant": 0.07},
  "confidence": 0.81,
  "report_text": "The CNN's highest-probability category is Normal (81.0% model confidence). ...",
  "generated_at": "2026-10-17T10:00:00+00:00",
  "model_version": "model",
  "disclaimer": "Research software; not validated for clinical diagnosis or treatment decisions."
}
```

Configuration uses environment variables with prefix `LUNG_`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LUNG_MODEL_PATH` | `artifacts/model.keras` | Keras checkpoint |
| `LUNG_MAX_UPLOAD_BYTES` | `20971520` | Maximum accepted upload (20 MiB) |
| `LUNG_LOG_LEVEL` | `INFO` | Python log level |
| `LUNG_HOST` | `127.0.0.1` | CLI server bind address |
| `LUNG_PORT` | `8000` | CLI server port |
| `LUNG_RANDOM_SEED` | `42` | Default split/training seed |
| `LUNG_API_KEY` | unset | Optional static API key for `/api/v1/classify` |

When configured, send `X-API-Key: <value>`. Set secrets through a secret manager, not source control. The default server bind is local; do not expose the unauthenticated development service publicly.

## Tests

```bash
pytest -q
```

Tests cover preprocessing shape/range and errors, exact CNN geometry and parameter counts, report behavior, and HTTP upload handling. The API test injects a deterministic test-only model to verify request plumbing; it is not used in the application or for model evaluation.

## Troubleshooting

- **Checkpoint missing:** train a model and set `LUNG_MODEL_PATH` to the resulting `.keras` file. No weights ship with the research paper.
- **DICOM decode fails:** install the pinned `pydicom` dependency and check that the file contains decodable pixel data and is a single 2-D frame.
- **Missing class / split error:** ensure all three exact class labels exist in sufficient numbers; for patient-aware splitting, provide enough distinct patient IDs per class.
- **TensorFlow install on Windows:** use WSL2 or the provided Linux container.
- **Out-of-memory training:** reduce `--batch-size`; the fully connected layer contains about 3.94 million parameters and large datasets can need substantial RAM.
- **Model performs poorly:** check label quality, patient leakage, image acquisition differences, preprocessing consistency, and held-out cohort size. Do not infer clinical readiness from accuracy alone.

## License

MIT. Dataset terms and redistribution rights are governed by each dataset's own license and consent terms.
