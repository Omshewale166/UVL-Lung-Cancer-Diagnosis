"""FastAPI upload and classification endpoints."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from lung_cancer.config import get_settings
from lung_cancer.inference import InferenceService
from lung_cancer.middleware import RequestSizeLimitMiddleware
from lung_cancer.logging_config import configure_logging
from lung_cancer.preprocessing import ImageInputError, decode_image

settings = get_settings()
configure_logging(settings.log_level)
LOGGER = logging.getLogger(__name__)
_service: InferenceService | None = None


class HealthResponse(BaseModel):
    """Process and model readiness status."""

    status: str
    model_ready: bool


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load an available checkpoint once; allow health/UI to serve before training."""
    global _service
    if settings.model_path.is_file():
        try:
            _service = InferenceService.from_checkpoint(settings.model_path)
        except Exception:
            LOGGER.exception("Model checkpoint could not be loaded")
            _service = None
    else:
        LOGGER.warning("No trained checkpoint found at %s", settings.model_path)
    yield
    _service = None


app = FastAPI(
    title="Lung CT Slice Classifier",
    description=(
        "Research implementation of a three-class CNN. Outputs are not clinical diagnoses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_bytes=settings.max_upload_bytes + 64 * 1024,
    protected_path="/api/v1/classify",
)


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require configured API key when the service is exposed beyond localhost."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report process health and checkpoint readiness."""
    return HealthResponse(status="ok", model_ready=_service is not None)


@app.post("/api/v1/classify", tags=["inference"], dependencies=[Depends(verify_api_key)])
async def classify(file: UploadFile = File(...)) -> dict[str, object]:
    """Classify one PNG, JPEG, or single-frame DICOM image."""
    service = _service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoint is loaded. Train a model and configure "
                "LUNG_MODEL_PATH."
            ),
        )
    allowed = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail="Supported formats: JPEG, PNG, and single-frame DICOM",
        )
    try:
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail="Uploaded image exceeds configured size limit",
            )
        image = await run_in_threadpool(decode_image, data, file.filename or "image")
        report = await run_in_threadpool(service.predict, image)
        return report.to_dict()
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Inference request failed")
        raise HTTPException(status_code=500, detail="Image classification failed") from exc
    finally:
        await file.close()


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    """Serve a small local upload interface."""
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-store"},
    )
