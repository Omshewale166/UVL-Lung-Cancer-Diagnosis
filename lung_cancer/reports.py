"""Conservative report generation based only on classifier output."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

from lung_cancer.model import CLASS_NAMES


@dataclass(frozen=True)
class ClassificationReport:
    """Structured inference result suitable for JSON serialization."""

    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    report_text: str
    generated_at: str
    model_version: str
    disclaimer: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return asdict(self)


def create_report(
    probabilities: Sequence[float], model_version: str = "unspecified"
) -> ClassificationReport:
    """Create a transparent summary without inventing radiological findings."""
    values = [float(value) for value in probabilities]
    if len(values) != len(CLASS_NAMES) or any(not 0 <= value <= 1 for value in values):
        raise ValueError("Expected three class probabilities in [0, 1]")
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError("Class probabilities must sum to 1")
    index = max(range(len(values)), key=values.__getitem__)
    predicted_class = CLASS_NAMES[index]
    confidence = values[index]
    report_text = (
        f"The CNN's highest-probability category is {predicted_class} "
        f"({confidence:.1%} model confidence). This output is a model classification "
        "only; the model does not locate or measure lesions and has not generated "
        "image-grounded radiological findings. Review the source study and interpret "
        "this result with a qualified radiologist."
    )
    return ClassificationReport(
        predicted_class=predicted_class,
        probabilities=dict(zip(CLASS_NAMES, values)),
        confidence=confidence,
        report_text=report_text,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_version=model_version,
        disclaimer=(
            "Research software; not validated for clinical diagnosis or treatment decisions."
        ),
    )
