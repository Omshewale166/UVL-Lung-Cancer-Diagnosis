"""Tests for conservative report generation."""

import pytest

from lung_cancer.reports import create_report


def test_report_contains_probabilities_and_no_unsupported_findings() -> None:
    report = create_report([0.1, 0.2, 0.7], "test-model")
    assert report.predicted_class == "Malignant"
    assert report.confidence == pytest.approx(0.7)
    assert "does not locate or measure lesions" in report.report_text
    assert report.model_version == "test-model"


def test_report_rejects_invalid_distribution() -> None:
    with pytest.raises(ValueError):
        create_report([0.2, 0.2, 0.2])
