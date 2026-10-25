"""Package metadata for the lung cancer image classification project."""

from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent


def read_requirements() -> list[str]:
    """Read runtime dependencies from the canonical requirements file."""
    requirements_path = PROJECT_ROOT / "requirements.txt"
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="lung-cancer-diagnosis",
    version="1.0.0",
    description="Research-faithful CT slice classification with a custom CNN",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    include_package_data=True,
    package_data={"lung_cancer": ["static/*.html"]},
    install_requires=read_requirements(),
    entry_points={"console_scripts": ["lung-cancer=lung_cancer.cli:main"]},
)
