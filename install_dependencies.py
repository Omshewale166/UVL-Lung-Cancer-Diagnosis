"""Install this project's Python dependencies from requirements.txt.

Run this script with the Python interpreter for the environment where the
project will run, preferably from an activated virtual environment:

    python install_dependencies.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def ensure_supported_python() -> None:
    """Fail early when the selected interpreter is outside the supported range."""
    version = sys.version_info
    if version < (3, 10) or version >= (3, 13):
        raise SystemExit(
            "This project supports Python 3.10, 3.11, and 3.12. "
            f"The selected interpreter is Python {version.major}.{version.minor}."
        )


def ensure_pip() -> None:
    """Use the selected interpreter's pip, bootstrapping it when necessary."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    print("pip is unavailable; attempting to enable it with ensurepip.")
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)


def install_dependencies(dry_run: bool = False) -> None:
    """Install the pinned project requirements using the active interpreter."""
    if not REQUIREMENTS_FILE.is_file():
        raise FileNotFoundError(f"Requirements file not found: {REQUIREMENTS_FILE}")

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        str(REQUIREMENTS_FILE),
    ]
    if dry_run:
        command.append("--dry-run")

    print(f"Installing dependencies from {REQUIREMENTS_FILE}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if dry_run:
        print("Dependency dry run completed; no packages were installed.")
    else:
        print("All required packages were installed successfully.")


def main() -> None:
    """Parse options and install project dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask pip to resolve dependencies without installing them.",
    )
    args = parser.parse_args()
    ensure_supported_python()
    ensure_pip()
    install_dependencies(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
