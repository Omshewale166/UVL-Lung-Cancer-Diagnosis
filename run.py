"""Prepare the Python environment and launch the model-backed HTTP application.

The application currently loads its TensorFlow model inside the FastAPI server's
startup lifecycle. Consequently, the launcher manages one server process and
waits for ``/health`` to report ``model_ready: true`` before declaring success.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import venv
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
DEPENDENCY_INSTALLER = PROJECT_ROOT / "install_dependencies.py"
LOGGER = logging.getLogger("lung_cancer.launcher")
_shutdown_requested = False


class LauncherError(RuntimeError):
    """An actionable startup or configuration error."""


class StartupInterrupted(LauncherError):
    """Raised when the user requests shutdown during startup."""


def _positive_float(value: str) -> float:
    """Parse a positive floating-point value for launcher options."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for launcher options."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _environment_float(name: str, default: float) -> float:
    """Read and validate a positive floating-point launcher setting."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        message = f"{name} must be a positive number; received {raw_value!r}."
        raise LauncherError(message) from exc
    if parsed <= 0:
        message = f"{name} must be greater than zero; received {raw_value!r}."
        raise LauncherError(message)
    return parsed


def _environment_int(name: str, default: int) -> int:
    """Read and validate a non-negative integer launcher setting."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        message = f"{name} must be a non-negative integer; received {raw_value!r}."
        raise LauncherError(message) from exc
    if parsed < 0:
        raise LauncherError(f"{name} must be zero or greater; received {raw_value!r}.")
    return parsed


def _parser() -> argparse.ArgumentParser:
    """Build the launcher CLI without importing project dependencies."""
    parser = argparse.ArgumentParser(
        description="Prepare dependencies, start the API, and wait for model readiness."
    )
    parser.add_argument(
        "--startup-timeout",
        type=_positive_float,
        default=_environment_float("LUNG_STARTUP_TIMEOUT_SECONDS", 180.0),
        metavar="SECONDS",
        help="Maximum time to wait for model readiness per attempt (default: 180).",
    )
    parser.add_argument(
        "--startup-retries",
        type=_nonnegative_int,
        default=_environment_int("LUNG_STARTUP_RETRIES", 1),
        metavar="COUNT",
        help="Additional server startup attempts after a failure (default: 1).",
    )
    parser.add_argument(
        "--retry-interval",
        type=_positive_float,
        default=_environment_float("LUNG_STARTUP_RETRY_INTERVAL_SECONDS", 3.0),
        metavar="SECONDS",
        help="Delay between startup attempts (default: 3).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Verify dependencies but do not install missing packages.",
    )
    return parser


def _python_is_supported(version: Sequence[int]) -> bool:
    """Return whether a Python major/minor version matches setup.py."""
    return tuple(version[:2]) >= (3, 10) and tuple(version[:2]) < (3, 13)


def _python_executable(environment: Path) -> Path:
    """Return the platform-specific Python executable inside a virtualenv."""
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return environment / relative


def _interpreter_version(executable: Path) -> tuple[int, int] | None:
    """Read an interpreter's version without importing the application."""
    try:
        result = subprocess.run(
            [
                str(executable),
                "-c",
                "import sys; print(sys.version_info.major, sys.version_info.minor)",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor = (int(part) for part in result.stdout.strip().split())
    except (TypeError, ValueError):
        return None
    return major, minor


def _find_existing_environment() -> Path | None:
    """Prefer the active virtualenv, then discover nearby project environments."""
    current = Path(sys.executable).resolve()
    if sys.prefix != sys.base_prefix and _python_is_supported(sys.version_info):
        return current

    candidates = [parent / ".venv" for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents)]
    seen: set[Path] = set()
    for environment in candidates:
        normalized = environment.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        executable = _python_executable(normalized)
        version = _interpreter_version(executable) if executable.is_file() else None
        if version is not None and _python_is_supported(version):
            return executable
    return None


def _select_environment() -> Path:
    """Select or create a supported virtual environment for this project."""
    existing = _find_existing_environment()
    if existing is not None:
        return existing

    if not _python_is_supported(sys.version_info):
        raise LauncherError(
            "This project requires Python >=3.10,<3.13. "
            f"The current interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}, "
            "and no compatible nearby virtual environment was found."
        )

    environment = PROJECT_ROOT / ".venv"
    LOGGER.info("Creating project virtual environment at %s", environment)
    try:
        venv.EnvBuilder(
            with_pip=True,
            clear=False,
            upgrade_deps=False,
        ).create(environment)
    except (OSError, subprocess.CalledProcessError) as exc:
        message = f"Could not create virtual environment at {environment}: {exc}"
        raise LauncherError(message) from exc

    executable = _python_executable(environment)
    version = _interpreter_version(executable)
    if version is None or not _python_is_supported(version):
        raise LauncherError(
            "The virtual environment at "
            f"{environment} does not contain supported Python."
        )
    return executable


def _relaunch_in_environment(
    executable: Path, arguments: Sequence[str]
) -> int | None:
    """Run this launcher under the selected interpreter when one was discovered."""
    current = Path(sys.executable).resolve()
    if executable.resolve() == current:
        return None

    LOGGER.info("Using project interpreter: %s", executable)
    try:
        options: dict[str, object] = {"cwd": PROJECT_ROOT}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(
            [str(executable), str(Path(__file__).resolve()), *arguments],
            **options,  # type: ignore[arg-type]
        )
    except OSError as exc:
        raise LauncherError(f"Could not relaunch with {executable}: {exc}") from exc

    while process.poll() is None:
        if _shutdown_requested:
            _stop_process(process)
            raise StartupInterrupted(
                "Startup interrupted while selecting Python environment."
            )
        time.sleep(0.25)
    return int(process.returncode or 0)


def _ensure_pip() -> None:
    """Confirm pip exists in the selected environment, enabling it if necessary."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    LOGGER.warning("pip is unavailable; attempting to enable it with ensurepip.")
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        message = "Could not enable pip in the selected Python environment."
        raise LauncherError(message) from exc


def _missing_requirements() -> list[str]:
    """Return manifest entries absent from the environment or outside their spec."""
    if not REQUIREMENTS_FILE.is_file():
        raise LauncherError(f"Dependency manifest not found: {REQUIREMENTS_FILE}")

    try:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.utils import canonicalize_name
    except ImportError as exc:
        message = "pip is installed but its requirement parser is unavailable."
        raise LauncherError(message) from exc

    missing: list[str] = []
    for line_number, raw_line in enumerate(
        REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        try:
            requirement = Requirement(entry)
        except Exception as exc:
            raise LauncherError(
                "Invalid requirement on line "
                f"{line_number} of {REQUIREMENTS_FILE}: {entry}"
            ) from exc
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed_version = importlib.metadata.version(
                canonicalize_name(requirement.name)
            )
        except importlib.metadata.PackageNotFoundError:
            missing.append(entry)
            continue
        if requirement.specifier and not requirement.specifier.contains(
            installed_version, prereleases=True
        ):
            missing.append(f"{entry} (installed: {installed_version})")
    return missing


def _prepare_dependencies(skip_install: bool) -> None:
    """Install project requirements when needed and validate package consistency."""
    _ensure_pip()
    missing = _missing_requirements()
    if missing:
        LOGGER.warning("Required Python packages are missing or incompatible:")
        for requirement in missing:
            LOGGER.warning("  %s", requirement)
        if skip_install:
            raise LauncherError(
                "Dependencies are missing and --skip-install was selected. "
                "Run `python install_dependencies.py` in the chosen environment."
            )
        if not DEPENDENCY_INSTALLER.is_file():
            raise LauncherError(
                f"Dependency installer not found: {DEPENDENCY_INSTALLER}"
            )
        LOGGER.info("Installing dependencies from requirements.txt")
        try:
            subprocess.run(
                [sys.executable, str(DEPENDENCY_INSTALLER)],
                cwd=PROJECT_ROOT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LauncherError(
                "Dependency installation failed. Review the pip output above and retry."
            ) from exc

        remaining = _missing_requirements()
        if remaining:
            raise LauncherError(
                "Some dependencies remain unavailable after installation: "
                + ", ".join(remaining)
            )
    else:
        LOGGER.info("All required project packages are available.")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise LauncherError(
            "The Python environment has conflicting package dependencies; "
            "see pip check output."
        ) from exc


def _install_signal_handlers() -> None:
    """Record termination requests so managed children can be stopped cleanly."""
    def request_shutdown(signum: int, _frame: object) -> None:
        global _shutdown_requested
        _shutdown_requested = True
        LOGGER.info(
            "Received %s; shutting down managed processes.",
            signal.Signals(signum).name,
        )

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_shutdown)


def _check_shutdown() -> None:
    """Raise when a signal requested shutdown during startup."""
    if _shutdown_requested:
        raise StartupInterrupted("Startup interrupted by a termination signal.")


def _health_url(host: str, port: int) -> str:
    """Build a health URL, translating wildcard bind addresses to loopback."""
    probe_host = host.strip()
    if probe_host in {"", "0.0.0.0", "::", "[::]", "*"}:
        probe_host = "127.0.0.1"
    if ":" in probe_host and not probe_host.startswith("["):
        probe_host = f"[{probe_host}]"
    return urllib.parse.urlunsplit(("http", f"{probe_host}:{port}", "/health", "", ""))


def _health_is_ready(url: str) -> tuple[bool, str]:
    """Check the API health endpoint and return readiness plus diagnostic text."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            if response.status != 200:
                return False, f"health endpoint returned HTTP {response.status}"
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"health endpoint returned invalid JSON: {exc}"

    if not isinstance(payload, dict):
        return False, "health endpoint returned a JSON value with the wrong shape"
    if payload.get("model_ready") is True and payload.get("status") == "ok":
        return True, "model is ready"
    return False, "server is responding, but model_ready is false"


def _start_server(host: str, port: int) -> subprocess.Popen[bytes]:
    """Start the existing CLI server as a managed subprocess."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    options: dict[str, object] = {
        "cwd": PROJECT_ROOT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True

    LOGGER.info(
        "Starting API and loading the model: %s",
        subprocess.list2cmdline(command),
    )
    try:
        return subprocess.Popen(command, **options)  # type: ignore[arg-type]
    except OSError as exc:
        raise LauncherError(f"Could not start the server process: {exc}") from exc


def _stop_process(
    process: subprocess.Popen[bytes], grace_seconds: float = 10.0
) -> None:
    """Stop the managed process group, escalating only if graceful stop times out."""
    if process.poll() is not None:
        return
    LOGGER.info("Stopping server process (PID %s).", process.pid)
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_C_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            return

    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        LOGGER.warning(
            "Server did not stop within %.1f seconds; forcing shutdown.",
            grace_seconds,
        )

    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        LOGGER.error(
            "Server process %s did not exit after forced shutdown.",
            process.pid,
        )


def _wait_until_ready(
    process: subprocess.Popen[bytes], health_url: str, timeout_seconds: float
) -> bool:
    """Poll readiness until the timeout, process exit, or a shutdown request."""
    deadline = time.monotonic() + timeout_seconds
    last_detail = "health endpoint has not responded yet"
    while time.monotonic() < deadline:
        _check_shutdown()
        return_code = process.poll()
        if return_code is not None:
            raise LauncherError(
                f"Server exited during startup with exit code {return_code}."
            )
        ready, last_detail = _health_is_ready(health_url)
        if ready:
            return True
        time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
    LOGGER.error(
        "Model did not become ready within %.1f seconds (%s).",
        timeout_seconds,
        last_detail,
    )
    return False


def _run_application(args: argparse.Namespace) -> int:
    """Start the API, wait for model readiness, and supervise it until shutdown."""
    # Imports are delayed until dependency bootstrapping has completed.
    from lung_cancer.config import get_settings

    settings = get_settings()
    host = settings.host
    port = settings.port
    model_path = settings.model_path
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise LauncherError(
            f"No trained model checkpoint was found at {model_path}. "
            "Train the model first with: "
            "python main.py train --data <dataset> --output artifacts"
        )

    url = _health_url(host, port)
    maximum_attempts = args.startup_retries + 1
    process: subprocess.Popen[bytes] | None = None
    try:
        for attempt in range(1, maximum_attempts + 1):
            _check_shutdown()
            LOGGER.info("Startup attempt %d of %d.", attempt, maximum_attempts)
            process = _start_server(host, port)
            try:
                if _wait_until_ready(process, url, args.startup_timeout):
                    break
            except StartupInterrupted:
                raise
            except LauncherError as exc:
                LOGGER.error("Startup attempt failed: %s", exc)
            _stop_process(process)
            process = None
            if attempt < maximum_attempts:
                LOGGER.info("Retrying in %.1f seconds.", args.retry_interval)
                time.sleep(args.retry_interval)
        else:
            raise LauncherError(
                "The application did not become ready after "
                f"{maximum_attempts} attempt(s)."
            )

        _check_shutdown()
        LOGGER.info("Application is ready at http://%s:%s/", host, port)
        LOGGER.info("Health: %s | API docs: http://%s:%s/docs", url, host, port)
        LOGGER.info("The model and API are running in the same server process.")
        assert process is not None
        while process.poll() is None and not _shutdown_requested:
            time.sleep(0.5)
        if _shutdown_requested:
            return 0
        return_code = process.returncode
        if return_code == 0:
            LOGGER.info("Server process exited normally.")
        else:
            LOGGER.error(
                "Server process exited unexpectedly with code %s.",
                return_code,
            )
        return int(return_code or 0)
    finally:
        if process is not None:
            _stop_process(process)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the environment, launch the application, and supervise shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        parser = _parser()
        args = parser.parse_args(argv)
        _install_signal_handlers()
        executable = _select_environment()
        child_exit_code = _relaunch_in_environment(
            executable, sys.argv[1:] if argv is None else argv
        )
        if child_exit_code is not None:
            return child_exit_code
        _check_shutdown()
        _prepare_dependencies(args.skip_install)
        return _run_application(args)
    except StartupInterrupted:
        LOGGER.info("Startup cancelled; managed processes have been stopped.")
        return 130
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; shutting down.")
        return 130
    except (LauncherError, OSError, ValueError) as exc:
        LOGGER.error("Startup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
