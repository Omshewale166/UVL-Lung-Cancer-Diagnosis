"""Central logging configuration."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure concise timestamped logs for CLI and API processes."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
