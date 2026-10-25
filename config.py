"""Compatibility entry point exposing validated application settings."""

from lung_cancer.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
