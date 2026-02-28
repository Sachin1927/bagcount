"""
Custom exception hierarchy for the BagCount pipeline.
All exceptions inherit from BagCountError so callers can catch
the entire family with a single except clause if needed.
"""


class BagCountError(Exception):
    """Base exception for all BagCount pipeline errors."""


class VideoLoadError(BagCountError):
    """Raised when a video file cannot be opened or a frame cannot be read."""


class ModelLoadError(BagCountError):
    """Raised when the YOLO model file is missing or fails to initialise."""


class ConfigError(BagCountError):
    """Raised when a configuration value is invalid or out of range."""


class OutputWriteError(BagCountError):
    """Raised when the output video writer cannot be created or written to."""
