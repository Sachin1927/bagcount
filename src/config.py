import os
from pathlib import Path

from src.exceptions import ConfigError

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw", "test_video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "output.mp4")

# --- Model & Tracking Settings ---
CONFIDENCE_THRESHOLD = 0.25
# COCO dataset classes: 24 (backpack), 26 (handbag), 28 (suitcase)
TARGET_CLASSES = [24, 26, 28]
TRACKER_TYPE = "bytetrack.yaml"

# --- Counting Settings ---
# Fraction of frame height where the virtual counting line is drawn (0.0–1.0).
# 0.5 places it at the vertical centre regardless of video resolution.
# Override at runtime with --line-ratio CLI flag.
COUNTING_LINE_RATIO = 0.40


def validate():
    """Raise ConfigError early if any critical setting is obviously wrong."""
    if not (0.0 < CONFIDENCE_THRESHOLD <= 1.0):
        raise ConfigError(
            f"CONFIDENCE_THRESHOLD must be in (0, 1], got {CONFIDENCE_THRESHOLD}"
        )
    if not (0.0 < COUNTING_LINE_RATIO < 1.0):
        raise ConfigError(
            f"COUNTING_LINE_RATIO must be in (0, 1), got {COUNTING_LINE_RATIO}"
        )