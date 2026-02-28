from ultralytics import YOLO

from src.exceptions import ModelLoadError
from src.logger import get_logger

logger = get_logger(__name__)


class BagTracker:
    def __init__(self, model_path: str, conf_threshold: float, classes: list, tracker_type: str):
        logger.info("Loading YOLO model from: %s", model_path)
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load YOLO model at '{model_path}': {exc}") from exc

        self.conf = conf_threshold
        self.classes = classes
        self.tracker_type = tracker_type
        logger.info(
            "BagTracker ready — conf=%.2f, classes=%s, tracker=%s",
            conf_threshold, classes, tracker_type,
        )

    def process_frame(self, frame):
        """
        Runs YOLOv8 detection + ByteTrack tracking on a single frame.
        persist=True retains tracking memory across frames.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            tracker=self.tracker_type,
            verbose=False,
        )
        return results[0]
