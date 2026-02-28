import cv2

from src.exceptions import VideoLoadError
from src.logger import get_logger

logger = get_logger(__name__)


class VideoLoader:
    def __init__(self, video_path: str):
        logger.info("Opening video source: %s", video_path)
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise VideoLoadError(f"Could not open video at: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # default 30 if undetectable
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Video loaded — %dx%d @ %d fps, ~%d frames",
            self.width, self.height, self.fps, self.total_frames,
        )

    def get_frame(self):
        """Returns (success: bool, frame: ndarray | None)."""
        return self.cap.read()

    def release(self):
        self.cap.release()
        logger.info("VideoLoader released.")
