"""
BagCount inference entry-point.

Usage examples
--------------
# Default paths from config:
    python scripts/run_inference.py

# Custom video, output, and line position:
    python scripts/run_inference.py --input data/raw/airport.mp4 --output data/processed/airport_out.mp4 --line-ratio 0.6

# Higher confidence threshold, no display window:
    python scripts/run_inference.py --conf 0.65 --no-display
"""

import argparse
import sys
import time
from pathlib import Path

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src import config
from src.data_loader import VideoLoader
from src.tracker import BagTracker
from src.counter import LineCounter
from src.exceptions import BagCountError, OutputWriteError
from src.logger import get_logger

logger = get_logger("bagcount.run_inference")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BagCount — real-time bag detection and counting pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input video file. Overrides config.VIDEO_PATH.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for the annotated output video. Overrides config.OUTPUT_PATH.",
    )
    parser.add_argument(
        "--line-ratio", type=float, default=None,
        metavar="RATIO",
        help="Counting line position as a fraction of frame height (0.0–1.0). "
             "Overrides config.COUNTING_LINE_RATIO.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Detection confidence threshold (0.0–1.0). "
             "Overrides config.CONFIDENCE_THRESHOLD.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the live preview window (useful for headless servers).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve effective settings (CLI overrides config defaults) ---
    video_path  = args.input      or config.VIDEO_PATH
    output_path = args.output     or config.OUTPUT_PATH
    line_ratio  = args.line_ratio or config.COUNTING_LINE_RATIO
    conf        = args.conf       or config.CONFIDENCE_THRESHOLD

    logger.info("=== BagCount pipeline starting ===")
    logger.info("Input  : %s", video_path)
    logger.info("Output : %s", output_path)
    logger.info("Line ratio : %.2f  |  Confidence : %.2f", line_ratio, conf)

    # Validate the ratio before doing any heavy work
    if not (0.0 < line_ratio < 1.0):
        logger.error("--line-ratio must be between 0.0 and 1.0 (exclusive), got %.2f", line_ratio)
        sys.exit(1)

    # --- Initialise modules ---
    loader  = VideoLoader(video_path)
    tracker = BagTracker(config.MODEL_PATH, conf, config.TARGET_CLASSES, config.TRACKER_TYPE)

    # Counting line Y is now computed from the actual video dimensions
    counting_line_y = int(loader.height * line_ratio)
    logger.info("Counting line placed at Y=%d (ratio=%.2f of height=%d)",
                counting_line_y, line_ratio, loader.height)

    counter = LineCounter(counting_line_y)

    # --- Video writer ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        loader.fps,
        (loader.width, loader.height),
    )
    if not writer.isOpened():
        raise OutputWriteError(f"Could not open VideoWriter for path: {output_path}")

    # --- Preview window (created once, before the loop) ---
    WINDOW_NAME = "Bag Counter V1"
    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)

    logger.info("Starting inference loop — press 'q' to quit early.")

    frame_idx   = 0
    fps_timer   = time.time()
    display_fps = 0.0

    # --- Processing loop ---
    try:
        while True:
            success, frame = loader.get_frame()
            if not success:
                logger.info("End of video stream reached.")
                break

            frame_idx += 1

            # Detection + tracking
            result    = tracker.process_frame(frame)
            boxes     = result.boxes.xyxy.cpu().numpy()      if result.boxes else None
            track_ids = result.boxes.id.int().cpu().numpy()  if (result.boxes and result.boxes.id is not None) else None

            # Update bag count
            current_count = counter.update_count(boxes, track_ids)

            # --- Annotations ---
            annotated = result.plot()

            # Virtual counting line
            cv2.line(
                annotated,
                (0, counting_line_y),
                (loader.width, counting_line_y),
                (0, 255, 0), 2,
            )

            # Bag count HUD
            cv2.putText(
                annotated, f"Bags Counted: {current_count}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
            )

            # Live FPS overlay
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_idx / elapsed
                fps_timer   = time.time()
                frame_idx   = 0
            cv2.putText(
                annotated, f"FPS: {display_fps:.1f}",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2,
            )

            # Write frame to output file
            writer.write(annotated)

            # Show preview
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed 'q' — stopping early.")
                    break

    except BagCountError:
        # Already logged by the raising module; re-raise to exit with non-zero code
        raise
    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        raise
    finally:
        loader.release()
        writer.release()
        cv2.destroyAllWindows()

    logger.info("=== Processing complete ===")
    logger.info("Total bags counted : %d", counter.total_count)
    logger.info("Output video saved : %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except BagCountError as exc:
        logger.error("Pipeline error: %s", exc)
        sys.exit(1)
