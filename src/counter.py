from src.logger import get_logger

logger = get_logger(__name__)


class LineCounter:
    def __init__(self, line_y: int):
        self.line_y = line_y
        self.track_history: dict[int, float] = {}  # {track_id: previous_center_y}
        self.counted_ids: set[int] = set()          # prevents double-counting
        self.total_count = 0
        logger.info("LineCounter initialised — counting line Y=%d", line_y)

    def update_count(self, boxes, track_ids) -> int:
        """
        Update the bag count for the current frame.

        Memory-leak fix: IDs that are no longer active are pruned from
        track_history so the dict stays bounded to the number of objects
        currently visible, not the entire history of the video.
        """
        if boxes is None or track_ids is None:
            return self.total_count

        active_ids = set(int(tid) for tid in track_ids)

        # --- Prune stale entries (fix for unbounded dict growth) ---
        stale_ids = set(self.track_history.keys()) - active_ids
        for sid in stale_ids:
            del self.track_history[sid]
        if stale_ids:
            logger.debug("Pruned %d stale track IDs from history.", len(stale_ids))

        # --- Evaluate each currently-tracked object ---
        for box, track_id in zip(boxes, track_ids):
            track_id = int(track_id)
            x1, y1, x2, y2 = box
            center_y = (y1 + y2) / 2

            if track_id in self.track_history:
                prev_y = self.track_history[track_id]

                crossed_down = prev_y < self.line_y <= center_y
                crossed_up   = prev_y > self.line_y >= center_y

                if (crossed_down or crossed_up) and track_id not in self.counted_ids:
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    direction = "↓" if crossed_down else "↑"
                    logger.info(
                        "Bag counted %s  ID=%d  total=%d",
                        direction, track_id, self.total_count,
                    )

            self.track_history[track_id] = center_y

        return self.total_count
