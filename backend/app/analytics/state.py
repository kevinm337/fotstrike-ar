from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.analytics.heatmap import HeatmapGrid


def bbox_center_xyxy(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class PlayerState:
    track_id: int
    last_position: Optional[Tuple[float, float]] = None  # (cx, cy) in px
    total_distance_px: float = 0.0
    total_distance: float = 0.0  # meters
    current_speed: float = 0.0   # m/s
    positions: List[Tuple[float, float]] = field(default_factory=list)


class MatchState:
    def __init__(
        self,
        fps: float,
        pixels_to_meters: float = 0.01,
        history_len: int = 200,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
        heatmap_grid_x: int = 60,
        heatmap_grid_y: int = 40,
    ):
        self.fps = float(fps)
        self.pixels_to_meters = float(pixels_to_meters)
        self.history_len = int(history_len)

        self.players: Dict[int, PlayerState] = {}
        self.frame_index = 0

        self.heatmap: Optional[HeatmapGrid] = None
        if video_width is not None and video_height is not None:
            self.heatmap = HeatmapGrid(
                width=int(video_width),
                height=int(video_height),
                grid_x=int(heatmap_grid_x),
                grid_y=int(heatmap_grid_y),
            )

    def update(self, tracked_players: List[dict]) -> None:
        """
        tracked_players items look like:
        {"track_id": int, "bbox": [x1,y1,x2,y2], "label": "player"}
        """
        for p in tracked_players:
            if p.get("label") != "player":
                continue

            tid = int(p["track_id"])
            bbox = [float(x) for x in p["bbox"]]
            cx, cy = bbox_center_xyxy(bbox)

            if tid not in self.players:
                self.players[tid] = PlayerState(track_id=tid)

            st = self.players[tid]

            # heatmap accumulation (Day 16)
            if self.heatmap is not None:
                self.heatmap.add_position(cx, cy)

            # speed + distance (Day 14)
            if st.last_position is not None:
                px_dist = ((cx - st.last_position[0]) ** 2 + (cy - st.last_position[1]) ** 2) ** 0.5
                st.total_distance_px += px_dist

                meters = px_dist * self.pixels_to_meters
                st.total_distance += meters

                # speed approx using fps
                st.current_speed = meters * self.fps

            st.last_position = (cx, cy)

            st.positions.append((cx, cy))
            if len(st.positions) > self.history_len:
                st.positions = st.positions[-self.history_len :]

        self.frame_index += 1
