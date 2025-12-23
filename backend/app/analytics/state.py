from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.analytics.heatmap import HeatmapGrid
from app.analytics.pressure import compute_pressure


def bbox_center_xyxy(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class PlayerState:
    track_id: int
    team_id: Optional[int] = None
    opponents_in_radius: int = 0
    under_pressure: bool = False
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
        pressure_radius_px: float = 60.0,
        pressure_threshold: int = 2,
    ):
        self.fps = float(fps)
        self.pixels_to_meters = float(pixels_to_meters)
        self.history_len = int(history_len)

        self.players: Dict[int, PlayerState] = {}
        self.frame_index = 0

        # pressure settings (Day 17)
        self.pressure_radius_px = float(pressure_radius_px)
        self.pressure_threshold = int(pressure_threshold)

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

        Optionally may include team info:
        {"team": int} or {"team_id": int}
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

            # keep team if upstream provides it
            if "team" in p and p["team"] is not None:
                st.team_id = int(p["team"])
            elif "team_id" in p and p["team_id"] is not None:
                st.team_id = int(p["team_id"])

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

        # pressure (Day 17): opponents within radius => under_pressure
        pressure_map = compute_pressure(tracked_players, radius=self.pressure_radius_px)
        for p in tracked_players:
            if p.get("label") != "player":
                continue
            tid = int(p["track_id"])
            st = self.players.get(tid)
            if st is None:
                continue

            # If team is present in the tracked dict, store it
            if "team" in p and p["team"] is not None:
                st.team_id = int(p["team"])
            elif "team_id" in p and p["team_id"] is not None:
                st.team_id = int(p["team_id"])

            st.opponents_in_radius = int(pressure_map.get(tid, 0))
            st.under_pressure = st.opponents_in_radius >= self.pressure_threshold

        self.frame_index += 1
