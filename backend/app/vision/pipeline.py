from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from app.models.yolo_detector import YOLODetector
from app.tracking.tracker import MultiObjectTracker


@dataclass
class PipelineConfig:
    model_path: str = "yolov8n.pt"
    conf_player: float = 0.35
    conf_ball: float = 0.05
    imgsz: int = 960

    iou_threshold: float = 0.30
    max_age: int = 15
    min_hits: int = 2


class VisionPipeline:
    """
    frame -> YOLO detection -> IOU tracking -> structured output
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_player: float = 0.35,
        conf_ball: float = 0.05,
        imgsz: int = 960,
        conf_th: Optional[float] = None,  # backward compat
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 2,
    ):
        # If user passes old conf_th, apply to both
        if conf_th is not None:
            conf_player = float(conf_th)
            conf_ball = float(conf_th)

        self.cfg = PipelineConfig(
            model_path=model_path,
            conf_player=float(conf_player),
            conf_ball=float(conf_ball),
            imgsz=int(imgsz),
            iou_threshold=float(iou_threshold),
            max_age=int(max_age),
            min_hits=int(min_hits),
        )

        self.detector = YOLODetector(
            model_path=self.cfg.model_path,
            conf_player=self.cfg.conf_player,
            conf_ball=self.cfg.conf_ball,
            imgsz=self.cfg.imgsz,
        )

        # If your current MultiObjectTracker doesn't accept these,
        # open backend/app/tracking/tracker.py and confirm it matches.
        self.tracker = MultiObjectTracker(
            iou_threshold=self.cfg.iou_threshold,
            max_age=self.cfg.max_age,
            min_hits=self.cfg.min_hits,
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray (OpenCV BGR image).")

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)

        players: List[Dict[str, Any]] = []
        ball: Optional[Dict[str, Any]] = None

        for t in tracks:
            item = {
                "track_id": int(t["track_id"]),
                "label": str(t["label"]),
                "bbox": [float(x) for x in t["bbox"]],
            }

            if item["label"] == "player":
                players.append(item)
            elif item["label"] == "ball" and ball is None:
                ball = item

        return {"players": players, "ball": ball}
