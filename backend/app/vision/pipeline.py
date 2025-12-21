from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from app.models.yolo_detector import YOLODetector
from app.tracking.tracker import MultiObjectTracker


@dataclass
class PipelineConfig:
    model_path: str = "yolov8n.pt"
    conf_th: float = 0.35

    # Tracker knobs (your tracker supports these)
    iou_threshold: float = 0.30
    max_age: int = 15
    min_hits: int = 2  # important: filters “unstable” tracks early


class VisionPipeline:
    """
    Encapsulates vision pipeline:
      frame -> YOLO detection -> IOU tracking -> structured output

    Output shape:
      {
        "players": [
            {"track_id": int, "bbox": [x1,y1,x2,y2], "label": "player"}
        ],
        "ball": {"track_id": int, "bbox": [...], "label": "ball"} or None
      }
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_th: float = 0.35,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 2,
    ):
        self.cfg = PipelineConfig(
            model_path=model_path,
            conf_th=conf_th,
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )

        self.detector = YOLODetector(model_path=self.cfg.model_path, conf_th=self.cfg.conf_th)

        # This matches your exact tracker signature
        self.tracker = MultiObjectTracker(
            iou_threshold=self.cfg.iou_threshold,
            max_age=self.cfg.max_age,
            min_hits=self.cfg.min_hits,
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray (OpenCV BGR image).")

        detections = self.detector.detect(frame)          # List[Detection]
        tracks = self.tracker.update(detections)          # List[dict]

        players: List[Dict[str, Any]] = []
        ball: Optional[Dict[str, Any]] = None

        # Note: your tracker might output multiple ball tracks over time.
        # We’ll just take the first ball track this frame (simple).
        for t in tracks:
            label = t["label"]
            item = {
                "track_id": int(t["track_id"]),
                "label": label,
                "bbox": [float(x) for x in t["bbox"]],
            }

            if label == "player":
                players.append(item)
            elif label == "ball" and ball is None:
                ball = item

        return {"players": players, "ball": ball}
