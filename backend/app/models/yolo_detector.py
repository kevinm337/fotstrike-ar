from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from app.schemas.detection import Detection


@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"

    # separate thresholds (ball needs lower threshold usually)
    conf_player: float = 0.35
    conf_ball: float = 0.05

    # image size for YOLO inference (bigger helps small ball)
    imgsz: int = 960

    keep_classes: Tuple[str, ...] = ("person", "sports ball")  # COCO names


class YOLODetector:
    """
    Reusable YOLO detector.

    Input:
      - frame: np.ndarray (OpenCV BGR image)

    Output:
      - List[Detection]
        Detection(label="player" or "ball", confidence=float, bbox=[x1,y1,x2,y2])
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_player: float = 0.35,
        conf_ball: float = 0.05,
        imgsz: int = 960,
    ):
        self.cfg = DetectorConfig(
            model_path=model_path,
            conf_player=float(conf_player),
            conf_ball=float(conf_ball),
            imgsz=int(imgsz),
        )
        self.model = YOLO(self.cfg.model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray (OpenCV image).")

        # OpenCV gives BGR; convert to RGB for YOLO
        frame_rgb = frame[:, :, ::-1]

        # Run YOLO
        results = self.model(frame_rgb, verbose=False, imgsz=self.cfg.imgsz)
        r = results[0]

        names = r.names
        out: List[Detection] = []

        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = names[cls_id]
            conf = float(b.conf[0])

            if cls_name not in self.cfg.keep_classes:
                continue

            # label mapping + per-class threshold
            if cls_name == "person":
                label = "player"
                if conf < self.cfg.conf_player:
                    continue
            else:
                label = "ball"
                if conf < self.cfg.conf_ball:
                    continue

            x1, y1, x2, y2 = map(float, b.xyxy[0])

            out.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                )
            )

        return out
