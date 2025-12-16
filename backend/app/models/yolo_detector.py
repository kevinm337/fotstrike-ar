from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    conf_th: float = 0.35
    keep_classes: tuple[str, ...] = ("person", "sports ball")  # COCO names


class YOLODetector:
    """
    Reusable YOLO detector.

    Input:
      - frame: np.ndarray (OpenCV BGR image)

    Output:
      - list of dicts:
        {
          "label": "player" or "ball",
          "confidence": float,
          "bbox": [x1, y1, x2, y2]
        }
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_th: float = 0.35):
        self.cfg = DetectorConfig(model_path=model_path, conf_th=conf_th)
        self.model = YOLO(self.cfg.model_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray (OpenCV image).")

        # Ultralytics can accept numpy images directly.
        # OpenCV images are BGR; Ultralytics usually handles it, but we can be explicit:
        frame_rgb = frame[:, :, ::-1]  # BGR -> RGB

        results = self.model(frame_rgb)
        r = results[0]

        names = r.names  # class_id -> class_name
        out: List[Dict[str, Any]] = []

        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = names[cls_id]
            conf = float(b.conf[0])

            if conf < self.cfg.conf_th:
                continue
            if cls_name not in self.cfg.keep_classes:
                continue

            x1, y1, x2, y2 = map(float, b.xyxy[0])  # pixel coords

            # Map to your app labels
            label = "player" if cls_name == "person" else "ball"

            out.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return out