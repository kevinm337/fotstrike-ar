from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.schemas.detection import Detection


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    label: str
    bbox: List[float]
    confidence: float
    age: int = 0
    time_since_update: int = 0
    hits: int = 1


class MultiObjectTracker:
    """
    Minimal multi-object tracker using greedy IOU matching (SORT-lite).

    Keeps tracks per label ("player" and "ball") so they don't steal IDs from each other.

    NOTE:
    - players: require min_hits for stability
    - ball: we allow min_hits=1 (ball detection is often flickery)
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 2,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)

        self._next_id = 1
        self._tracks_by_label: Dict[str, List[Track]] = {"player": [], "ball": []}

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _match_greedy(
        self, tracks: List[Track], detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        pairs = []
        for ti, t in enumerate(tracks):
            for di, d in enumerate(detections):
                pairs.append((iou_xyxy(t.bbox, d.bbox), ti, di))

        pairs.sort(reverse=True, key=lambda x: x[0])

        used_t = set()
        used_d = set()
        matches: List[Tuple[int, int]] = []

        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))

        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_t]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection]) -> List[dict]:
        # split detections by label
        dets_by_label: Dict[str, List[Detection]] = {"player": [], "ball": []}
        for d in detections:
            if d.label in dets_by_label:
                dets_by_label[d.label].append(d)

        outputs: List[dict] = []

        for label in ("player", "ball"):
            tracks = self._tracks_by_label[label]
            dets = dets_by_label[label]

            # age tracks
            for t in tracks:
                t.age += 1
                t.time_since_update += 1

            matches, _, unmatched_dets = self._match_greedy(tracks, dets)

            # update matched
            for ti, di in matches:
                t = tracks[ti]
                d = dets[di]
                t.bbox = list(d.bbox)
                t.confidence = float(d.confidence)
                t.time_since_update = 0
                t.hits += 1

            # create new tracks for unmatched detections
            for di in unmatched_dets:
                d = dets[di]
                tracks.append(
                    Track(
                        track_id=self._new_id(),
                        label=label,
                        bbox=list(d.bbox),
                        confidence=float(d.confidence),
                    )
                )

            # remove old tracks
            tracks[:] = [t for t in tracks if t.time_since_update <= self.max_age]

            # emit
            for t in tracks:
                required_hits = self.min_hits if t.label == "player" else 1  # ✅ ball emits fast
                if t.hits >= required_hits or t.time_since_update == 0:
                    outputs.append(
                        {"track_id": t.track_id, "bbox": t.bbox, "label": t.label}
                    )

        return outputs
