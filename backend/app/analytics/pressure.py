from __future__ import annotations

from typing import Dict, List, Tuple, Optional


def _center_from_player_dict(p: dict) -> Tuple[float, float]:
    """
    Resolve a player's (cx, cy) in *pixel coordinates* from a tracked player dict.

    Supported shapes:
      - {"pos": (cx, cy)} or {"pos": [cx, cy]}
      - {"center": (cx, cy)} or {"center": [cx, cy]}
      - {"bbox": [x1,y1,x2,y2]} (xyxy)
    """
    if "pos" in p and p["pos"] is not None:
        cx, cy = p["pos"]
        return float(cx), float(cy)
    if "center" in p and p["center"] is not None:
        cx, cy = p["center"]
        return float(cx), float(cy)

    bbox = p.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError("Player dict must contain 'pos'/'center' or a 4-element 'bbox' (xyxy).")
    x1, y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def _team_from_player_dict(p: dict) -> Optional[int]:
    """
    Resolve team id from common keys.
    Returns None if team is unknown.
    """
    team = p.get("team")
    if team is None:
        team = p.get("team_id")
    if team is None:
        team = p.get("team_idx")
    if team is None:
        return None
    return int(team)


def compute_pressure(players: List[dict], radius: float) -> Dict[int, int]:
    """
    Return mapping track_id -> count of opponents within `radius` (pixels).

    Required keys per player:
      - track_id (int)
      - bbox (xyxy) OR pos/center (cx,cy)

    Team logic:
      - If *both* players have team ids, only count those with different team.
      - If team id is missing for either player, fall back to counting any nearby player
        (cannot distinguish opponents vs teammates).

    This makes the feature usable even before team-classification is wired in.
    """
    if radius <= 0:
        return {int(p.get("track_id", -1)): 0 for p in players if "track_id" in p}

    r2 = float(radius) * float(radius)

    pts: List[Tuple[int, Optional[int], float, float]] = []  # (track_id, team, cx, cy)
    for p in players:
        if "track_id" not in p:
            continue
        tid = int(p["track_id"])
        team = _team_from_player_dict(p)
        cx, cy = _center_from_player_dict(p)
        pts.append((tid, team, cx, cy))

    out: Dict[int, int] = {tid: 0 for tid, _, _, _ in pts}

    # O(n^2) is fine for typical football player counts (<= 22)
    for i in range(len(pts)):
        tid_i, team_i, xi, yi = pts[i]
        count = 0
        for j in range(len(pts)):
            if i == j:
                continue
            _, team_j, xj, yj = pts[j]

            # If we know both teams, only count opponents
            if team_i is not None and team_j is not None and team_i == team_j:
                continue

            dx = xj - xi
            dy = yj - yi
            if (dx * dx + dy * dy) <= r2:
                count += 1
        out[tid_i] = count

    return out
