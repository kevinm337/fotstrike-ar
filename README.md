# FotStrike AR — Real‑Time Football Analytics Lens

FotStrike AR is a real‑time football analytics “lens” that overlays key information on match video (and later live camera/AR),
so viewers can _see_ spacing, pressure, and player context instead of watching passively.

## What you’ll build in the MVP (high level)

- Load a match video and preview frames
- Detect players/ball (YOLO)
- Track objects across frames
- Compute simple overlays (IDs, speed, zones, heat hints)
- Render overlays in a lightweight web viewer

## Repo map

- `backend/` — Python services + utilities (vision, tracking, APIs later)
- `notebooks/` — R&D notebooks (frame extraction, YOLO tests, tracking experiments)
- `frontend/` — Web lens viewer (later)
- `docs/` — Vision, notes, screenshots, diagrams
- `data/` — Local dev data (videos, rosters). Keep large videos out of git.

## Day 1 deliverables

- Repo structure created
- `docs/vision.md` written (problem → solution → MVP features → future)

## Quickstart (Day 2+)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- Put a test match video at: `data/matches/match1.mp4`
- Save sample frames here: `docs/images/`
