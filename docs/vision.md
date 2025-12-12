# FotStrike AR — Vision

## Problem
Watching matches is mostly passive: you *see the ball*, but you don’t clearly see spacing, off‑ball runs, pressure lines,
or who is where at a glance — unless you pause, rewind, or rely on commentators.

## Solution
Build an AR-style analytics lens that overlays real-time information on top of video:
player IDs, ball location, simple tactical cues (spacing, pressure zones), and lightweight stats.

## MVP (Dec 12 → Jan 12): Web Lens with Overlays
- Load a football match video and render it in a web viewer
- Detect players (and ball if possible) on frames
- Track detections across frames to keep stable IDs
- Draw overlays: bounding boxes + IDs + basic metrics (speed estimate, team color cluster later)
- Export overlayed frames or a rendered overlay video

## Later (Post‑MVP)
- Better ball detection + occlusion handling
- Team classification (kits/colors)
- Jersey number OCR / name mapping via roster
- Advanced metrics: xG hints, pressure intensity, passing lanes
- Mobile AR prototype (live camera)

## Example “AI prompt” you can reuse later
> Create a Python project structure as described for FotStrike AR,
> including empty placeholders and templates for README and docs/vision.md.
