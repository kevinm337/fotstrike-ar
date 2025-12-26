#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

source backend/.venv/bin/activate

cd backend
python -m app.vision_demo \
  --video-path ../data/matches/match1.mp4 \
  --conf 0.35 \
  --trail-len 25 \
  --max-frames 300 \
  --save-video ../docs/videos/day12_vision_demo.mp4


