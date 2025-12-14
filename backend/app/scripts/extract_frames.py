import argparse
from pathlib import Path
from app.utils.video import extract_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to save")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = extract_frames(in_path, out_dir, target_fps=args.fps)
    print(f"✅ Saved {saved} frames to: {out_dir}")


if __name__ == "__main__":
    main()