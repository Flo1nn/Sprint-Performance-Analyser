"""Render a pose skeleton overlay onto a sprint video.

Usage:
    python main.py [input_video] [output_video]
"""

import sys
from pathlib import Path

from sprint_analyzer.config import DEFAULT_OUTPUT, DEFAULT_VIDEO
from sprint_analyzer.draw import draw_skeleton
from sprint_analyzer.pose import create_landmarker, detect_frame, to_pixels
from sprint_analyzer.video import create_writer, open_video


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    capture, fps, width, height = open_video(video_path)
    print(f"fps: {fps}, size: {width}x{height}")

    writer = create_writer(output_path, fps, width, height)
    landmarker = create_landmarker()

    frame_idx = 0
    missed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        timestamp_ms = int(frame_idx / fps * 1000)
        landmarks = detect_frame(landmarker, frame, timestamp_ms)

        if landmarks:
            draw_skeleton(frame, to_pixels(landmarks, width, height))
        else:
            missed += 1

        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()
    print(f"Done. {frame_idx} frames, {missed} with no pose detected.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
