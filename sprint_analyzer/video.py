"""Video input/output and validation."""

import cv2

from .config import MIN_FPS_FOR_TIMING


def open_video(path):
    """Open a video file and return (capture, fps, width, height).

    Raises SystemExit with a readable message if the file can't be used.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {path}. Check the file path.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise SystemExit(f"Could not read frame rate from {path}.")

    if fps < MIN_FPS_FOR_TIMING:
        print(
            f"Warning: {fps:.0f} fps. Ground contact timing needs "
            f"{MIN_FPS_FOR_TIMING}+ fps to be meaningful."
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, fps, width, height


def create_writer(path, fps, width, height):
    """Create an mp4 writer matching the source video's dimensions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))
