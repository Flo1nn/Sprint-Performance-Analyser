"""MediaPipe pose detection."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import MODEL_PATH


def create_landmarker():
    """Build a PoseLandmarker configured for video (tracks across frames)."""
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Pose model not found at {MODEL_PATH}. "
            "See the README for where to get it."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=8
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_frame(landmarker, frame, timestamp_ms):
    """Detect pose in one BGR frame. Returns landmarks, or None if no pose."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.pose_landmarks:
        return []
    return result.pose_landmarks  # first person found


def to_pixels(landmarks, width, height):
    """Convert normalised landmarks to (x, y) pixel coordinates."""
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

#Additonal function for data validation -> data now has to meet a minimum quality requirement to be consider for usage
#Removes redundant errors that could mess up the estimations
def landmarks_visible(landmarks, indices, threshold=0.5):
    """Checking whethe all selected landmarks are sufficiently visible"""
    return all(
        landmarks[i].visibility >= threshold
        for i in indices 
    )