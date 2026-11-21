"""Skeleton rendering onto video frames."""

import cv2

# MediaPipe Pose landmark indices, grouped by body part.
POSE_CONNECTIONS = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

BONE_COLOUR = (255, 255, 255)
JOINT_COLOUR = (0, 255, 0)


def draw_skeleton(frame, points):
    """Draw bones and joints onto the frame in place."""
    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], BONE_COLOUR, 2)

    for point in points:
        cv2.circle(frame, point, 4, JOINT_COLOUR, -1)

    return frame
