import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "models/pose_landmarker_full.task"
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

def main():
    capture = cv2.VideoCapture("file3.mp4")



    fps = capture.get(cv2.CAP_PROP_FPS)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"fps: {fps}, size: {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("out.mp4", fourcc, fps, (w,h))



    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.VIDEO)


    landmarker =  vision.PoseLandmarker.create_from_options(options)

    frame_idx = 0
    missed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) 

        timestamp_ms = int(frame_idx / fps * 1000)        
        result = landmarker.detect_for_video(mp_Image, timestamp_ms)


        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0] # first person found
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for start_idx, end_idx in POSE_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (255,255,255), 2)
            
            for (x, y) in points:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        else:
            missed += 1
            
        writer.write(frame)
        frame_idx += 1


    capture.release()
    writer.release()
    print(f"Done. {frame_idx} frame, {missed} with no pose detected.")

if __name__ == "__main__":
    main()


    