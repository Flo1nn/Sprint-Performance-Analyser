import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "models/pose_landmarker_full.task"


capture = cv2.VideoCapture("file2.mp4")



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
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0,255,0), -1)

    else:
        missed += 1
        
    writer.write(frame)
    frame_idx += 1


capture.release()
writer.release()
print(f"Done. {frame_idx} frame, {missed} with no pose detected.")