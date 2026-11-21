import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "models/pose_landmarker_full.task"
image_path = "data/frame_0037.jpg"

frame = cv2.imread(image_path)

if frame is None:
    raise SystemError(f"Could not read {image_path}. Check the path.")


#Coversion for RBG numpy array -> for MediaPipe Image 
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.IMAGE)

#Create and Detect image
landmarker = vision.PoseLandmarker.create_from_options(options)
result = landmarker.detect(mp_Image)

if not result.pose_landmarks:
    print("No pose detected.")
else:
    landmarks = result.pose_landmarks[0] # first person found
    print(f"Found {len(landmarks)} landmarks")

    h, w = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0,255,0), -1)
    
    cv2.imwrite("data/frame_0037_pose.jpg", frame)
    print("Saved frame_0037_pose.jpg")
