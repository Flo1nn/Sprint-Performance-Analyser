import cv2
from ultralytics import YOLO
import mediapipe as mp
from sprint_analyzer.pose import create_landmarker, detect_frame

current_boxes = []
current_ids = []


selected_track_id = None


def on_mouse_click(event, x, y, flags, param):
    global selected_track_id

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")

        for box, track_id in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = box

            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_track_id = track_id
                print(f"Selected tracking ID: {track_id}")
                return

        print("No runner selected")

model = YOLO("yolo11n.pt")
landmarker = create_landmarker()

capture = cv2.VideoCapture("data/Race_Mathletes.mp4")

if not capture.isOpened():
    raise SystemExit("Could not open video.")

cv2.namedWindow("Runner Tracking")
cv2.setMouseCallback("Runner Tracking", on_mouse_click)

frame_number = 0
fps = capture.get(cv2.CAP_PROP_FPS)

  
while True:
    ok, frame = capture.read()

    if not ok:
        break

    frame_number += 1 
    timestamp_ms = int(frame_number * 1000 / fps)


    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.25,
        tracker="bytetrack.yaml"
    )


    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        current_boxes = boxes.xyxy.cpu().numpy()
        current_ids = boxes.id.int().cpu().tolist()
    else:
        current_boxes = []
        current_ids = []

    display = results[0].plot()

    if selected_track_id is not None:
        for box, track_id in zip(current_boxes, current_ids):

            if track_id == selected_track_id:
                x1, y1, x2, y2 = map(int, box)

                # Draw green rectangle around selected runner
                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Crop selected runner from original frame
                runner_crop = frame[y1:y2, x1:x2]

                # Show the cropped runner in another window
                if runner_crop.size > 0:
                    cv2.imshow("Selected Runner Crop", runner_crop)

                    poses = detect_frame(
                        landmarker,
                        runner_crop,
                        timestamp_ms
                    )

                    if poses:
                        landmarks = poses[0]


                        print(f"Number of landmarks: {len(landmarks)}")
                    else:
                        print("No pose detected")

    

                    runner_rgb = cv2.cvtColor(
                        runner_crop,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=runner_rgb
                    )
                    print(f"MediaPipe image: {mp_image.width}x{mp_image.height}")
                   

                break


    cv2.imshow("Runner Tracking", display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        print("Video paused. Click a runner.")

        while True:
            pause_key = cv2.waitKey(30) & 0xFF

            if pause_key == ord(" "):
                print("Video resumed.")
                break

            elif pause_key == ord("q"):
                capture.release()
                cv2.destroyAllWindows()
                raise SystemExit

landmarker.close()
capture.release()
cv2.destroyAllWindows()