import cv2
from sprint_analyzer.tracking import find_closest_athlete , get_athlete_center
from sprint_analyzer.pose import create_landmarker, detect_frame, to_pixels
from sprint_analyzer.draw import draw_skeleton

# Store the coordinates of the user's click
click_position = None


def on_mouse_click(event, x, y, flags, param):
    global click_position

    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        print(f"Clicked at: {click_position}")

        selected_index = find_closest_athlete(athletes, click_position)
        print(f"Selected athlete index: {selected_index}")

# Open the race footage
capture = cv2.VideoCapture("data/Race_Mathletes.mp4")

if not capture.isOpened():
    raise SystemExit("Could not open video.")

# Create MediaPipe landmarker
landmarker = create_landmarker()

fps = capture.get(cv2.CAP_PROP_FPS)
target_frame = 300

# Process frames from 0 to 300
for frame_index in range(target_frame + 1):
    ok, frame = capture.read()

    if not ok:
        break

    timestamp_ms = int(frame_index / fps * 1000)
    poses = detect_frame(landmarker, frame, timestamp_ms)

capture.release()

if not ok:
    landmarker.close()
    raise SystemExit("Could not reach frame 300.")



height, width = frame.shape[:2]

athletes = []
athlete_points = []

for landmarks in poses:
    points = to_pixels(landmarks, width, height)

    center = get_athlete_center(points)

    athletes.append(center)
    athlete_points.append(points)

    left_hip_visibility = landmarks[23].visibility
    right_hip_visibility = landmarks[24].visibility

    print(f"Left hip visibility: {left_hip_visibility:.2f}")
    print(f"Right hip visibility: {right_hip_visibility:.2f}")



landmarker.close()

print(f"Detected {len(athletes)} athletes")

# Create a window and register our mouse function
cv2.namedWindow("Select Athlete")
cv2.setMouseCallback("Select Athlete", on_mouse_click)



while True:
    display = frame.copy()

    for points in athlete_points:
        draw_skeleton(display, points)

    # Draw every detected athlete's centre
    for index, center in enumerate(athletes):
        center_x = int(center[0])
        center_y = int(center[1])

        cv2.circle(
            display,
            (center_x, center_y),
            10,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            display,
            f"Athlete {index}",
            (center_x + 15, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    if click_position is not None:
        cv2.circle(display, click_position, 8, (0, 255, 0), -1)

    cv2.imshow("Select Athlete", display)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()