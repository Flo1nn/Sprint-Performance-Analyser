
import cv2
import csv
import sys
from pathlib import Path
from statistics import mean
from ultralytics import YOLO

from sprint_analyzer.pose import (
    create_landmarker,
    detect_frame,
    to_pixels,
    landmarks_visible
)

from sprint_analyzer.draw import draw_skeleton, draw_angle
from sprint_analyzer.angles import calculate_angle


# Configuration
VIDEO_PATH = (
    sys.argv[1] if len(sys.argv) > 1
    else "data/Race_Mathletes.mp4"
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "sprint_measurements.csv"
VIDEO_OUTPUT = OUTPUT_DIR / "analysed_sprint.mp4"
REPORT_PATH = OUTPUT_DIR / "performance_summary.txt"

current_boxes = []
current_ids = []
selected_track_id = None
measurements = []

# Joint landmark indices
JOINTS = {
    "left_knee": [23, 25, 27],
    "right_knee": [24, 26, 28],
    "left_elbow": [11, 13, 15],
    "right_elbow": [12, 14, 16]
}


# Select athlete
def on_mouse_click(event, x, y, flags, param):
    global selected_track_id

    if event == cv2.EVENT_LBUTTONDOWN:

        for box, track_id in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = box

            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_track_id = track_id
                print(f"Selected athlete: {track_id}")
                return

        print("No athlete selected")


# Calculate joint angles
def analyse_joints(landmarks, points, frame):

    angles = {}

    for joint_name, indices in JOINTS.items():

        angles[joint_name] = None

        if landmarks_visible(landmarks, indices):

            a, b, c = indices

            angle = calculate_angle(
                points[a],
                points[b],
                points[c]
            )

            if angle is not None:

                angles[joint_name] = round(angle, 2)

                draw_angle(frame, angle, points[b])

    return angles


# Save measurements to CSV
def save_measurements():

    columns = [
        "frame",
        "time_seconds",
        "athlete_id",
        *JOINTS.keys()
    ]

    with open(CSV_PATH, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(measurements)

    print(f"Measurements saved: {CSV_PATH}")


# Generate performance summary
def generate_summary():

    lines = [
        "SPRINT PERFORMANCE ANALYSIS",
        "=" * 35,
        f"Video: {VIDEO_PATH}",
        f"Analysed frames: {len(measurements)}",
        ""
    ]

    athlete_ids = sorted({
        row["athlete_id"]
        for row in measurements
    })

    for athlete_id in athlete_ids:

        lines.append(f"Athlete ID: {athlete_id}")
        lines.append("-" * 35)

        athlete_data = [
            row for row in measurements
            if row["athlete_id"] == athlete_id
        ]

        for joint in JOINTS:

            values = [
                row[joint]
                for row in athlete_data
                if row[joint] is not None
            ]

            lines.append(
                joint.replace("_", " ").title()
            )

            if values:

                lines.append(
                    f"Average: {mean(values):.2f} degrees"
                )

                lines.append(
                    f"Minimum: {min(values):.2f} degrees"
                )

                lines.append(
                    f"Maximum: {max(values):.2f} degrees"
                )

                lines.append(
                    f"Range of motion: "
                    f"{max(values) - min(values):.2f} degrees"
                )

                lines.append(
                    f"Valid measurements: "
                    f"{len(values)}/{len(athlete_data)}"
                )

            else:
                lines.append("Insufficient valid data")

            lines.append("")

    if not athlete_ids:
        lines.append("No athlete measurements collected.")

    lines.append(
        "Note: These are estimated 2D joint angles. "
        "Camera position and landmark visibility "
        "may affect accuracy."
    )

    summary = "\n".join(lines)

    REPORT_PATH.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"\nReport saved: {REPORT_PATH}")


# Initialise models
model = YOLO("yolo11n.pt")
landmarker = create_landmarker()

capture = cv2.VideoCapture(VIDEO_PATH)

if not capture.isOpened():
    landmarker.close()
    capture.release()
    raise SystemExit("Could not open video.")

fps = capture.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

video_width = int(
    capture.get(cv2.CAP_PROP_FRAME_WIDTH)
)

video_height = int(
    capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# Initialise video export
writer = cv2.VideoWriter(
    str(VIDEO_OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (video_width, video_height)
)

if not writer.isOpened():
    capture.release()
    landmarker.close()
    writer.release()
    raise SystemExit("Could not create output video.")


# Initialise display
cv2.namedWindow("Runner Tracking")

cv2.setMouseCallback(
    "Runner Tracking",
    on_mouse_click
)

frame_number = 0
running = True

print("SPACE: Pause / Resume")
print("LEFT CLICK: Select athlete")
print("Q: Quit and save results")


try:

    while running:

        ok, frame = capture.read()

        if not ok:
            break

        frame_number += 1

        time_seconds = (frame_number - 1) / fps
        timestamp_ms = int(time_seconds * 1000)

        # Detect and track athletes
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.25,
            tracker="bytetrack.yaml",
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:

            current_boxes = boxes.xyxy.cpu().numpy()
            current_ids = boxes.id.int().cpu().tolist()

        else:

            current_boxes = []
            current_ids = []

        display = frame.copy()

        # Draw athlete bounding boxes
        for box, track_id in zip(
            current_boxes,
            current_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            selected = track_id == selected_track_id

            colour = (
                (0, 255, 0) if selected
                else (255, 255, 255)
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                colour,
                2
            )

            cv2.putText(
                display,
                f"ID: {track_id}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2
            )

        # Analyse selected athlete
        if selected_track_id is not None:

            for box, track_id in zip(
                current_boxes,
                current_ids
            ):

                if track_id != selected_track_id:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Keep crop inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(video_width, x2)
                y2 = min(video_height, y2)

                runner_crop = frame[
                    y1:y2, x1:x2
                ].copy()

                if runner_crop.size == 0:
                    break

                poses = detect_frame(
                    landmarker,
                    runner_crop,
                    timestamp_ms
                )

                angles = {
                    joint: None for joint in JOINTS
                }

                if poses:

                    landmarks = poses[0]

                    height, width = runner_crop.shape[:2]

                    points = to_pixels(
                        landmarks,
                        width,
                        height
                    )

                    # Draw skeleton
                    draw_skeleton(
                        runner_crop,
                        points
                    )

                    # Calculate and draw joint angles
                    angles = analyse_joints(
                        landmarks,
                        points,
                        runner_crop
                    )

                # Store measurements
                measurements.append({
                    "frame": frame_number,
                    "time_seconds": round(
                        time_seconds, 3
                    ),
                    "athlete_id": track_id,
                    **angles
                })

                # Add analysed crop to full video
                display[y1:y2, x1:x2] = runner_crop

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Display selected athlete
                cv2.imshow(
                    "Selected Runner Crop",
                    runner_crop
                )

                break

        # Display and export frame
        cv2.imshow("Runner Tracking", display)
        writer.write(display)

        # Keyboard controls
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            running = False

        elif key == ord(" "):

            print("Paused. Click an athlete.")

            while True:

                pause_key = cv2.waitKey(30) & 0xFF

                if pause_key == ord(" "):
                    print("Video resumed.")
                    break

                elif pause_key == ord("q"):
                    running = False
                    break

finally:

    # Release resources
    landmarker.close()
    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save final results
    save_measurements()
    generate_summary()

    print(f"Analysed video saved: {VIDEO_OUTPUT}")
    print("Analysis complete.")
