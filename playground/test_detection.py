import cv2
from ultralytics import YOLO
from sprint_analyzer.tracking import point_inside_box

# Load a pretrained object detection model
model = YOLO("yolo11n.pt")

# Open our race footage
capture = cv2.VideoCapture("data/Race_Mathletes.mp4")

if not capture.isOpened():
    raise SystemExit("Could not open video.")

# Jump to frame 300
capture.set(cv2.CAP_PROP_POS_FRAMES, 300)

ok, frame = capture.read()
capture.release()

if not ok:
    raise SystemExit("Could not read frame 300.")

# Detect people in the frame
results = model(frame, classes=[0], conf=0.25)

# Draw detected bounding boxes
annotated_frame = results[0].plot()
boxes = results[0].boxes.xyxy.cpu().numpy()

print(f"Detected {len(results[0].boxes)} people")

selected_index = None

def on_mouse_click(event, x, y, flags, param):
    global selected_index

    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)

        selected_index = None

        for index, box in enumerate(boxes):
            if point_inside_box(click_position, box):
                selected_index = index
                print(f"Selected runner: {index}")
                return

        print("No runner selected")



cv2.namedWindow("YOLO Person Detection")

cv2.setMouseCallback("YOLO Person Detection", on_mouse_click)

while True:
    display = annotated_frame.copy()

    if selected_index is not None:
        x1, y1, x2, y2 = map(int, boxes[selected_index])

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    cv2.imshow("YOLO Person Detection", display)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()