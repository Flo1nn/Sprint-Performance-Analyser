import cv2
from pathlib import Path

cap = cv2.VideoCapture("file.mp4")


#Error checker if file is not found / cannot be opened
if not cap.isOpened():
    raise SystemExit("Could not open video. Check the file path")


fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



if fps <= 0:
    print("Warning: could not read FPS from file; timing estimates unavailable.")
else:
    print(f"fps: {fps}, frames: {total}, duration: {total/ fps:.2f}s")
    if fps < 120:
        print("Warning: low frame rate. Ground contact time will be imprecise.")

i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imwrite(f"frames/frame_{i:04d}.jpg", frame)
    print(f"frame {i} at {i/ fps:.3f}s")

    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    i += 1

cap.release()
cv2.destroyAllWindows()
