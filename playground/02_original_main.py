"""Render a pose skeleton overlay onto a sprint video.

Usage:
    python main.py [input_video] [output_video]
"""

import sys
from pathlib import Path
from sprint_analyzer.angles import calculate_angle


from sprint_analyzer.config import DEFAULT_OUTPUT, DEFAULT_VIDEO
from sprint_analyzer.draw import draw_skeleton, draw_angle
from sprint_analyzer.pose import create_landmarker, detect_frame, to_pixels, landmarks_visible
from sprint_analyzer.video import create_writer, open_video


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    capture, fps, width, height = open_video(video_path)
    print(f"fps: {fps}, size: {width}x{height}")

    writer = create_writer(output_path, fps, width, height)
    landmarker = create_landmarker()

    frame_idx = 0
    missed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        timestamp_ms = int(frame_idx / fps * 1000)
        # landmarks = detect_frame(landmarker, frame, timestamp_ms)

        # if landmarks:
        #     points = to_pixels(landmarks, width, height)

        #     draw_skeleton(frame , points)

        #     #Left Knee
        #     left_knee_angle = None 

        #     if landmarks_visible(landmarks, [23,25,27]):
        #         left_knee_angle = calculate_angle(
        #             points[23],points[25],points[27]
        #         )
            

        #     #Right Knee

        #     right_knee_angle = None

        #     if landmarks_visible(landmarks, [23,25,27]):
        #         right_knee_angle = calculate_angle(
        #             points[24],points[26],points[28]
        #         )

        #     #Draw Knee Angles on the video
        #     draw_angle(frame, left_knee_angle, points[25])
        #     draw_angle(frame, right_knee_angle, points[26])
        
        # else:
        #     missed += 1
        
        #Detect multiple athletes
        poses = detect_frame(landmarker, frame, timestamp_ms)

        if poses:
            for landmarks in poses:
                points = to_pixels(landmarks, width, height)
                draw_skeleton(frame, points)
        else:
            missed += 1

        

        #save every processed frame
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()
    print(f"Done. {frame_idx} frames, {missed} with no pose detected.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
