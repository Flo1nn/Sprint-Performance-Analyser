# Sprint Performance Analyser

A computer vision tool that analyses a sprinter's form from video using athlete tracking, pose estimation and joint angle calculations.

## Why

Sprinters often have to rely on their own judgement when analysing training and race performance when their coaches are unavailable.

As a competitive sprinter, I wanted to develop a tool that combines athletics and artificial intelligence to help athletes better understand their sprint mechanics.

## Status

Version 1 completed.

**Working now:**

- Multi-athlete detection and tracking.
- Interactive athlete selection.
- Full-body skeleton overlay.
- Left and right knee and elbow angle calculations.
- Annotated video export.
- Frame-by-frame CSV measurements.
- Automated performance summary.

## Tech

- Python 3.13 — application development
- YOLO11 — person detection
- ByteTrack — athlete tracking
- MediaPipe Pose — pose estimation
- OpenCV — video processing and drawing
- NumPy — joint angle calculations

## Setup

```bash
git clone https://github.com/Flo1nn/Sprint-Performance-Analyser.git
cd Sprint-Performance-Analyser

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```text
main.py              Main application
sprint_analyser/
    config.py        Paths and configuration
    video.py         Video processing
    pose.py          Landmark detection
    draw.py          Skeleton and angle rendering
    angles.py        Joint angle calculations
    tracking.py      Athlete tracking
data/                Input videos (gitignored)
output/              Generated results (gitignored)
models/              MediaPipe pose model
playground/          Development and testing scripts
requirements.txt     Project dependencies
```

## Usage

1. Place a sprint video inside `data/`.
2. Activate the virtual environment.
3. Run:

```bash
python main.py data/my_sprint.mp4
```

4. Press Space to pause the video.
5. Click on the athlete you want to analyse.
6. Press Space to resume.
7. Press Q to quit and save the results.

## Output

The following files are generated inside `output/`:

- `analysed_sprint.mp4` — annotated sprint video.
- `sprint_measurements.csv` — recorded joint angles.
- `performance_summary.txt` — average, minimum, maximum and range of joint angles.

## Notes

- Always activate the virtual environment before running the application.
- Side-view footage is recommended for more accurate joint angle measurements.
- Joint angles are estimated in 2D and may be affected by camera position, motion blur and athlete visibility.
- Measurements are intended for exploratory performance analysis.

## Future Improvements

- Improved tracking accuracy.
- Stride length and frequency analysis.
- Ground contact time estimation.
- Automated sprint technique feedback.


