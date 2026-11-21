"""Project paths and tuning constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_VIDEO = DATA_DIR / "file.mp4"
DEFAULT_OUTPUT = OUTPUT_DIR / "out.mp4"

# Ground contact is measured in frames, so low frame rates make the
# timing meaningless. 120 fps gives ~8ms resolution.
MIN_FPS_FOR_TIMING = 120
