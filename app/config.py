from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

@dataclass(frozen=True)
class Settings:
    window_size: float = 10.0
    window_step: float = 5.0
    frame_sample_fps: float = float(os.getenv("FRAME_SAMPLE_FPS", "2"))
    event_safety_margin: float = 0.0
    temporal_feature_set: str = "ordered_v1"
    threshold: float = 0.5
    random_seed: int = 42
    holdout_folds: int = 5
    cv_folds: int = 4
    evidence_per_category: int = 3
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

SETTINGS = Settings()
