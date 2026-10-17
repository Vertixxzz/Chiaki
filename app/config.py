from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

@dataclass(frozen=True)
class Settings:
    window_size: float = 10.0
    window_step: float = 5.0
    frame_sample_fps: float = 2.0
    threshold: float = 0.5
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

SETTINGS = Settings()
