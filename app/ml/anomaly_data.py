"""Leak-free data split and score helpers for the normal-only experiment."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from app.config import OUTPUTS_DIR, SETTINGS
from app.data.dataset import VideoLabel
from app.ml.anomaly import aggregate_scores
from app.ml.features import FeatureDataset

@dataclass(frozen=True)
class AnomalySplit:
    normal_train_ids: list[str]
    normal_validation_ids: list[str]
    test_ids: list[str]

def make_split(videos: list[VideoLabel], fixed_test_ids: set[str]) -> AnomalySplit:
    """Keep legacy held-out videos fixed; split only remaining normal videos."""
    normal_candidates = sorted(v.video_id for v in videos if not v.violation and v.video_id not in fixed_test_ids)
    train_ids, validation_ids = train_test_split(normal_candidates, test_size=.25, random_state=SETTINGS.random_seed)
    return AnomalySplit(sorted(train_ids), sorted(validation_ids), sorted(fixed_test_ids))

def save_split(split: AnomalySplit, path: Path = OUTPUTS_DIR / 'anomaly_split.json') -> None:
    path.write_text(json.dumps(asdict(split), indent=2), encoding='utf-8')

def video_scores(dataset: FeatureDataset, scores: np.ndarray, video_ids: set[str], strategy: str) -> tuple[list[str], np.ndarray]:
    ids = sorted(video_ids, key=int); values = []
    for video_id in ids:
        mask = dataset.groups.astype(str) == video_id
        values.append(aggregate_scores(scores[mask], strategy))
    return ids, np.asarray(values, dtype=float)

def mask_for_ids(dataset: FeatureDataset, ids: set[str]) -> np.ndarray:
    return np.isin(dataset.groups.astype(str), list(ids))
