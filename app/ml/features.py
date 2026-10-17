from dataclasses import dataclass
from pathlib import Path
import numpy as np
from app.config import SETTINGS
from app.ml.embeddings import embed_frames
from app.video.reader import read_frames, video_info
from app.video.windows import labeled_windows, sliding_windows
from app.data.dataset import VideoLabel

@dataclass
class FeatureDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

def aggregate(frame_embeddings: np.ndarray) -> np.ndarray:
    return np.concatenate([frame_embeddings.mean(0), frame_embeddings.std(0), frame_embeddings.max(0)]).astype(np.float32)

def window_feature(path: Path, start: float, end: float) -> np.ndarray:
    frames = list(read_frames(path, start, end, SETTINGS.frame_sample_fps))
    return aggregate(embed_frames(frames))

def make_training_features(videos: list[VideoLabel]) -> FeatureDataset:
    rows: list[np.ndarray] = []; labels: list[int] = []; groups: list[str] = []; starts: list[float] = []; ends: list[float] = []
    for item in videos:
        _, duration = video_info(item.path)
        event = item.timestamp if item.violation else None
        for window in labeled_windows(duration, event, SETTINGS.window_size, SETTINGS.window_step):
            rows.append(window_feature(item.path, window.start, window.end)); labels.append(window.label)
            groups.append(item.video_id); starts.append(window.start); ends.append(window.end)
    return FeatureDataset(np.vstack(rows), np.array(labels), np.array(groups), np.array(starts), np.array(ends))

def save_features(path: Path, dataset: FeatureDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=dataset.x, y=dataset.y, groups=dataset.groups, starts=dataset.starts, ends=dataset.ends)

def load_features(path: Path) -> FeatureDataset:
    data = np.load(path)
    return FeatureDataset(data['x'], data['y'], data['groups'], data['starts'], data['ends'])

def inference_features(path: Path) -> tuple[np.ndarray, list[tuple[float, float]]]:
    _, duration = video_info(path)
    windows = list(sliding_windows(duration, SETTINGS.window_size, SETTINGS.window_step))
    return np.vstack([window_feature(path, s, e) for s, e in windows]), windows
