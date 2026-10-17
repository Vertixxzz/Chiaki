from dataclasses import dataclass
from pathlib import Path
import hashlib
import inspect
import numpy as np
from app.config import SETTINGS
from app.ml.embeddings import embed_frames
from app.video.reader import read_frames, video_info
from app.video.windows import inference_windows, labeled_windows
from app.data.dataset import VideoLabel, parse_timestamp

@dataclass
class FeatureDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

def aggregate(frame_embeddings: np.ndarray) -> np.ndarray:
    """Compact ordered representation: summary, positions, and frame-to-frame change."""
    middle = frame_embeddings[len(frame_embeddings) // 2]
    if len(frame_embeddings) > 1:
        diffs = np.diff(frame_embeddings, axis=0)
        diff_mean, diff_std = diffs.mean(0), diffs.std(0)
    else:
        diff_mean = diff_std = np.zeros_like(frame_embeddings[0])
    return np.concatenate([frame_embeddings.mean(0), frame_embeddings.std(0), frame_embeddings.max(0),
                           frame_embeddings[0], middle, frame_embeddings[-1], diff_mean, diff_std]).astype(np.float32)

def aggregate_anomaly(frame_embeddings: np.ndarray) -> np.ndarray:
    """Extended normal-model vector including mean/std delta magnitude."""
    base = aggregate(frame_embeddings)
    if len(frame_embeddings) > 1:
        magnitudes = np.linalg.norm(np.diff(frame_embeddings, axis=0), axis=1)
        magnitude = np.array([magnitudes.mean(), magnitudes.std()], dtype=np.float32)
    else: magnitude = np.zeros(2, dtype=np.float32)
    return np.concatenate([base, magnitude]).astype(np.float32)

def window_feature(path: Path, start: float, end: float) -> np.ndarray:
    frames = list(read_frames(path, start, end, SETTINGS.frame_sample_fps))
    return aggregate(embed_frames(frames))

def anomaly_window_feature(path: Path, start: float, end: float) -> np.ndarray:
    frames = list(read_frames(path, start, end, SETTINGS.frame_sample_fps))
    return aggregate_anomaly(embed_frames(frames))

def make_training_features(videos: list[VideoLabel]) -> FeatureDataset:
    rows: list[np.ndarray] = []; labels: list[int] = []; groups: list[str] = []; starts: list[float] = []; ends: list[float] = []
    for item in videos:
        _, duration = video_info(item.path)
        event = item.timestamp if item.violation else None
        for window in labeled_windows(duration, event, SETTINGS.window_size, SETTINGS.window_step,
                                      SETTINGS.event_ambiguity_margin):
            rows.append(window_feature(item.path, window.start, window.end)); labels.append(window.label)
            groups.append(item.video_id); starts.append(window.start); ends.append(window.end)
    return FeatureDataset(np.vstack(rows), np.array(labels), np.array(groups), np.array(starts), np.array(ends))

def make_inference_grid_features(videos: list[VideoLabel]) -> FeatureDataset:
    """Extract features on the same fixed grid used in deployed inference.

    The labels array is intentionally all zero: anomaly training must not see
    violation labels or timestamps while constructing its feature cache.
    """
    rows: list[np.ndarray] = []; groups: list[str] = []; starts: list[float] = []; ends: list[float] = []
    for item in videos:
        _, duration = video_info(item.path)
        for start, end in inference_windows(duration, SETTINGS.window_size, SETTINGS.window_step):
            rows.append(anomaly_window_feature(item.path, start, end)); groups.append(item.video_id)
            starts.append(start); ends.append(end)
    return FeatureDataset(np.vstack(rows), np.zeros(len(rows), dtype=np.int8), np.array(groups), np.array(starts), np.array(ends))

def feature_fingerprint(videos: list[VideoLabel]) -> str:
    """Fingerprint label source, video revisions and window/feature code/settings."""
    digest = hashlib.sha256()
    labels_file = videos[0].path.parent / 'open_labels.xlsx'
    digest.update(labels_file.read_bytes())
    for video in sorted(videos, key=lambda item: item.video_id):
        stat = video.path.stat()
        digest.update(f'{video.video_id}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    digest.update(repr((SETTINGS.window_size, SETTINGS.window_step, SETTINGS.frame_sample_fps,
                        SETTINGS.event_safety_margin, SETTINGS.event_ambiguity_margin,
                        SETTINGS.temporal_feature_set)).encode())
    digest.update(inspect.getsource(labeled_windows).encode())
    digest.update(inspect.getsource(parse_timestamp).encode())
    digest.update(inspect.getsource(aggregate).encode())
    return digest.hexdigest()

def anomaly_feature_fingerprint(videos: list[VideoLabel]) -> str:
    """Fingerprint only inputs that affect inference-equivalent visual features."""
    digest = hashlib.sha256()
    for video in sorted(videos, key=lambda item: item.video_id):
        stat = video.path.stat(); digest.update(f'{video.video_id}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    digest.update(repr((SETTINGS.window_size, SETTINGS.window_step, SETTINGS.frame_sample_fps,
                        SETTINGS.temporal_feature_set)).encode())
    digest.update(inspect.getsource(inference_windows).encode())
    digest.update(inspect.getsource(aggregate_anomaly).encode())
    return digest.hexdigest()

def save_features(path: Path, dataset: FeatureDataset, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=dataset.x, y=dataset.y, groups=dataset.groups, starts=dataset.starts, ends=dataset.ends, fingerprint=np.array(fingerprint))

def cache_is_valid(path: Path, fingerprint: str) -> bool:
    if not path.exists(): return False
    with np.load(path) as data:
        return 'fingerprint' in data and str(data['fingerprint']) == fingerprint

def load_features(path: Path, expected_fingerprint: str | None = None) -> FeatureDataset:
    data = np.load(path)
    if expected_fingerprint is not None and ('fingerprint' not in data or str(data['fingerprint']) != expected_fingerprint):
        raise ValueError('Feature cache fingerprint mismatch. Rebuild features before training.')
    return FeatureDataset(data['x'], data['y'], data['groups'], data['starts'], data['ends'])

def inference_features(path: Path) -> tuple[np.ndarray, list[tuple[float, float]]]:
    _, duration = video_info(path)
    windows = inference_windows(duration, SETTINGS.window_size, SETTINGS.window_step)
    return np.vstack([window_feature(path, s, e) for s, e in windows]), windows
