"""Leak-free grouped evaluation utilities shared by train and evaluate scripts."""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from app.config import SETTINGS
from app.ml.classifier import make_classifier
from app.ml.features import FeatureDataset

def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'accuracy': accuracy_score(y, pred), 'precision': precision, 'recall': recall, 'f1': f1,
            'roc_auc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else float('nan'), 'positive_rate': float(pred.mean())}

def aggregate_video_probability(probabilities: np.ndarray) -> float:
    """Robust video score: average the two strongest window probabilities."""
    values = np.sort(probabilities)
    return float(values[-min(2, len(values)):].mean())

def video_scores(groups: np.ndarray, probabilities: np.ndarray, video_targets: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = sorted(set(groups)); scores = []; labels = []
    for video_id in ids:
        scores.append(aggregate_video_probability(probabilities[groups == video_id]))
        labels.append(video_targets[video_id])
    return np.array(labels), np.array(scores), ids

def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Validation-only F2 threshold, constraining trivially all-positive output."""
    candidates = np.arange(.15, .86, .05); best = (.5, -1.0)
    for threshold in candidates:
        metric = binary_metrics(y, p, threshold)
        if metric['precision'] >= .15 and metric['positive_rate'] <= .70:
            f2 = 5 * metric['precision'] * metric['recall'] / (4 * metric['precision'] + metric['recall'] + 1e-12)
            if f2 > best[1]: best = (float(threshold), f2)
    return best[0]

def choose_video_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Choose threshold on validation *videos*, maximizing F1 and then recall.

    Input scores must be out-of-fold: each video is first scored by a model that
    did not train on that video. The positive-rate guard avoids all-positive
    solutions on the small dataset.
    """
    best = (.5, -1.0, -1.0)
    for threshold in np.arange(.15, .86, .05):
        metric = binary_metrics(y, scores, threshold)
        if metric['precision'] >= .15 and metric['positive_rate'] <= .70:
            candidate = (float(threshold), metric['f1'], metric['recall'])
            if candidate[1:] > best[1:]: best = candidate
    return best[0]

@dataclass
class CVResult:
    probabilities: np.ndarray
    folds: list[dict[str, float]]
    video_folds: list[dict[str, float]]
    video_thresholds: list[float]

def grouped_cv(dataset: FeatureDataset, train_mask: np.ndarray, video_targets: dict[str, int], classifier: str) -> CVResult:
    indices = np.flatnonzero(train_mask); groups = dataset.groups[indices]
    group_y = np.array([video_targets[g] for g in groups])
    splitter = StratifiedGroupKFold(n_splits=SETTINGS.cv_folds, shuffle=True, random_state=SETTINGS.random_seed)
    probabilities = np.full(len(indices), np.nan); folds = []; video_folds = []; video_thresholds = []
    for fit, valid in splitter.split(dataset.x[indices], group_y, groups):
        model = make_classifier(classifier); model.fit(dataset.x[indices][fit], dataset.y[indices][fit])
        p = model.predict_proba(dataset.x[indices][valid])[:, 1]; probabilities[valid] = p
        folds.append(binary_metrics(dataset.y[indices][valid], p, .5))
        # `valid` contains whole unseen videos. Aggregate only these windows,
        # then select/evaluate this fold's video threshold without held-out data.
        y_video, p_video, _ = video_scores(groups[valid], p, video_targets)
        threshold = choose_video_threshold(y_video, p_video)
        video_thresholds.append(threshold)
        video_folds.append(binary_metrics(y_video, p_video, threshold))
    return CVResult(probabilities, folds, video_folds, video_thresholds)
