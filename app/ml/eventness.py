"""Weakly supervised eventness baselines built from the fixed inference grid.

Only normal-video windows are known negatives.  In a violation video the one
window whose centre is nearest to the annotation is positive; every other
window is deliberately left unknown (``-1``) and is never fitted as negative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import SETTINGS
from app.ml.anomaly import feature_view
from app.ml.features import FeatureDataset

REPRESENTATIONS = ("absolute", "temporal", "combined")
MODELS = ("logistic_regression", "random_forest")
UNKNOWN = -1


@dataclass(frozen=True)
class EventnessRows:
    labels: np.ndarray
    sample_weight: np.ndarray


def nearest_event_window(starts: np.ndarray, ends: np.ndarray, timestamp: float) -> int:
    """Return the cached window whose centre is closest to ``timestamp``.

    ``argmin`` gives a deterministic earliest-window tie break, which matters
    for reproducible short videos and timestamps exactly between two centres.
    """
    if len(starts) == 0:
        raise ValueError("A video has no cached inference windows")
    centres = (np.asarray(starts, dtype=float) + np.asarray(ends, dtype=float)) / 2
    return int(np.abs(centres - float(timestamp)).argmin())


def known_window_labels(data: FeatureDataset, labels: dict[str, object]) -> np.ndarray:
    """Make weak labels for cached grid rows without changing their boundaries."""
    output = np.full(len(data.x), UNKNOWN, dtype=np.int8)
    groups = data.groups.astype(str)
    for video_id in np.unique(groups):
        item = labels[video_id]
        indices = np.flatnonzero(groups == video_id)
        if not item.violation:
            output[indices] = 0
        else:
            if item.timestamp is None:
                raise ValueError(f"Violation video {video_id} has no timestamp")
            local = nearest_event_window(data.starts[indices], data.ends[indices], item.timestamp)
            output[indices[local]] = 1
    return output


def supervised_rows(data: FeatureDataset, labels: dict[str, object], video_ids: Iterable[str]) -> EventnessRows:
    """Return fit labels and balanced per-video weights for a set of videos."""
    weak = known_window_labels(data, labels)
    mask = np.isin(data.groups.astype(str), list(map(str, video_ids)))
    selected = np.where(mask, weak, UNKNOWN).astype(np.int8)
    weights = np.zeros(len(selected), dtype=np.float64)
    groups = data.groups.astype(str)
    # A long normal recording must not outweigh a positive video merely because
    # it generated more overlapping grid windows.
    for video_id in np.unique(groups[mask]):
        rows = np.flatnonzero((groups == video_id) & (selected >= 0))
        if len(rows):
            weights[rows] = 1.0 / len(rows)
    return EventnessRows(selected, weights)


def make_estimator(representation: str, model_name: str) -> Pipeline | RandomForestClassifier:
    if representation not in REPRESENTATIONS or model_name not in MODELS:
        raise ValueError("Unsupported eventness configuration")
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000,
                                                random_state=SETTINGS.random_seed)),
        ])
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=SETTINGS.random_seed,
        n_jobs=-1,
    )


@dataclass
class EventnessModel:
    representation: str
    model_name: str
    estimator: object

    def score(self, x: np.ndarray) -> np.ndarray:
        # Both supported classifiers expose a calibrated-in-[0,1] positive
        # probability.  Larger always means more event-like.
        return self.estimator.predict_proba(feature_view(x, self.representation))[:, 1]


def fit_eventness_model(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                        representation: str, model_name: str) -> EventnessModel:
    """Fit only known normal windows and one positive per violation video."""
    valid = np.asarray(y) >= 0
    if set(np.asarray(y)[valid]) != {0, 1}:
        raise ValueError("A training fold must contain known examples of both classes")
    estimator = make_estimator(representation, model_name)
    view = feature_view(x[valid], representation)
    if isinstance(estimator, Pipeline):
        estimator.fit(view, np.asarray(y)[valid], classifier__sample_weight=np.asarray(sample_weight)[valid])
    else:
        estimator.fit(view, np.asarray(y)[valid], sample_weight=np.asarray(sample_weight)[valid])
    return EventnessModel(representation, model_name, estimator)


def grouped_video_folds(video_ids: list[str], video_labels: list[int], n_splits: int | None = None):
    """Split once per video; no window can cross a development fold boundary."""
    ids = np.asarray(video_ids, dtype=str)
    y = np.asarray(video_labels, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits or SETTINGS.cv_folds,
                                    shuffle=True, random_state=SETTINGS.random_seed)
    yield from splitter.split(np.zeros(len(ids)), y, groups=ids)


def video_peak_scores(groups: np.ndarray, scores: np.ndarray, video_ids: Iterable[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Max pooling: one well-localised event window can flag a video."""
    all_groups = np.asarray(groups).astype(str)
    ids, values, peak_indices = [], [], []
    for video_id in sorted(map(str, video_ids), key=int):
        rows = np.flatnonzero(all_groups == video_id)
        if not len(rows):
            raise ValueError(f"No cached rows for video {video_id}")
        local = int(np.asarray(scores)[rows].argmax())
        ids.append(video_id); values.append(float(np.asarray(scores)[rows][local])); peak_indices.append(int(rows[local]))
    return ids, np.asarray(values), np.asarray(peak_indices)


def localization_error(predicted_time: float, timestamp: float | None) -> float:
    return float("nan") if timestamp is None else abs(float(predicted_time) - float(timestamp))


def threshold_tradeoff(video_scores: np.ndarray, video_labels: np.ndarray,
                       normal_fpr_cap: float = 0.25) -> tuple[float, list[dict[str, float]]]:
    """Select a development-only video threshold with an explicit normal-FPR cap.

    Scores are OOF scores.  The candidate set includes a value above the
    largest score so a zero-alert policy is represented as well.
    """
    scores = np.asarray(video_scores, dtype=float)
    labels = np.asarray(video_labels, dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection needs normal and violation videos")
    candidates = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        prediction = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, prediction, average="binary", zero_division=0
        )
        normal = prediction[labels == 0]
        rows.append({"threshold": float(threshold), "video_precision": float(precision),
                     "video_recall": float(recall), "video_f1": float(f1),
                     "normal_video_fpr": float(normal.mean())})
    eligible = [row for row in rows if row["normal_video_fpr"] <= normal_fpr_cap]
    # If the requested cap is unattainable, maintain the lowest observed FPR.
    pool = eligible or [row for row in rows if row["normal_video_fpr"] == min(r["normal_video_fpr"] for r in rows)]
    best = max(pool, key=lambda row: (row["video_f1"], row["video_recall"], row["video_precision"], row["threshold"]))
    return float(best["threshold"]), rows
