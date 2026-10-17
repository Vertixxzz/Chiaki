"""Evaluate the saved model on the untouched held-out videos.

This script never trains a model or selects a threshold. Window labels are
used only for window-level evaluation; video scores are computed solely from
model probabilities and the configured aggregation strategy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

# Allow running this file as ``python scripts/evaluate.py`` from the project root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS  # noqa: E402
from app.data.dataset import load_labels  # noqa: E402
from app.ml.evaluation import video_scores  # noqa: E402
from app.ml.features import load_features  # noqa: E402


def _binary_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Calculate metrics and confusion matrix from exactly the same predictions."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if len(y_true) != len(scores):
        raise ValueError(f"y_true has {len(y_true)} rows but scores has {len(scores)}")
    if len(y_true) == 0:
        raise ValueError("No evaluated samples remain after filtering ignored labels")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("_binary_report expects only labels 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values")

    y_pred = (scores >= float(threshold)).astype(np.int64)
    # Explicit label order guarantees [[TN, FP], [FN, TP]], even if a class
    # is absent from predictions.
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    accuracy = (tn + tp) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    auc = (
        float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )

    # Internal consistency checks: metrics must agree with the displayed matrix.
    matrix_accuracy = (tn + tp) / len(y_true)
    matrix_precision = tp / (tp + fp) if tp + fp else 0.0
    matrix_recall = tp / (tp + fn) if tp + fn else 0.0
    matrix_f1 = (
        2 * matrix_precision * matrix_recall / (matrix_precision + matrix_recall)
        if matrix_precision + matrix_recall
        else 0.0
    )
    assert np.isclose(accuracy, matrix_accuracy)
    assert np.isclose(precision, matrix_precision)
    assert np.isclose(recall, matrix_recall)
    assert np.isclose(f1, matrix_f1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
    }, matrix


def _thresholds(*values: float) -> list[float]:
    """Return unique thresholds while preserving the caller's order."""
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        threshold = float(value)
        key = round(threshold, 8)
        if key not in seen:
            result.append(threshold)
            seen.add(key)
    return result


def _print_metrics(prefix: str, metric: dict[str, float]) -> None:
    auc_text = (
        f"{metric['roc_auc']:.3f}"
        if np.isfinite(metric["roc_auc"])
        else "n/a (only one true class)"
    )
    print(
        f"{prefix}: accuracy={metric['accuracy']:.3f} "
        f"precision={metric['precision']:.3f} "
        f"recall={metric['recall']:.3f} F1={metric['f1']:.3f} "
        f"AUC={auc_text}"
    )


def main() -> None:
    fps = SETTINGS.frame_sample_fps
    feature_path = OUTPUTS_DIR / f"features_ordered_v1_{fps:g}fps.npz"
    split_path = OUTPUTS_DIR / "split.json"
    model_path = MODELS_DIR / "baseline.joblib"

    for path in (feature_path, split_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")

    features = load_features(feature_path)
    split: dict[str, Any] = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = {str(video_id) for video_id in split["test_ids"]}
    window_threshold = float(split["window_threshold"])
    video_threshold = float(split["video_threshold"])
    aggregation = split.get("aggregation_strategy", "top2_mean")

    # Load labels once. They are only used for ground-truth labels and the
    # localization metric, never to create scores or choose window positions.
    items = load_labels()
    items_by_id = {str(item.video_id): item for item in items}
    targets = {video_id: int(item.violation) for video_id, item in items_by_id.items()}

    heldout_mask = np.isin(features.groups.astype(str), list(test_ids))
    if not heldout_mask.any():
        raise ValueError("The held-out split matched zero rows in the feature cache")

    heldout_groups = features.groups[heldout_mask].astype(str)
    heldout_y_all = np.asarray(features.y[heldout_mask], dtype=np.int64)
    heldout_x = features.x[heldout_mask]
    heldout_starts = np.asarray(features.starts[heldout_mask], dtype=np.float64)
    heldout_ends = np.asarray(features.ends[heldout_mask], dtype=np.float64)

    model = joblib.load(model_path)
    all_class_probs = np.asarray(model.predict_proba(heldout_x), dtype=np.float64)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Saved classifier has no positive class 1; classes={classes}")
    probabilities_all = all_class_probs[:, classes.index(1)]

    # Ignored windows (label -1) are excluded from window metrics and confusion
    # matrix, but still participate in video scores exactly as they do at inference.
    metric_mask = heldout_y_all >= 0
    y_window = heldout_y_all[metric_mask]
    p_window = probabilities_all[metric_mask]

    count_positive = int(np.count_nonzero(heldout_y_all == 1))
    count_negative = int(np.count_nonzero(heldout_y_all == 0))
    count_ignored = int(np.count_nonzero(heldout_y_all < 0))
    print(
        f"Held-out videos: {len(test_ids)}; "
        f"windows before filtering: {len(heldout_y_all)}; "
        f"positive={count_positive}; negative={count_negative}; ignored={count_ignored}; "
        f"evaluated={int(metric_mask.sum())}"
    )
    print("Window metric convention: positive label = 1; ignored label = -1; AUC is threshold-independent.")

    for threshold in _thresholds(0.3, 0.5, 0.7, window_threshold):
        metric, _ = _binary_report(y_window, p_window, threshold)
        _print_metrics(f"window threshold={threshold:.2f}", metric)

    selected_window_metric, selected_window_cm = _binary_report(
        y_window, p_window, window_threshold
    )
    print(f"Window confusion matrix at selected validation threshold ({window_threshold:.3f}):")
    print(selected_window_cm)
    print("Matrix format: [[TN, FP], [FN, TP]]")
    _print_metrics("window selected-threshold check", selected_window_metric)

    # IMPORTANT: video scores use every inference-equivalent window. Ground-truth
    # timestamps/labels are passed only so video_scores can attach video targets;
    # they do not affect the model probabilities or aggregation.
    video_y, video_p, video_ids = video_scores(
        heldout_groups,
        probabilities_all,
        targets,
        aggregation,
    )
    video_y = np.asarray(video_y, dtype=np.int64)
    video_p = np.asarray(video_p, dtype=np.float64)
    video_ids = [str(video_id) for video_id in video_ids]

    if len(video_y) != len(video_p) or len(video_y) != len(video_ids):
        raise ValueError(
            "video_scores returned inconsistent lengths: "
            f"labels={len(video_y)}, scores={len(video_p)}, ids={len(video_ids)}"
        )

    print(f"Video aggregation: {aggregation} (configured in split.json)")
    for threshold in _thresholds(0.3, 0.5, 0.7, video_threshold):
        metric, _ = _binary_report(video_y, video_p, threshold)
        _print_metrics(f"video threshold={threshold:.3f}", metric)

    selected_video_metric, selected_video_cm = _binary_report(
        video_y, video_p, video_threshold
    )
    print(f"Video confusion matrix at selected validation threshold ({video_threshold:.3f}):")
    print(selected_video_cm)
    print("Matrix format: [[TN, FP], [FN, TP]]")
    _print_metrics("video selected-threshold check", selected_video_metric)

    # Localization diagnostic: choose the highest-scoring inference window in
    # each annotated violation video. The timestamp is used only for evaluation.
    localization_errors: list[float] = []
    window_centers = (heldout_starts + heldout_ends) / 2.0
    for video_id in video_ids:
        item = items_by_id.get(video_id)
        if item is None or not item.violation or item.timestamp is None:
            continue
        local_indices = np.flatnonzero(heldout_groups == video_id)
        if local_indices.size == 0:
            continue
        best_local_index = int(local_indices[np.argmax(probabilities_all[local_indices])])
        predicted_center = float(window_centers[best_local_index])
        localization_errors.append(abs(predicted_center - float(item.timestamp)))

    if localization_errors:
        print(
            "Timestamp localization absolute error (highest-scoring window centre): "
            f"mean={np.mean(localization_errors):.3f}s "
            f"median={np.median(localization_errors):.3f}s "
            f"over {len(localization_errors)} violation videos"
        )
    else:
        print("Timestamp localization error: unavailable (no annotated held-out violations)")


if __name__ == "__main__":
    main()
