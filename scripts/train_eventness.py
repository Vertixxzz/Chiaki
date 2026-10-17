"""Train small weakly-supervised eventness baselines from the existing cache.

This script never triggers CLIP extraction.  It requires the fixed,
inference-grid cache built by the anomaly experiment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.ml.eventness import (MODELS, REPRESENTATIONS, fit_eventness_model,
                              grouped_video_folds, known_window_labels,
                              supervised_rows, threshold_tradeoff,
                              video_peak_scores)
from app.ml.features import load_features

CACHE_PATH = OUTPUTS_DIR / "features_anomaly_grid_2fps.npz"
NORMAL_FPR_CAP = 0.25


def check_cache(data) -> None:
    if not (len(data.x) == len(data.groups) == len(data.starts) == len(data.ends)):
        raise ValueError("Feature cache arrays have inconsistent lengths")
    if data.x.ndim != 2 or data.x.shape[1] < 10:
        raise ValueError("Feature cache lacks usable aggregated temporal features")
    if np.any(data.ends < data.starts):
        raise ValueError("Feature cache contains invalid window boundaries")


def oof_scores(data, labels, development_ids, representation: str, model_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Produce window scores with every video's score made by an excluding fold."""
    development_ids = sorted(map(str, development_ids), key=int)
    video_y = [int(labels[video_id].violation) for video_id in development_ids]
    result = np.full(len(data.x), np.nan, dtype=float)
    folds = np.full(len(data.x), -1, dtype=int)
    for fold, (train_pos, validation_pos) in enumerate(grouped_video_folds(development_ids, video_y)):
        train_ids = [development_ids[index] for index in train_pos]
        validation_ids = [development_ids[index] for index in validation_pos]
        train = supervised_rows(data, labels, train_ids)
        model = fit_eventness_model(data.x, train.labels, train.sample_weight, representation, model_name)
        validation_mask = np.isin(data.groups.astype(str), validation_ids)
        result[validation_mask] = model.score(data.x[validation_mask])
        folds[validation_mask] = fold
    mask = np.isin(data.groups.astype(str), development_ids)
    if np.isnan(result[mask]).any() or (folds[mask] < 0).any():
        raise RuntimeError("Grouped OOF scoring did not cover every development window")
    weak = known_window_labels(data, labels)
    rows = pd.DataFrame({
        "video_id": data.groups[mask].astype(str), "window_start": data.starts[mask], "window_end": data.ends[mask],
        "event_score": result[mask], "known_window_label": weak[mask], "fold": folds[mask],
    })
    rows["video_label"] = rows.video_id.map(lambda video_id: int(labels[video_id].violation))
    rows["known_timestamp"] = rows.video_id.map(lambda video_id: labels[video_id].timestamp)
    return result, rows


def _safe_auc(y: np.ndarray, scores: np.ndarray, function) -> float:
    return float(function(y, scores)) if len(np.unique(y)) == 2 else float("nan")


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing {CACHE_PATH}; run the existing cache builder explicitly first.")
    data = load_features(CACHE_PATH); check_cache(data)
    labels = {item.video_id: item for item in load_labels()}
    fixed = json.loads((OUTPUTS_DIR / "split.json").read_text(encoding="utf-8"))
    test_ids = set(map(str, fixed["test_ids"]))
    cached_ids = set(data.groups.astype(str))
    if cached_ids != set(labels):
        raise ValueError("Cache and XLSX label video IDs differ; do not train on a stale cache")
    development_ids = sorted(cached_ids - test_ids, key=int)
    print(f"Using cached inference-grid features: rows={len(data.x)}, dimensions={data.x.shape[1]}")
    print(f"Fixed held-out videos={len(test_ids)}; development videos={len(development_ids)} "
          f"(normal={sum(not labels[i].violation for i in development_ids)}, violation={sum(labels[i].violation for i in development_ids)})")

    validation_rows, artifacts, summaries = [], [], []
    for representation in REPRESENTATIONS:
        for model_name in MODELS:
            scores, window_rows = oof_scores(data, labels, development_ids, representation, model_name)
            ids, video_scores, peak_rows = video_peak_scores(data.groups, scores, development_ids)
            video_y = np.asarray([int(labels[video_id].violation) for video_id in ids])
            threshold, tradeoff = threshold_tradeoff(video_scores, video_y, NORMAL_FPR_CAP)
            chosen = next(row for row in tradeoff if row["threshold"] == threshold)
            weak = known_window_labels(data, labels)
            known_mask = np.isin(data.groups.astype(str), development_ids) & (weak >= 0)
            window_y = weak[known_mask]
            window_scores = scores[known_mask]
            # Rows are retained so the threshold curve can be audited, but OOF
            # predictions are not fit again nor replaced by in-sample values.
            for row in tradeoff:
                validation_rows.append({"representation": representation, "model": model_name,
                                        "selection_normal_fpr_cap": NORMAL_FPR_CAP, **row,
                                        "is_selected_threshold": int(row["threshold"] == threshold)})
            summaries.append({"representation": representation, "model": model_name, **chosen,
                              "window_roc_auc": _safe_auc(window_y, window_scores, roc_auc_score),
                              "window_pr_auc": _safe_auc(window_y, window_scores, average_precision_score),
                              "development_known_windows": int(known_mask.sum()),
                              "development_unknown_windows": int((np.isin(data.groups.astype(str), development_ids) & (weak < 0)).sum())})
            final = supervised_rows(data, labels, development_ids)
            fitted = fit_eventness_model(data.x, final.labels, final.sample_weight, representation, model_name)
            artifacts.append({"representation": representation, "model_name": model_name, "threshold": threshold,
                              "model": fitted, "selection": chosen, "development_ids": development_ids,
                              "test_ids": sorted(test_ids, key=int), "normal_fpr_cap": NORMAL_FPR_CAP})

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUTPUTS_DIR / "eventness_validation.csv", index=False)
    summary = pd.DataFrame(summaries)
    # Selection is explicitly video-level: F1 under the normal-FPR cap, then
    # higher recall, then lower FPR. Window metrics are diagnostic only.
    selected_row = max(summaries, key=lambda row: (row["video_f1"], row["video_recall"], -row["normal_video_fpr"], row["video_precision"]))
    selected_index = summaries.index(selected_row)
    selected = artifacts[selected_index]
    selected["summary"] = selected_row
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"candidates": artifacts, "summary": summaries, "cache": str(CACHE_PATH)}, MODELS_DIR / "eventness_candidates.joblib")
    joblib.dump(selected, MODELS_DIR / "eventness_selected.joblib")
    print("\nOOF development comparison (model and threshold selection are development-only):")
    print(summary.to_string(index=False))
    print(f"\nSelected eventness baseline: model={selected['model_name']}, representation={selected['representation']}, "
          f"video_threshold={selected['threshold']:.6f}; normal-FPR cap={NORMAL_FPR_CAP:.2f}")
    print(f"Saved {OUTPUTS_DIR / 'eventness_validation.csv'} and eventness model artifacts.")


if __name__ == "__main__":
    main()
