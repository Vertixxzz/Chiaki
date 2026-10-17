"""Evaluate frozen eventness candidates on the unchanged nine-video holdout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score)

from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.ml.eventness import known_window_labels, localization_error, video_peak_scores
from app.ml.features import load_features


def safe_metric(y, scores, fn) -> float:
    return float(fn(y, scores)) if len(np.unique(y)) == 2 else float("nan")


def binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def candidate_report(candidate: dict, data, labels: dict[str, object]) -> tuple[dict, pd.DataFrame]:
    test_ids = list(map(str, candidate["test_ids"]))
    mask = np.isin(data.groups.astype(str), test_ids)
    scores = candidate["model"].score(data.x[mask])
    groups = data.groups[mask].astype(str)
    starts, ends = data.starts[mask], data.ends[mask]
    # Work on this compact test-only view so peak indices are directly usable.
    ids, video_scores, peaks = video_peak_scores(groups, scores, test_ids)
    video_y = np.asarray([int(labels[video_id].violation) for video_id in ids])
    threshold = float(candidate["threshold"])
    video = binary_metrics(video_y, video_scores, threshold)
    weak_full = known_window_labels(data, labels)
    weak = weak_full[mask]
    known = weak >= 0
    window = binary_metrics(weak[known], scores[known], threshold)
    rows = []
    for video_id, video_score, peak, video_label, prediction in zip(ids, video_scores, peaks, video_y, (video_scores >= threshold).astype(int)):
        item = labels[video_id]
        predicted_timestamp = float((starts[peak] + ends[peak]) / 2)
        error = localization_error(predicted_timestamp, item.timestamp)
        rows.append({"video_id": video_id, "video_label": int(video_label), "known_timestamp": item.timestamp,
                     "predicted_timestamp": predicted_timestamp, "event_score": float(video_score),
                     "threshold": threshold, "predicted_label": int(prediction), "timestamp_error": error,
                     "peak_window_start": float(starts[peak]), "peak_window_end": float(ends[peak]),
                     "representation": candidate["representation"], "model": candidate["model_name"]})
    per_video = pd.DataFrame(rows)
    localization = per_video.loc[per_video.video_label == 1, "timestamp_error"].dropna().to_numpy()
    normal_fpr = float(((video_scores >= threshold).astype(int)[video_y == 0]).mean())
    report = {"representation": candidate["representation"], "model": candidate["model_name"], "threshold": threshold,
              "window_roc_auc": safe_metric(weak[known], scores[known], roc_auc_score),
              "window_pr_auc": safe_metric(weak[known], scores[known], average_precision_score),
              "window_precision": window["precision"], "window_recall": window["recall"], "window_f1": window["f1"],
              "video_roc_auc": safe_metric(video_y, video_scores, roc_auc_score),
              "video_pr_auc": safe_metric(video_y, video_scores, average_precision_score),
              "video_precision": video["precision"], "video_recall": video["recall"], "video_f1": video["f1"],
              "normal_video_fpr": normal_fpr,
              "localization_median_error_sec": float(np.median(localization)) if len(localization) else float("nan"),
              "localization_mean_error_sec": float(np.mean(localization)) if len(localization) else float("nan"),
              "localization_hit_2_5s": float((localization <= 2.5).mean()) if len(localization) else float("nan"),
              "localization_hit_5s": float((localization <= 5).mean()) if len(localization) else float("nan"),
              "localization_hit_10s": float((localization <= 10).mean()) if len(localization) else float("nan"),
              "known_test_windows": int(known.sum()), "unknown_test_windows_excluded": int((~known).sum())}
    return report, per_video


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "eventness_candidates.joblib")
    data = load_features(Path(bundle["cache"]))
    labels = {item.video_id: item for item in load_labels()}
    reports, selected_videos = [], None
    selected = joblib.load(MODELS_DIR / "eventness_selected.joblib")
    for candidate in bundle["candidates"]:
        report, videos = candidate_report(candidate, data, labels)
        reports.append(report)
        if candidate["representation"] == selected["representation"] and candidate["model_name"] == selected["model_name"]:
            selected_videos = videos
    comparison = pd.DataFrame(reports)
    comparison.to_csv(OUTPUTS_DIR / "eventness_method_comparison.csv", index=False)
    if selected_videos is None:
        raise RuntimeError("The selected model was not found in candidate artifact")
    selected_videos.to_csv(OUTPUTS_DIR / "eventness_test_videos.csv", index=False)
    print("Held-out test comparison (thresholds/models were frozen from development OOF predictions):")
    print(comparison.to_string(index=False))
    selected_report = next(row for row in reports if row["representation"] == selected["representation"] and row["model"] == selected["model_name"])
    print("\nSelected eventness held-out result:")
    print(pd.DataFrame([selected_report]).to_string(index=False))
    anomaly_path = OUTPUTS_DIR / "anomaly_method_comparison.csv"
    if anomaly_path.exists():
        anomaly = pd.read_csv(anomaly_path)
        print("\nSaved normal-only baseline reference (not used for any eventness choice):")
        print(anomaly[["representation", "detector", "aggregation", "F1", "normal FPR"]].to_string(index=False))
    print(f"Saved {OUTPUTS_DIR / 'eventness_method_comparison.csv'} and {OUTPUTS_DIR / 'eventness_test_videos.csv'}")


if __name__ == "__main__":
    main()
