"""Evaluate the frozen normal-only anomaly model on the untouched fixed test videos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.ml.anomaly import smooth_scores
from app.ml.anomaly_data import mask_for_ids, video_scores
from app.ml.features import load_features

def main() -> None:
    artifact = joblib.load(MODELS_DIR / 'anomaly_normal_only.joblib')
    data = load_features(OUTPUTS_DIR / 'features_anomaly_grid_2fps.npz', artifact['feature_fingerprint'])
    split = artifact['split']; test_ids = set(split.test_ids)
    labels = {item.video_id: item for item in load_labels()}
    candidates = joblib.load(MODELS_DIR / 'anomaly_candidates.joblib')['candidates']
    comparison = []
    for candidate in candidates:
        candidate_scores = candidate['model'].score(data.x)
        ids_cmp, score_cmp = video_scores(data, candidate_scores, test_ids, candidate['aggregation'])
        y_cmp = np.array([int(labels[video_id].violation) for video_id in ids_cmp]); pred_cmp = (score_cmp >= candidate['threshold']).astype(int)
        precision_cmp, recall_cmp, f1_cmp, _ = precision_recall_fscore_support(y_cmp, pred_cmp, average='binary', zero_division=0)
        comparison.append({'representation': candidate['representation'], 'detector': candidate['detector'], 'aggregation': candidate['aggregation'],
                           'video ROC-AUC': roc_auc_score(y_cmp, score_cmp), 'PR-AUC': average_precision_score(y_cmp, score_cmp),
                           'recall': recall_cmp, 'precision': precision_cmp, 'F1': f1_cmp,
                           'normal FPR': float(pred_cmp[y_cmp == 0].mean())})
    table = pd.DataFrame(comparison); table.to_csv(OUTPUTS_DIR / 'anomaly_method_comparison.csv', index=False)
    print(table.to_string(index=False))
    scores = artifact['model'].score(data.x)
    ids, video_score = video_scores(data, scores, test_ids, artifact['aggregation'])
    y = np.array([int(labels[video_id].violation) for video_id in ids]); pred = (video_score >= artifact['threshold']).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    normal_fpr = float(pred[y == 0].mean()) if np.any(y == 0) else float('nan')
    print(f"Video test metrics: ROC-AUC={roc_auc_score(y, video_score):.3f} PR-AUC={average_precision_score(y, video_score):.3f} precision={precision:.3f} recall={recall:.3f} F1={f1:.3f} normal_FPR={normal_fpr:.3f}")
    rows = []
    for video_id, score, label, prediction in zip(ids, video_score, y, pred):
        local = np.flatnonzero(data.groups.astype(str) == video_id)
        raw = scores[local]; smooth = smooth_scores(raw); peak = local[int(np.argmax(smooth))]
        timestamp = labels[video_id].timestamp
        predicted_time = float((data.starts[peak] + data.ends[peak]) / 2)
        rows.append({'video_id': video_id, 'video_label': int(label), 'known_timestamp': timestamp,
                     'predicted_timestamp': predicted_time,
                     'timestamp_error': abs(predicted_time - timestamp) if timestamp is not None else np.nan,
                     'video_anomaly_score': float(score), 'threshold': artifact['threshold'], 'predicted_label': int(prediction),
                     'peak_window_start': float(data.starts[peak]), 'peak_window_end': float(data.ends[peak])})
    report = pd.DataFrame(rows); report.to_csv(OUTPUTS_DIR / 'anomaly_test_videos.csv', index=False)
    localization = report.timestamp_error.dropna()
    if len(localization): print(f'Localization error: mean={localization.mean():.3f}s median={localization.median():.3f}s over {len(localization)} violation videos')
    print(f'Saved {OUTPUTS_DIR / "anomaly_test_videos.csv"}')

if __name__ == '__main__': main()
