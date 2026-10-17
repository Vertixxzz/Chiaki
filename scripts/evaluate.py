"""Report window and video metrics on the untouched held-out videos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS
from app.ml.features import load_features
from app.ml.evaluation import binary_metrics, video_scores
from app.data.dataset import load_labels

def main() -> None:
    features = load_features(OUTPUTS_DIR / f'features_ordered_v1_{SETTINGS.frame_sample_fps:g}fps.npz')
    split = json.loads((OUTPUTS_DIR / 'split.json').read_text())
    test_ids = set(split['test_ids']); window_threshold = split['window_threshold']; video_threshold = split['video_threshold']
    targets = {v.video_id: int(v.violation) for v in load_labels()}
    mask = np.isin(features.groups, list(test_ids)); y = features.y[mask]
    probabilities = joblib.load(MODELS_DIR / 'baseline.joblib').predict_proba(features.x[mask])[:, 1]
    print(f'Held-out videos: {len(test_ids)}; windows: {len(y)}; positive windows: {y.sum()}')
    for threshold in (0.3, 0.5, 0.7):
        metric = binary_metrics(y, probabilities, threshold)
        print(f"window threshold={threshold:.1f}: accuracy={metric['accuracy']:.3f} precision={metric['precision']:.3f} recall={metric['recall']:.3f} F1={metric['f1']:.3f}")
    print(f"Window confusion matrix at selected validation threshold ({window_threshold:.2f}):\n", confusion_matrix(y, probabilities >= window_threshold))
    vy, vp, _ = video_scores(features.groups[mask], probabilities, targets)
    for threshold in (0.3, 0.5, 0.7, window_threshold, video_threshold):
        metric = binary_metrics(vy, vp, threshold)
        print(f"video threshold={threshold:.2f}: accuracy={metric['accuracy']:.3f} precision={metric['precision']:.3f} recall={metric['recall']:.3f} F1={metric['f1']:.3f} AUC={metric['roc_auc']:.3f}")

if __name__ == '__main__': main()
