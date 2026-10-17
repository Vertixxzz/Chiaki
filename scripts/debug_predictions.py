"""Save held-out window predictions and representative evidence frames; never used for training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, joblib, cv2, numpy as np, pandas as pd
from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS
from app.ml.features import load_features
from app.data.dataset import load_labels

def save_frame(video: Path, timestamp: float, destination: Path) -> None:
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000); ok, frame = cap.read(); cap.release()
    if ok: cv2.imwrite(str(destination), frame)

def main() -> None:
    split = json.loads((OUTPUTS_DIR / 'split.json').read_text()); test_ids = set(split['test_ids']); threshold = split['window_threshold']
    data = load_features(OUTPUTS_DIR / f'features_ordered_v1_{SETTINGS.frame_sample_fps:g}fps.npz'); mask = np.isin(data.groups, list(test_ids))
    probabilities = joblib.load(MODELS_DIR / 'baseline.joblib').predict_proba(data.x[mask])[:, 1]
    report = pd.DataFrame({'video_id': data.groups[mask], 'start_sec': data.starts[mask], 'end_sec': data.ends[mask], 'true_label': data.y[mask], 'predicted_probability': probabilities})
    report['predicted_label'] = (report.predicted_probability >= threshold).astype(int)
    report = report.sort_values('predicted_probability', ascending=False)
    report.to_csv(OUTPUTS_DIR / 'heldout_windows.csv', index=False)
    paths = {v.video_id: v.path for v in load_labels()}; evidence = OUTPUTS_DIR / 'evidence_frames'; evidence.mkdir(parents=True, exist_ok=True)
    report['category'] = np.select([(report.true_label == 1) & (report.predicted_label == 1), (report.true_label == 0) & (report.predicted_label == 1), (report.true_label == 1) & (report.predicted_label == 0)], ['tp', 'fp', 'fn'], default='tn')
    for category, rows in report.groupby('category'):
        # Highest scores for positives, lowest scores for missed positives/true negatives.
        rows = rows.sort_values('predicted_probability', ascending=category in ('fn', 'tn')).head(SETTINGS.evidence_per_category)
        for index, row in rows.iterrows():
            output = evidence / f"{category}_{row.video_id}_{row.start_sec:.1f}_{row.predicted_probability:.3f}.jpg"
            save_frame(paths[str(row.video_id)], (row.start_sec + row.end_sec) / 2, output)
    print(f'Saved {OUTPUTS_DIR / "heldout_windows.csv"} and evidence frames in {evidence}')

if __name__ == '__main__': main()
