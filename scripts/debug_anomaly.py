"""Evidence export for anomaly peaks; never used for fitting or threshold selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.debug.evidence import event_context_timestamps, window_timestamps
from app.ml.anomaly import smooth_scores
from app.ml.features import load_features
from app.video.reader import video_info
from debug_predictions import save_contact_sheet

def main() -> None:
    artifact = joblib.load(MODELS_DIR / 'anomaly_normal_only.joblib')
    data = load_features(OUTPUTS_DIR / 'features_anomaly_grid_2fps.npz', artifact['feature_fingerprint'])
    labels = {item.video_id: item for item in load_labels()}; test_ids = set(artifact['split'].test_ids)
    scores = artifact['model'].score(data.x); root = OUTPUTS_DIR / 'anomaly_evidence'; root.mkdir(exist_ok=True)
    for video_id in sorted(test_ids, key=int):
        item = labels[video_id]; indices = np.flatnonzero(data.groups.astype(str) == video_id)
        local_scores = scores[indices]; peak = indices[int(np.argmax(smooth_scores(local_scores)))]
        fps, duration = video_info(item.path); limit = max(0., duration - 1 / fps)
        save_contact_sheet(item.path, window_timestamps(data.starts[peak], data.ends[peak], limit),
                           root / f'video_{video_id}_predicted_peak.jpg',
                           f'Anomaly peak video={video_id}; score={scores[peak]:.5f}; this is not a violation type')
        if item.violation and item.timestamp is not None:
            save_contact_sheet(item.path, event_context_timestamps(item.timestamp, limit),
                               root / f'video_{video_id}_known_event.jpg',
                               f'Known annotation video={video_id}; timestamp={item.timestamp:.3f}s')
    print(f'Saved anomaly evidence to {root}')

if __name__ == '__main__': main()
