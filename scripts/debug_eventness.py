"""Export visual evidence for frozen eventness results; never used for fitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.debug.evidence import event_context_timestamps, window_timestamps
from app.ml.features import load_features
from app.video.reader import video_info
from debug_predictions import save_contact_sheet


def main() -> None:
    artifact = joblib.load(MODELS_DIR / "eventness_selected.joblib")
    data = load_features(OUTPUTS_DIR / "features_anomaly_grid_2fps.npz")
    labels = {item.video_id: item for item in load_labels()}
    report = pd.read_csv(OUTPUTS_DIR / "eventness_test_videos.csv", dtype={"video_id": str})
    root = OUTPUTS_DIR / "eventness_evidence"; root.mkdir(exist_ok=True)
    manifest = []
    for row in report.itertuples(index=False):
        item = labels[row.video_id]
        fps, duration = video_info(item.path); frame_limit = max(0.0, duration - 1 / fps)
        peak_path = root / f"video_{row.video_id}_predicted_peak.jpg"
        title = (f"Eventness peak video={row.video_id}; label={row.video_label}; score={row.event_score:.4f}; "
                 f"threshold={row.threshold:.4f}; no violation type is inferred")
        for sample in save_contact_sheet(item.path, window_timestamps(row.peak_window_start, row.peak_window_end, frame_limit), peak_path, title):
            manifest.append({"video_id": row.video_id, "sheet_type": "predicted_peak", "path": str(peak_path), **sample})
        if item.violation and item.timestamp is not None:
            event_path = root / f"video_{row.video_id}_known_event.jpg"
            for sample in save_contact_sheet(item.path, event_context_timestamps(item.timestamp, frame_limit), event_path,
                                             f"Known annotation video={row.video_id}; timestamp={item.timestamp:.3f}s"):
                manifest.append({"video_id": row.video_id, "sheet_type": "known_event", "path": str(event_path), **sample})
    pd.DataFrame(manifest).to_csv(root / "frame_manifest.csv", index=False)
    normal = report[report.video_label == 0].sort_values("event_score", ascending=False)
    missed = report[(report.video_label == 1) & (report.predicted_label == 0)]
    distant = report[(report.video_label == 1) & (report.timestamp_error > 10)]
    print(f"Saved eventness evidence to {root}")
    print("Most suspicious normal videos:\n" + normal[["video_id", "event_score", "predicted_timestamp"]].to_string(index=False))
    print("Missed violation videos:\n" + missed[["video_id", "event_score", "known_timestamp", "predicted_timestamp"]].to_string(index=False))
    print("Violation peaks >10s from timestamp:\n" + distant[["video_id", "timestamp_error", "known_timestamp", "predicted_timestamp"]].to_string(index=False))


if __name__ == "__main__":
    main()
