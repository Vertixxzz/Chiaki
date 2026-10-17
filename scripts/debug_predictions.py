"""Export held-out evidence only; these images are never used for training/tuning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import joblib
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS
from app.data.dataset import load_labels
from app.debug.evidence import event_context_timestamps, nearby_window_timestamps, window_timestamps
from app.ml.evaluation import aggregate_video_probability
from app.ml.features import load_features
from app.video.reader import video_info
from app.video.windows import labeled_windows

def read_frame(path: Path, requested: float) -> tuple[np.ndarray | None, float | None]:
    """Read at requested seconds and retain decoder-reported timestamp for auditing."""
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, requested * 1000)
    ok, frame = cap.read()
    # POS_MSEC after read points past the decoded frame. Convert the returned
    # frame index back to the decoded frame's timestamp instead.
    fps = cap.get(cv2.CAP_PROP_FPS); position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    actual = max(0.0, (position - 1) / fps) if ok and fps > 0 else None
    cap.release()
    return frame if ok else None, actual

def save_contact_sheet(path: Path, samples: list[tuple[str, float]], output: Path, title: str) -> list[dict[str, float | str]]:
    """Save labelled frames. Captions show requested and decoder-reported seconds."""
    fps, _ = video_info(path); cells = []; manifest = []
    for label, requested in samples:
        frame, actual = read_frame(path, requested)
        if frame is None: continue
        error = abs((actual or 0) - requested)
        verified = error <= max(1 / fps, .05)
        if not verified:
            raise RuntimeError(f'Decoder timestamp mismatch for {path.name}: requested={requested:.3f}, decoded={actual:.3f}')
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); image.thumbnail((320, 200))
        cells.append((image, f'{label}: requested={requested:.3f}s, decoded={actual:.3f}s'))
        manifest.append({'frame_role': label, 'requested_timestamp_sec': requested, 'decoded_timestamp_sec': actual,
                         'timestamp_error_sec': error, 'timestamp_verified': verified})
    rows = max(1, (len(cells) + 2) // 3)
    sheet = Image.new('RGB', (960, 28 + rows * 235), 'white'); draw = ImageDraw.Draw(sheet); draw.text((10, 8), title, fill='black')
    for index, (image, caption) in enumerate(cells):
        x, y = (index % 3) * 320, 28 + (index // 3) * 235
        sheet.paste(image, (x + 4, y + 4)); draw.text((x + 4, y + 208), caption, fill='black')
    output.parent.mkdir(parents=True, exist_ok=True); sheet.save(output, quality=92)
    return manifest

def window_truth(start: float, end: float, violation: bool, event_time: float | None, duration: float) -> int:
    """Diagnostic truth from the same fixed-grid labelling rule as training."""
    if not violation or event_time is None: return 0
    for window in labeled_windows(duration, event_time, SETTINGS.window_size, SETTINGS.window_step,
                                  SETTINGS.event_ambiguity_margin):
        if abs(window.start - start) < 1e-6 and abs(window.end - end) < 1e-6:
            return window.label
    raise ValueError(f'Window {start}-{end} is not in the inference grid')

def error_type(true_label: int, predicted_label: int, scope: str) -> str:
    if true_label and predicted_label: return f'{scope}_tp'
    if not true_label and predicted_label: return f'{scope}_fp'
    if true_label and not predicted_label: return f'{scope}_fn'
    return f'{scope}_tn'

def main() -> None:
    split = json.loads((OUTPUTS_DIR / 'split.json').read_text()); test_ids = set(split['test_ids'])
    window_threshold = float(split['window_threshold']); video_threshold = float(split['video_threshold'])
    strategy = split.get('aggregation_strategy', 'top2_mean')
    data = load_features(OUTPUTS_DIR / f'features_ordered_v1_{SETTINGS.frame_sample_fps:g}fps.npz')
    mask = np.isin(data.groups, list(test_ids)); probabilities = joblib.load(MODELS_DIR / 'baseline.joblib').predict_proba(data.x[mask])[:, 1]
    labels = {item.video_id: item for item in load_labels()}
    infos = {key: video_info(item.path) for key, item in labels.items()}
    durations = {key: info[1] for key, info in infos.items()}
    # A timestamp exactly at duration can be past the final decodable frame.
    frame_limits = {key: max(0.0, duration - 1 / fps) for key, (fps, duration) in infos.items()}
    report = pd.DataFrame({'video_id': data.groups[mask], 'window_start': data.starts[mask], 'window_end': data.ends[mask],
                           'predicted_probability': probabilities})
    report['video_label'] = report.video_id.map(lambda ident: int(labels[str(ident)].violation))
    report['event_timestamp'] = report.video_id.map(lambda ident: labels[str(ident)].timestamp)
    report['true_window_label'] = [window_truth(row.window_start, row.window_end, bool(row.video_label), row.event_timestamp,
                                                 durations[str(row.video_id)]) for _, row in report.iterrows()]
    report['predicted_window_label'] = (report.predicted_probability >= window_threshold).astype(int)
    report['window_error_type'] = [error_type(a, b, 'window') if a >= 0 else 'window_ignored' for a, b in zip(report.true_window_label, report.predicted_window_label)]
    scores = {str(video_id): aggregate_video_probability(group.predicted_probability.to_numpy(), strategy)
              for video_id, group in report.groupby('video_id')}
    report['final_video_score'] = report.video_id.map(lambda ident: scores[str(ident)])
    report['predicted_video_label'] = (report.final_video_score >= video_threshold).astype(int)
    report['video_error_type'] = [error_type(a, b, 'video') for a, b in zip(report.video_label, report.predicted_video_label)]
    report = report.sort_values('predicted_probability', ascending=False)
    diagnostic_path = OUTPUTS_DIR / 'heldout_diagnostic.csv'; report.to_csv(diagnostic_path, index=False)

    root = OUTPUTS_DIR / 'evidence_frames'; event_dir = root / 'event_contact_sheets'; predicted_dir = root / 'predicted_windows'
    manifest: list[dict[str, object]] = []
    # Every held-out violation gets source-timestamp event context, independently of prediction outcome.
    for video_id in sorted(test_ids, key=int):
        item = labels[video_id]
        if item.violation and item.timestamp is not None:
            path = event_dir / f'video_{video_id}_event_{item.timestamp:.3f}.jpg'
            for row in save_contact_sheet(item.path, event_context_timestamps(item.timestamp, frame_limits[video_id]), path,
                                          f'video={video_id} label=VIOLATION source_event={item.timestamp:.3f}s'):
                manifest.append({'video_id': video_id, 'sheet_type': 'event_context', **row, 'path': str(path)})
    # Highest-scoring predicted window for every held-out video: start/middle/end evidence.
    for video_id, group in report.groupby('video_id'):
        best = group.iloc[0]; item = labels[str(video_id)]
        path = predicted_dir / f'video_{video_id}_best_window_{best.window_start:.3f}_{best.window_end:.3f}.jpg'
        for row in save_contact_sheet(item.path, window_timestamps(best.window_start, best.window_end, frame_limits[str(video_id)]), path,
                                      f'video={video_id} best window={best.window_start:.3f}-{best.window_end:.3f}s p={best.predicted_probability:.3f}'):
            manifest.append({'video_id': video_id, 'sheet_type': 'best_predicted_window', **row, 'path': str(path)})
    # Window FP is intentionally separate from video FP; preserve local context for each selected FP window.
    for _, row in report[report.window_error_type == 'window_fp'].head(SETTINGS.evidence_per_category).iterrows():
        item = labels[str(row.video_id)]; path = predicted_dir / f'window_fp_{row.video_id}_{row.window_start:.3f}_{row.window_end:.3f}.jpg'
        midpoint = (row.window_start + row.window_end) / 2
        # The model scores windows, not individual frames. This is the midpoint
        # of the high-scoring window plus nearby visual context.
        samples = [('top_scoring_window_midpoint', midpoint)] + nearby_window_timestamps(row.window_start, row.window_end, frame_limits[str(row.video_id)])
        for sample in save_contact_sheet(item.path, samples, path, f'WINDOW FP video={row.video_id}; video status={row.video_error_type}; p={row.predicted_probability:.3f}'):
            manifest.append({'video_id': row.video_id, 'sheet_type': 'window_fp_context', **sample, 'path': str(path)})
    manifest_path = root / 'frame_manifest.csv'; pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f'Saved diagnostic CSV: {diagnostic_path}')
    print(f'Saved event contact sheets: {event_dir}')
    print(f'Saved predicted-window evidence: {predicted_dir}')
    print(f'Saved timestamp manifest: {manifest_path}')

if __name__ == '__main__': main()
