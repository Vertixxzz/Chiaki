"""Audit XLSX → labels → temporal-window integrity. It never trains a model."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app.config import DATASET_DIR, OUTPUTS_DIR, SETTINGS
from app.data.dataset import load_labels
from app.video.reader import video_info
from app.video.windows import labeled_windows

def sync_csv() -> None:
    labels = load_labels()
    pd.DataFrame({'name': [f'{x.video_id}.mov' for x in labels], 'timestamp': [x.timestamp for x in labels],
                  'violation': [x.violation for x in labels]}).to_csv(DATASET_DIR / 'labels.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-csv', action='store_true', help='Regenerate stale derived labels.csv from XLSX.')
    args = parser.parse_args()
    if args.sync_csv: sync_csv()
    labels = load_labels(); csv_path = DATASET_DIR / 'labels.csv'
    csv = pd.read_csv(csv_path) if csv_path.exists() else None
    csv_by_id = {str(int(Path(row.name).stem)): row for row in csv.itertuples(index=False)} if csv is not None else {}
    rows = []; failures = 0
    for item in labels:
        _, duration = video_info(item.path); problems: list[str] = []
        if item.video_id != item.path.stem: problems.append('video_id/path mismatch')
        if not item.original_filename or item.video_id not in item.original_filename: problems.append('prefix/id mismatch')
        positive_start = positive_end = None
        if item.violation:
            if item.timestamp is None: problems.append('missing violation timestamp')
            elif not 0 <= item.timestamp <= duration: problems.append('timestamp outside duration')
            else:
                windows = labeled_windows(duration, item.timestamp, SETTINGS.window_size, SETTINGS.window_step, SETTINGS.event_ambiguity_margin)
                positive = [w for w in windows if w.label == 1]
                if not positive: problems.append('positive-window count')
                else:
                    positive_start, positive_end = min(w.start for w in positive), max(w.end for w in positive)
                    if not all(w.start <= item.timestamp <= w.end for w in positive): problems.append('event outside positive window')
                    guard_start, guard_end = item.timestamp - SETTINGS.event_ambiguity_margin, item.timestamp + SETTINGS.event_ambiguity_margin
                    if any(w.label == 0 and w.start < guard_end and w.end > guard_start for w in windows): problems.append('negative inside ambiguity margin')
        elif item.timestamp is not None: problems.append('normal has timestamp')
        csv_row = csv_by_id.get(item.video_id)
        if csv is None: problems.append('labels.csv missing')
        elif csv_row is None: problems.append('labels.csv id missing')
        else:
            csv_time = None if pd.isna(csv_row.timestamp) else float(csv_row.timestamp)
            if csv_time != item.timestamp or bool(csv_row.violation) != item.violation: problems.append('labels.csv stale')
        status = 'OK' if not problems else '; '.join(problems)
        failures += status != 'OK'
        rows.append({'original_filename': item.original_filename, 'video_id': item.video_id, 'raw_timestamp': item.raw_timestamp,
                     'parsed_timestamp_sec': item.timestamp, 'duration_sec': round(duration, 3),
                     'positive_window_start': positive_start, 'positive_window_end': positive_end, 'status': status})
    report = pd.DataFrame(rows); OUTPUTS_DIR.mkdir(exist_ok=True); report.to_csv(OUTPUTS_DIR / 'label_audit.csv', index=False)
    print(report.to_string(index=False))
    print(f'\nAudit summary: videos={len(report)}, violation={sum(x.violation for x in labels)}, normal={sum(not x.violation for x in labels)}, passed={len(report)-failures}, failed={failures}')
    print(f'Report: {OUTPUTS_DIR / "label_audit.csv"}')
    if failures: raise SystemExit(1)

if __name__ == '__main__': main()
