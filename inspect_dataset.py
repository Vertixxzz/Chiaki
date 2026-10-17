from pathlib import Path

import cv2
import pandas as pd


VIDEO_DIR = Path(__file__).resolve().parent / "dataset"
LABELS_FILE = Path(__file__).resolve().parent / "open_labels.xlsx"

rows = []

for path in sorted(VIDEO_DIR.glob("*.mov")):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        rows.append({
            "file": path.name,
            "readable": False,
        })
        print(f"[ERROR] {path.name}: cannot open")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0

    ok, frame = cap.read()
    cap.release()

    rows.append({
        "file": path.name,
        "readable": ok,
        "fps": round(fps, 2),
        "frames": frames,
        "duration_sec": round(duration, 2),
        "width": width,
        "height": height,
    })

report = pd.DataFrame(rows)
print("\n=== VIDEO REPORT ===")
print(report.to_string(index=False))

report.to_csv("video_report.csv", index=False)

print(f"\nVideo count: {len(report)}")
print(f"Readable: {report['readable'].sum()}")

labels = pd.read_excel(LABELS_FILE)
print("\n=== LABELS ===")
print(labels.to_string(index=False))
print("\nColumns:", labels.columns.tolist())