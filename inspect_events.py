from pathlib import Path
import re
import unicodedata

import cv2
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "dataset"
LABELS_FILE = ROOT / "open_labels.xlsx"
OUT_DIR = ROOT / "inspection"
FRAME_DIR = OUT_DIR / "frames"

FRAME_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(value):
    name = unicodedata.normalize("NFKC", str(value)).strip()
    name = name.replace("Х", "X").replace("х", "x")
    name = re.sub(r"\.mov$", "", name, flags=re.I)
    return name.casefold()


def parse_timestamp(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    minutes = re.search(r"(\d+)\s*мин", text)
    seconds = re.search(r"(\d+)\s*сек", text)

    return (
        int(minutes.group(1)) * 60 if minutes else 0
    ) + (
        int(seconds.group(1)) if seconds else 0
    )


videos = {}
for path in VIDEO_DIR.glob("*.mov"):
    key = normalize_name(path.name)
    if key in videos:
        raise ValueError(f"Duplicate normalized filename: {path.name}")
    videos[key] = path

labels = pd.read_excel(LABELS_FILE)
records = []

for _, row in labels.iterrows():
    key = normalize_name(row["наименование файла"])
    path = videos.get(key)

    if path is None:
        print(f"[MISSING VIDEO] {row['наименование файла']}")
        continue

    label = str(row["нарушение / без нарушения"]).strip().casefold()
    timestamp = parse_timestamp(row["таймстемп"])
    is_violation = label == "нарушение"

    if is_violation and timestamp is None:
        print(f"[MISSING TIMESTAMP] {path.name}")
        continue

    records.append({
        "path": path,
        "label": label,
        "timestamp": timestamp,
    })

print(f"Matched labels: {len(records)}/{len(labels)}")

tiles = []

for item in records:
    path = item["path"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    if item["label"] == "нарушение":
        target = item["timestamp"]
        times = sorted({
            max(0, min(duration - 0.05, target + delta))
            for delta in (-1, 0, 1)
        })
    else:
        times = sorted({
            duration * fraction for fraction in (0.25, 0.5, 0.75)
        })

    for second in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()

        if not ok:
            print(f"[FRAME ERROR] {path.name} at {second:.1f}s")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        filename = f"{path.stem}_{second:06.1f}s.jpg"
        image.save(FRAME_DIR / filename, quality=92)

        tiles.append((
            image.copy(),
            f"{path.name} | {item['label']} | {second:.1f}s"
        ))

    cap.release()

columns, rows_per_page = 3, 4
cell_w, cell_h = 360, 270
page_size = columns * rows_per_page

for page_no in range((len(tiles) + page_size - 1) // page_size):
    page_tiles = tiles[page_no * page_size:(page_no + 1) * page_size]
    sheet = Image.new(
        "RGB",
        (columns * cell_w, rows_per_page * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for i, (image, caption) in enumerate(page_tiles):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h

        image.thumbnail((cell_w - 10, cell_h - 45))
        sheet.paste(image, (x + 5, y + 5))
        draw.text((x + 5, y + cell_h - 35), caption, fill="black")

    sheet.save(OUT_DIR / f"contact_sheet_{page_no + 1}.jpg", quality=90)

print(f"Extracted frames: {len(tiles)}")
print(f"Output directory: {OUT_DIR}")