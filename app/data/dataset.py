from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
from app.config import DATASET_DIR

@dataclass(frozen=True)
class VideoLabel:
    video_id: str
    path: Path
    violation: bool
    timestamp: float | None
    original_filename: str
    raw_timestamp: str | None

def _number(value: object) -> str:
    match = re.search(r"(\d+)", str(value))
    if not match: raise ValueError(f"No numeric video id in {value!r}")
    return str(int(match.group(1)))

def parse_timestamp(value: object) -> float | None:
    """Parse Russian timestamp strings, including compact minute/second forms."""
    if pd.isna(value): return None
    text = str(value).strip().lower().replace(',', '.')
    number = r"(\d+(?:\.\d+)?)"
    minute = re.search(number + r"\s*(?:минута|минуты|минут|мин\.?)", text)
    second = re.search(number + r"\s*(?:секунда|секунды|секунд|сек\.?)", text)
    if minute:
        return float(minute.group(1)) * 60 + (float(second.group(1)) if second else 0.0)
    if second: return float(second.group(1))
    # Numeric XLSX values are interpreted as seconds.
    fallback = re.search(number, text)
    return float(fallback.group(1)) if fallback else None

def _is_violation(value: object) -> bool:
    text = str(value).strip().lower()
    if "без нарушения" in text: return False
    if "наруш" in text: return True
    raise ValueError(f"Unknown label: {value!r}")

def load_labels(dataset_dir: Path = DATASET_DIR) -> list[VideoLabel]:
    table = pd.read_excel(dataset_dir / "open_labels.xlsx")
    cols = {str(c).strip().lower(): c for c in table.columns}
    file_col = next((c for n, c in cols.items() if "наимен" in n or "файл" in n), None)
    time_col = next((c for n, c in cols.items() if "тайм" in n or "timestamp" in n), None)
    label_col = next((c for n, c in cols.items() if "наруш" in n), None)
    if not all((file_col, time_col, label_col)): raise ValueError(f"Unexpected XLSX columns: {list(table.columns)}")
    paths = {p.stem: p for p in dataset_dir.glob("*.mov")}
    result = []
    for _, row in table.iterrows():
        video_id = _number(row[file_col])
        if video_id not in paths: raise FileNotFoundError(f"XLSX refers to {row[file_col]!r}; expected {video_id}.mov")
        raw_time = None if pd.isna(row[time_col]) else str(row[time_col])
        result.append(VideoLabel(video_id, paths[video_id], _is_violation(row[label_col]), parse_timestamp(row[time_col]), str(row[file_col]), raw_time))
    return result
