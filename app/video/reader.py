from pathlib import Path
from typing import Iterator
import cv2
import numpy as np

def video_info(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {path}")
    fps, frames = float(cap.get(cv2.CAP_PROP_FPS)), float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0: raise ValueError(f"Invalid FPS in {path}")
    return fps, frames / fps

def read_frames(path: Path, start: float, end: float, sample_fps: float) -> Iterator[np.ndarray]:
    """Read RGB frames at evenly spaced timestamps in [start, end)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {path}")
    time = start
    try:
        while time < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            ok, frame = cap.read()
            if ok: yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time += 1 / sample_fps
    finally: cap.release()
