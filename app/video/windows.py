from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class Window:
    start: float
    end: float
    label: int

def sliding_windows(duration: float, size: float, step: float) -> Iterator[tuple[float, float]]:
    if duration <= 0 or size <= 0 or step <= 0: return
    start = 0.0
    while start < duration:
        end = min(start + size, duration)
        yield start, end
        if end == duration: break
        start += step

def inference_windows(duration: float, size: float, step: float) -> list[tuple[float, float]]:
    """The single window-boundary source used by training, evaluation and inference."""
    return list(sliding_windows(duration, size, step))

def labeled_windows(duration: float, event_time: float | None, size: float, step: float,
                    safety_margin: float = 0.0) -> list[Window]:
    """Label a fixed inference grid; never move a boundary toward an event.

    Labels: 1 when an event timestamp is inside the window, 0 when safely
    outside its ambiguity region, and -1 for ambiguous windows.  The ambiguity
    region is event_time ± safety_margin; ignored windows are never fitted or
    included in window-level metrics, but still exist during inference.
    """
    windows = inference_windows(duration, size, step)
    if event_time is None: return [Window(start, end, 0) for start, end in windows]
    guard_start, guard_end = event_time - safety_margin, event_time + safety_margin
    result = []
    for start, end in windows:
        if start <= event_time <= end:
            label = 1
        elif start < guard_end and end > guard_start:
            label = -1
        else:
            label = 0
        result.append(Window(start, end, label))
    return result
