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

def labeled_windows(duration: float, event_time: float | None, size: float, step: float) -> list[Window]:
    """Make one event-centred positive window and safe negative windows."""
    if event_time is None: return [Window(s, e, 0) for s, e in sliding_windows(duration, size, step)]
    start = max(0.0, min(event_time - size / 2, max(0.0, duration - size)))
    result = [Window(start, min(duration, start + size), 1)]
    result += [Window(s, e, 0) for s, e in sliding_windows(duration, size, step) if not s <= event_time <= e]
    return result
