"""Pure timestamp helpers for evidence export; never used by model code."""
from __future__ import annotations

def clamp_timestamp(timestamp: float, duration: float) -> float:
    return max(0.0, min(timestamp, duration))

def event_context_timestamps(event_time: float, duration: float) -> list[tuple[str, float]]:
    """Requested event context timestamps, clamped to the readable video range."""
    return [(f'event{offset:+d}s', clamp_timestamp(event_time + offset, duration))
            for offset in (-5, -2, 0, 2, 5)]

def window_timestamps(start: float, end: float, duration: float) -> list[tuple[str, float]]:
    """Start/middle/end timestamps of an already valid temporal window."""
    return [('window_start', clamp_timestamp(start, duration)),
            ('window_middle', clamp_timestamp((start + end) / 2, duration)),
            ('window_end', clamp_timestamp(end, duration))]

def nearby_window_timestamps(start: float, end: float, duration: float) -> list[tuple[str, float]]:
    """Midpoint and nearby context for a high-scoring window (not frame scores)."""
    middle = (start + end) / 2
    return [(f'nearby{offset:+d}s', clamp_timestamp(middle + offset, duration)) for offset in (-2, 0, 2)]
