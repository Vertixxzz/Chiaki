"""Validate that XLSX labels map one-to-one to local videos before training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.dataset import load_labels
from app.video.reader import video_info

def main() -> None:
    videos = load_labels()
    violations = [v for v in videos if v.violation]
    print('Dataset summary\n--------------')
    print(f'Total videos: {len(videos)}\nViolation: {len(violations)}\nNormal: {len(videos)-len(violations)}')
    times = [v.timestamp for v in violations if v.timestamp is not None]
    print(f'\nViolation timestamps:\nmin: {min(times):.2f}\nmax: {max(times):.2f}')
    print('\nfilename\tlabel\ttimestamp\tduration\tfps')
    for item in videos:
        fps, duration = video_info(item.path)
        print(f'{item.path.name}\t{"VIOLATION" if item.violation else "NORMAL"}\t{item.timestamp or "-"}\t{duration:.2f}\t{fps:.2f}')

if __name__ == '__main__': main()
