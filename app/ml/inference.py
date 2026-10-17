from pathlib import Path
import json, joblib
from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS
from app.ml.evaluation import aggregate_video_probability
from app.ml.features import inference_features

def analyze_video(path: Path) -> dict:
    model_path = MODELS_DIR / 'baseline.joblib'
    if not model_path.exists(): raise FileNotFoundError('Model not found. Run: python scripts/train.py')
    features, windows = inference_features(path)
    probabilities = joblib.load(model_path).predict_proba(features)[:, 1]
    best = int(probabilities.argmax())
    start, end = windows[best]
    threshold = SETTINGS.threshold
    split_path = OUTPUTS_DIR / 'split.json'
    strategy = 'top2_mean'
    if split_path.exists():
        selected = json.loads(split_path.read_text())
        threshold = float(selected.get('video_threshold', SETTINGS.threshold))
        strategy = selected.get('aggregation_strategy', strategy)
    probability = aggregate_video_probability(probabilities, strategy)
    return {'verdict': 'VIOLATION' if probability >= threshold else 'NORMAL', 'probability': probability,
            'timestamp': (start + end) / 2, 'window_start': start, 'window_end': end,
            'windows': [{'start': s, 'end': e, 'probability': float(p)} for (s, e), p in zip(windows, probabilities)]}
