from pathlib import Path
import joblib
from app.config import MODELS_DIR, SETTINGS
from app.ml.features import inference_features

def analyze_video(path: Path) -> dict:
    model_path = MODELS_DIR / 'baseline.joblib'
    if not model_path.exists(): raise FileNotFoundError('Model not found. Run: python scripts/train.py')
    features, windows = inference_features(path)
    probabilities = joblib.load(model_path).predict_proba(features)[:, 1]
    best = int(probabilities.argmax())
    probability = float(probabilities[best])
    start, end = windows[best]
    return {'verdict': 'VIOLATION' if probability >= SETTINGS.threshold else 'NORMAL', 'probability': probability,
            'timestamp': (start + end) / 2, 'window_start': start, 'window_end': end,
            'windows': [{'start': s, 'end': e, 'probability': float(p)} for (s, e), p in zip(windows, probabilities)]}
