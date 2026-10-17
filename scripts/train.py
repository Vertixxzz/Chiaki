"""Extract frozen CLIP features, split by video, and train the baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.ml.classifier import make_classifier
from app.ml.features import make_training_features, save_features

def main() -> None:
    videos = load_labels()
    video_y = [int(v.violation) for v in videos]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(videos, video_y, groups=[v.video_id for v in videos]))
    train_ids = {videos[i].video_id for i in train_idx}; test_ids = {videos[i].video_id for i in test_idx}
    print(f'Train videos: {len(train_ids)}; test videos: {len(test_ids)}')
    features = make_training_features(videos)
    save_features(OUTPUTS_DIR / 'features.npz', features)
    train_mask = __import__('numpy').isin(features.groups, list(train_ids))
    test_mask = ~train_mask
    model = make_classifier(); model.fit(features.x[train_mask], features.y[train_mask])
    MODELS_DIR.mkdir(exist_ok=True); joblib.dump(model, MODELS_DIR / 'baseline.joblib')
    (OUTPUTS_DIR / 'split.txt').write_text('\n'.join(sorted(test_ids)), encoding='utf-8')
    print(f'Train windows: {train_mask.sum()} (positive: {features.y[train_mask].sum()}); test windows: {test_mask.sum()} (positive: {features.y[test_mask].sum()})')
    print(f'Saved model to {MODELS_DIR / "baseline.joblib"}. Run scripts/evaluate.py for held-out metrics.')

if __name__ == '__main__': main()
