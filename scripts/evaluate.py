"""Report metrics only on held-out videos selected by train.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from app.config import MODELS_DIR, OUTPUTS_DIR
from app.ml.features import load_features

def main() -> None:
    features = load_features(OUTPUTS_DIR / 'features.npz')
    test_ids = set((OUTPUTS_DIR / 'split.txt').read_text().splitlines())
    mask = np.isin(features.groups, list(test_ids)); y = features.y[mask]
    probabilities = joblib.load(MODELS_DIR / 'baseline.joblib').predict_proba(features.x[mask])[:, 1]
    print(f'Held-out videos: {len(test_ids)}; windows: {len(y)}; positive windows: {y.sum()}')
    for threshold in (0.3, 0.5, 0.7):
        pred = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        print(f' threshold={threshold:.1f}: accuracy={accuracy_score(y,pred):.3f} precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}')
    print('Confusion matrix at 0.5:\n', confusion_matrix(y, probabilities >= .5))
    if len(np.unique(y)) == 2: print(f'ROC-AUC: {roc_auc_score(y, probabilities):.3f}')

if __name__ == '__main__': main()
