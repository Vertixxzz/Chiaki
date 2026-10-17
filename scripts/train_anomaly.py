"""Train a normal-only anomaly baseline; violation samples never enter fit()."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data.dataset import load_labels
from app.ml.anomaly import AGGREGATIONS, DETECTORS, REPRESENTATIONS, fit_normal_model, percentile_threshold
from app.ml.anomaly_data import make_split, mask_for_ids, save_split, video_scores
from app.ml.features import anomaly_feature_fingerprint, cache_is_valid, load_features, make_inference_grid_features, save_features

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--rebuild-features', action='store_true')
    args = parser.parse_args(); videos = load_labels()
    legacy = json.loads((OUTPUTS_DIR / 'split.json').read_text(encoding='utf-8'))
    split = make_split(videos, set(map(str, legacy['test_ids']))); save_split(split)
    cache = OUTPUTS_DIR / 'features_anomaly_grid_2fps.npz'; fingerprint = anomaly_feature_fingerprint(videos)
    if args.rebuild_features or not cache_is_valid(cache, fingerprint):
        print('Building inference-grid feature cache (no violation labels are stored or used).')
        data = make_inference_grid_features(videos); save_features(cache, data, fingerprint)
    else: data = load_features(cache, fingerprint)
    train_mask = mask_for_ids(data, set(split.normal_train_ids)); val_ids = set(split.normal_validation_ids)
    print(f'Normal-only split: train videos={len(split.normal_train_ids)}, validation normal videos={len(val_ids)}, fixed test videos={len(split.test_ids)}')
    print(f'Normal training windows={int(train_mask.sum())}; all grid windows={len(data.x)}')
    rows = []; candidates = []; fitted_candidates = []
    for representation in REPRESENTATIONS:
        for detector in DETECTORS:
            model = fit_normal_model(data.x[train_mask], representation, detector)
            all_scores = model.score(data.x)
            for aggregation in AGGREGATIONS:
                _, normal_scores = video_scores(data, all_scores, val_ids, aggregation)
                threshold = percentile_threshold(normal_scores, 95.)
                fpr = float((normal_scores >= threshold).mean())
                row = {'representation': representation, 'detector': detector, 'aggregation': aggregation,
                       'threshold': threshold, 'validation_normal_fpr': fpr, 'validation_score_std': float(normal_scores.std())}
                rows.append(row); candidates.append((fpr, row['validation_score_std'], representation, detector, aggregation, threshold))
                fitted_candidates.append({'representation': representation, 'detector': detector, 'aggregation': aggregation,
                                          'threshold': threshold, 'model': model})
    report = pd.DataFrame(rows); report.to_csv(OUTPUTS_DIR / 'anomaly_validation.csv', index=False)
    # Selection policy sees only normal validation scores: lowest FPR, then most stable scores.
    _, _, representation, detector, aggregation, threshold = min(candidates)
    final_model = fit_normal_model(data.x[train_mask], representation, detector)
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {'model': final_model, 'representation': representation, 'detector': detector, 'aggregation': aggregation,
                'threshold': threshold, 'feature_fingerprint': fingerprint, 'split': split}
    joblib.dump(artifact, MODELS_DIR / 'anomaly_normal_only.joblib')
    joblib.dump({'candidates': fitted_candidates, 'feature_fingerprint': fingerprint, 'split': split}, MODELS_DIR / 'anomaly_candidates.joblib')
    print(report.to_string(index=False))
    print(f'Selected using normal validation only: representation={representation}, detector={detector}, aggregation={aggregation}, threshold={threshold:.6f}')

if __name__ == '__main__': main()
