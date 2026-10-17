"""Cache features, choose a model with grouped CV, then fit training videos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import random
import argparse
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS
from app.data.dataset import load_labels
from app.ml.classifier import classifier_names, make_classifier
from app.ml.evaluation import AGGREGATION_STRATEGIES, final_video_threshold, nested_grouped_cv
from app.ml.features import cache_is_valid, feature_fingerprint, make_training_features, save_features, load_features

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-features', action='store_true', help='Ignore feature cache and extract it again.')
    args = parser.parse_args()
    random.seed(SETTINGS.random_seed); np.random.seed(SETTINGS.random_seed); torch.manual_seed(SETTINGS.random_seed)
    videos = load_labels(); targets = {v.video_id: int(v.violation) for v in videos}
    video_y = [targets[v.video_id] for v in videos]
    splitter = StratifiedGroupKFold(n_splits=SETTINGS.holdout_folds, shuffle=True, random_state=SETTINGS.random_seed)
    train_idx, test_idx = next(splitter.split(videos, video_y, groups=[v.video_id for v in videos]))
    train_ids = {videos[i].video_id for i in train_idx}; test_ids = {videos[i].video_id for i in test_idx}
    cache = OUTPUTS_DIR / f'features_ordered_v1_{SETTINGS.frame_sample_fps:g}fps.npz'
    fingerprint = feature_fingerprint(videos)
    if args.rebuild_features or not cache_is_valid(cache, fingerprint):
        print('Feature cache missing or stale; rebuilding from audited labels.')
        features = make_training_features(videos)
        save_features(cache, features, fingerprint)
    else:
        features = load_features(cache, fingerprint)
    train_mask = np.isin(features.groups, list(train_ids))
    supervised_train_mask = train_mask & (features.y >= 0)
    ignored = features.y < 0
    print(f'Videos: {len(videos)} (positive {sum(targets.values())}, negative {len(videos)-sum(targets.values())})')
    print(f'Windows before ignored filtering: {len(features.y)}')
    print(f'Windows: positive={(features.y == 1).sum()}, negative={(features.y == 0).sum()}, ignored={ignored.sum()}, used_for_fit={supervised_train_mask.sum()}')
    print(f'Held-out test videos: {len(test_ids)}; training videos: {len(train_ids)}')
    comparison = {}; nested_results = {}; best_name = None; best_f1 = -1.; best_recall = -1.
    for name in classifier_names():
        result = nested_grouped_cv(features, supervised_train_mask, targets, name)
        nested_results[name] = result
        summary = {key: {'mean': float(np.nanmean([fold[key] for fold in result.aggregation_folds['top2_mean']])), 'std': float(np.nanstd([fold[key] for fold in result.aggregation_folds['top2_mean']]))} for key in result.aggregation_folds['top2_mean'][0]}
        comparison[name] = {'video_fold_metrics': summary, 'video_fold_thresholds': result.aggregation_thresholds['top2_mean']}
        print(f"Nested CV {name}: video F1={summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f}, recall={summary['recall']['mean']:.3f}±{summary['recall']['std']:.3f}, precision={summary['precision']['mean']:.3f}±{summary['precision']['std']:.3f}")
        f1, recall = summary['f1']['mean'], summary['recall']['mean']
        if f1 > best_f1 + .01 or (abs(f1 - best_f1) <= .01 and recall > best_recall):
            best_name, best_f1, best_recall = name, f1, recall
    aggregation_comparison = {}; selected_aggregation = 'top2_mean'; aggregation_f1 = -1.; aggregation_recall = -1.
    print('Nested CV aggregation comparison (selected classifier only):')
    for strategy in AGGREGATION_STRATEGIES:
        metrics = nested_results[best_name].aggregation_folds[strategy]
        fold_thresholds = nested_results[best_name].aggregation_thresholds[strategy]
        summary = {key: {'mean': float(np.nanmean([fold[key] for fold in metrics])), 'std': float(np.nanstd([fold[key] for fold in metrics]))} for key in metrics[0]}
        aggregation_comparison[strategy] = {'metrics': summary, 'fold_thresholds': fold_thresholds}
        print(f"{strategy}: video F1={summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f}, recall={summary['recall']['mean']:.3f}±{summary['recall']['std']:.3f}, precision={summary['precision']['mean']:.3f}±{summary['precision']['std']:.3f}")
        f1, recall = summary['f1']['mean'], summary['recall']['mean']
        if f1 > aggregation_f1 + .01 or (abs(f1 - aggregation_f1) <= .01 and recall > aggregation_recall):
            selected_aggregation, aggregation_f1, aggregation_recall = strategy, f1, recall
    best_video_threshold = final_video_threshold(features, supervised_train_mask, targets, best_name, selected_aggregation)
    best_window_threshold = .5  # Diagnostic only; deployment uses video_threshold.
    model = make_classifier(best_name); model.fit(features.x[supervised_train_mask], features.y[supervised_train_mask])
    MODELS_DIR.mkdir(exist_ok=True); OUTPUTS_DIR.mkdir(exist_ok=True); joblib.dump(model, MODELS_DIR / 'baseline.joblib')
    payload = {'train_ids': sorted(train_ids), 'test_ids': sorted(test_ids), 'selected_model': best_name, 'window_threshold': best_window_threshold, 'video_threshold': best_video_threshold, 'aggregation_strategy': selected_aggregation, 'comparison': comparison, 'aggregation_comparison': aggregation_comparison}
    (OUTPUTS_DIR / 'split.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('CV model comparison:')
    for name, result in comparison.items():
        metrics = result['video_fold_metrics']
        print(f"{name}: video F1={metrics['f1']['mean']:.3f}±{metrics['f1']['std']:.3f}, recall={metrics['recall']['mean']:.3f}±{metrics['recall']['std']:.3f}, precision={metrics['precision']['mean']:.3f}±{metrics['precision']['std']:.3f}")
    print(f'Selected aggregation from nested CV only: strategy={selected_aggregation}; deployment video_threshold={best_video_threshold:.2f}')
    print(f'Selected from CV only: classifier={best_name}; video_threshold={best_video_threshold:.2f}. Run evaluate.py for untouched test metrics.')

if __name__ == '__main__': main()
