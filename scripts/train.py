"""Cache features, choose a model with grouped CV, then fit training videos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import random
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from app.config import MODELS_DIR, OUTPUTS_DIR, SETTINGS
from app.data.dataset import load_labels
from app.ml.classifier import classifier_names, make_classifier
from app.ml.evaluation import AGGREGATION_STRATEGIES, aggregation_cv, binary_metrics, choose_threshold, choose_video_threshold, grouped_cv, video_scores
from app.ml.features import make_training_features, save_features, load_features

def main() -> None:
    random.seed(SETTINGS.random_seed); np.random.seed(SETTINGS.random_seed); torch.manual_seed(SETTINGS.random_seed)
    videos = load_labels(); targets = {v.video_id: int(v.violation) for v in videos}
    video_y = [targets[v.video_id] for v in videos]
    splitter = StratifiedGroupKFold(n_splits=SETTINGS.holdout_folds, shuffle=True, random_state=SETTINGS.random_seed)
    train_idx, test_idx = next(splitter.split(videos, video_y, groups=[v.video_id for v in videos]))
    train_ids = {videos[i].video_id for i in train_idx}; test_ids = {videos[i].video_id for i in test_idx}
    cache = OUTPUTS_DIR / f'features_ordered_v1_{SETTINGS.frame_sample_fps:g}fps.npz'
    features = load_features(cache) if cache.exists() else make_training_features(videos)
    if not cache.exists(): save_features(cache, features)
    train_mask = np.isin(features.groups, list(train_ids))
    print(f'Videos: {len(videos)} (positive {sum(targets.values())}, negative {len(videos)-sum(targets.values())})')
    print(f'Windows: {len(features.y)} (positive {features.y.sum()}, negative {len(features.y)-features.y.sum()})')
    print(f'Held-out test videos: {len(test_ids)}; training videos: {len(train_ids)}')
    comparison = {}; cv_results = {}; best_name = None; best_f1 = -1.; best_recall = -1.; best_window_threshold = .5; best_video_threshold = .5
    for name in classifier_names():
        result = grouped_cv(features, train_mask, targets, name)
        cv_results[name] = result
        window_threshold = choose_threshold(features.y[train_mask], result.probabilities)
        # Every OOF prediction is from that fold's validation partition. Aggregate
        # each validation video's windows before selecting the product threshold.
        cv_video_y, cv_video_scores, _ = video_scores(features.groups[train_mask], result.probabilities, targets)
        video_threshold = choose_video_threshold(cv_video_y, cv_video_scores)
        old_video_metric = binary_metrics(cv_video_y, cv_video_scores, window_threshold)
        new_video_metric = binary_metrics(cv_video_y, cv_video_scores, video_threshold)
        summary = {key: {'mean': float(np.nanmean([fold[key] for fold in result.folds])), 'std': float(np.nanstd([fold[key] for fold in result.folds]))} for key in result.folds[0]}
        video_summary = {key: {'mean': float(np.nanmean([fold[key] for fold in result.video_folds])), 'std': float(np.nanstd([fold[key] for fold in result.video_folds]))} for key in result.video_folds[0]}
        summary['window_threshold'] = window_threshold; summary['video_threshold'] = video_threshold
        summary['video_at_window_threshold'] = old_video_metric; summary['video_at_video_threshold'] = new_video_metric
        summary['video_fold_metrics'] = video_summary; summary['video_fold_thresholds'] = result.video_thresholds
        comparison[name] = summary
        print(f"CV {name}: video F1={video_summary['f1']['mean']:.3f}±{video_summary['f1']['std']:.3f}, video recall={video_summary['recall']['mean']:.3f}±{video_summary['recall']['std']:.3f}, video precision={video_summary['precision']['mean']:.3f}±{video_summary['precision']['std']:.3f}, video threshold={video_threshold:.2f}")
        print(f"  diagnostics only: window F1={summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f}; old/window threshold video F1={old_video_metric['f1']:.3f}; OOF pooled video threshold F1={new_video_metric['f1']:.3f}")
        # Primary selection is mean video F1. Recall breaks practically tied F1.
        f1, recall = video_summary['f1']['mean'], video_summary['recall']['mean']
        if f1 > best_f1 + .01 or (abs(f1 - best_f1) <= .01 and recall > best_recall):
            best_name, best_f1, best_recall, best_window_threshold, best_video_threshold = name, f1, recall, window_threshold, video_threshold
    # Controlled aggregation experiment: same selected classifier and same OOF
    # validation predictions for all strategies; held-out videos are untouched.
    aggregation_comparison = {}; selected_aggregation = 'top2_mean'; aggregation_f1 = -1.; aggregation_recall = -1.
    print('CV aggregation comparison (selected classifier only):')
    for strategy in AGGREGATION_STRATEGIES:
        metrics, fold_thresholds, threshold = aggregation_cv(cv_results[best_name], targets, strategy)
        summary = {key: {'mean': float(np.nanmean([fold[key] for fold in metrics])), 'std': float(np.nanstd([fold[key] for fold in metrics]))} for key in metrics[0]}
        aggregation_comparison[strategy] = {'metrics': summary, 'fold_thresholds': fold_thresholds, 'video_threshold': threshold}
        print(f"{strategy}: video F1={summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f}, recall={summary['recall']['mean']:.3f}±{summary['recall']['std']:.3f}, precision={summary['precision']['mean']:.3f}±{summary['precision']['std']:.3f}, video_threshold={threshold:.2f}")
        f1, recall = summary['f1']['mean'], summary['recall']['mean']
        if f1 > aggregation_f1 + .01 or (abs(f1 - aggregation_f1) <= .01 and recall > aggregation_recall):
            selected_aggregation, aggregation_f1, aggregation_recall, best_video_threshold = strategy, f1, recall, threshold
    model = make_classifier(best_name); model.fit(features.x[train_mask], features.y[train_mask])
    MODELS_DIR.mkdir(exist_ok=True); OUTPUTS_DIR.mkdir(exist_ok=True); joblib.dump(model, MODELS_DIR / 'baseline.joblib')
    payload = {'train_ids': sorted(train_ids), 'test_ids': sorted(test_ids), 'selected_model': best_name, 'window_threshold': best_window_threshold, 'video_threshold': best_video_threshold, 'aggregation_strategy': selected_aggregation, 'comparison': comparison, 'aggregation_comparison': aggregation_comparison}
    (OUTPUTS_DIR / 'split.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('CV model comparison:')
    for name, result in comparison.items():
        metrics = result['video_fold_metrics']
        print(f"{name}: video F1={metrics['f1']['mean']:.3f}±{metrics['f1']['std']:.3f}, recall={metrics['recall']['mean']:.3f}±{metrics['recall']['std']:.3f}, precision={metrics['precision']['mean']:.3f}±{metrics['precision']['std']:.3f}")
    print(f'Selected aggregation from CV only: strategy={selected_aggregation}; video_threshold={best_video_threshold:.2f}')
    print(f'Selected from CV only: classifier={best_name}; video_threshold={best_video_threshold:.2f}. Run evaluate.py for untouched test metrics.')

if __name__ == '__main__': main()
