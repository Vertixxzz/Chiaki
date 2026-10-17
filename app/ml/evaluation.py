"""Leak-free grouped evaluation utilities shared by train and evaluate scripts."""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from app.config import SETTINGS
from app.ml.classifier import make_classifier
from app.ml.features import FeatureDataset

def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'accuracy': accuracy_score(y, pred), 'precision': precision, 'recall': recall, 'f1': f1,
            'roc_auc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else float('nan'), 'positive_rate': float(pred.mean())}

AGGREGATION_STRATEGIES = (
    'max', 'top2_mean', 'top3_mean', 'top2_weighted', 'top3_weighted',
)

def aggregate_video_probability(probabilities: np.ndarray, strategy: str = 'top2_mean') -> float:
    """Aggregate sorted window probabilities without changing the classifier."""
    values = np.sort(probabilities)
    if strategy == 'max': return float(values[-1])
    if strategy == 'top2_mean': return float(values[-min(2, len(values)):].mean())
    if strategy == 'top3_mean': return float(values[-min(3, len(values)):].mean())
    if strategy == 'top2_weighted':
        top = values[-min(2, len(values)):][::-1]; weights = np.array([.7, .3])[:len(top)]
    elif strategy == 'top3_weighted':
        top = values[-min(3, len(values)):][::-1]; weights = np.array([.6, .3, .1])[:len(top)]
    else: raise ValueError(f'Unknown aggregation strategy: {strategy}')
    return float(np.average(top, weights=weights))

def video_scores(groups: np.ndarray, probabilities: np.ndarray, video_targets: dict[str, int], strategy: str = 'top2_mean') -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = sorted(set(groups)); scores = []; labels = []
    for video_id in ids:
        scores.append(aggregate_video_probability(probabilities[groups == video_id], strategy))
        labels.append(video_targets[video_id])
    return np.array(labels), np.array(scores), ids

def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Validation-only F2 threshold, constraining trivially all-positive output."""
    candidates = np.arange(.15, .86, .05); best = (.5, -1.0)
    for threshold in candidates:
        metric = binary_metrics(y, p, threshold)
        if metric['precision'] >= .15 and metric['positive_rate'] <= .70:
            f2 = 5 * metric['precision'] * metric['recall'] / (4 * metric['precision'] + metric['recall'] + 1e-12)
            if f2 > best[1]: best = (float(threshold), f2)
    return best[0]

def choose_video_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Choose threshold on validation *videos*, maximizing F1 and then recall.

    Input scores must be out-of-fold: each video is first scored by a model that
    did not train on that video. The positive-rate guard avoids all-positive
    solutions on the small dataset.
    """
    best = (.5, -1.0, -1.0)
    for threshold in np.arange(.15, .86, .05):
        metric = binary_metrics(y, scores, threshold)
        if metric['precision'] >= .15 and metric['positive_rate'] <= .70:
            candidate = (float(threshold), metric['f1'], metric['recall'])
            if candidate[1:] > best[1:]: best = candidate
    return best[0]

@dataclass
class CVResult:
    probabilities: np.ndarray
    folds: list[dict[str, float]]
    video_folds: list[dict[str, float]]
    video_thresholds: list[float]
    fold_groups: list[np.ndarray]
    fold_probabilities: list[np.ndarray]

@dataclass
class NestedCVResult:
    window_folds: list[dict[str, float]]
    aggregation_folds: dict[str, list[dict[str, float]]]
    aggregation_thresholds: dict[str, list[float]]

def grouped_cv(dataset: FeatureDataset, train_mask: np.ndarray, video_targets: dict[str, int], classifier: str) -> CVResult:
    indices = np.flatnonzero(train_mask); groups = dataset.groups[indices]
    group_y = np.array([video_targets[g] for g in groups])
    splitter = StratifiedGroupKFold(n_splits=SETTINGS.cv_folds, shuffle=True, random_state=SETTINGS.random_seed)
    probabilities = np.full(len(indices), np.nan); folds = []; video_folds = []; video_thresholds = []; fold_groups = []; fold_probabilities = []
    for fit, valid in splitter.split(dataset.x[indices], group_y, groups):
        model = make_classifier(classifier); model.fit(dataset.x[indices][fit], dataset.y[indices][fit])
        p = model.predict_proba(dataset.x[indices][valid])[:, 1]; probabilities[valid] = p
        fold_groups.append(groups[valid]); fold_probabilities.append(p)
        folds.append(binary_metrics(dataset.y[indices][valid], p, .5))
        # `valid` contains whole unseen videos. Aggregate only these windows,
        # then select/evaluate this fold's video threshold without held-out data.
        y_video, p_video, _ = video_scores(groups[valid], p, video_targets)
        threshold = choose_video_threshold(y_video, p_video)
        video_thresholds.append(threshold)
        video_folds.append(binary_metrics(y_video, p_video, threshold))
    return CVResult(probabilities, folds, video_folds, video_thresholds, fold_groups, fold_probabilities)

def aggregation_cv(result: CVResult, video_targets: dict[str, int], strategy: str) -> tuple[list[dict[str, float]], list[float], float]:
    """Evaluate one video aggregator only on the model's grouped CV validation outputs."""
    fold_metrics: list[dict[str, float]] = []; fold_thresholds: list[float] = []
    all_groups = []; all_probabilities = []
    for groups, probabilities in zip(result.fold_groups, result.fold_probabilities):
        y, scores, _ = video_scores(groups, probabilities, video_targets, strategy)
        threshold = choose_video_threshold(y, scores)
        fold_thresholds.append(threshold); fold_metrics.append(binary_metrics(y, scores, threshold))
        all_groups.append(groups); all_probabilities.append(probabilities)
    y, scores, _ = video_scores(np.concatenate(all_groups), np.concatenate(all_probabilities), video_targets, strategy)
    return fold_metrics, fold_thresholds, choose_video_threshold(y, scores)

def _splits(indices: np.ndarray, groups: np.ndarray, targets: dict[str, int], folds: int):
    labels = np.array([targets[group] for group in groups[indices]])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SETTINGS.random_seed)
    return splitter.split(np.zeros(len(indices)), labels, groups[indices])

def nested_grouped_cv(dataset: FeatureDataset, supervised_mask: np.ndarray, video_targets: dict[str, int], classifier: str) -> NestedCVResult:
    """Nested group CV: inner videos choose thresholds; outer videos evaluate them."""
    indices = np.flatnonzero(supervised_mask)
    groups = dataset.groups
    window_folds: list[dict[str, float]] = []
    metrics = {strategy: [] for strategy in AGGREGATION_STRATEGIES}
    thresholds = {strategy: [] for strategy in AGGREGATION_STRATEGIES}
    for outer_fit_rel, outer_valid_rel in _splits(indices, groups, video_targets, SETTINGS.cv_folds):
        outer_fit, outer_valid = indices[outer_fit_rel], indices[outer_valid_rel]
        # Inner OOF probabilities belong only to outer-training videos.
        inner_oof = np.full(len(outer_fit), np.nan)
        for inner_fit_rel, inner_valid_rel in _splits(outer_fit, groups, video_targets, SETTINGS.cv_folds):
            model = make_classifier(classifier)
            model.fit(dataset.x[outer_fit[inner_fit_rel]], dataset.y[outer_fit[inner_fit_rel]])
            inner_oof[inner_valid_rel] = model.predict_proba(dataset.x[outer_fit[inner_valid_rel]])[:, 1]
        outer_model = make_classifier(classifier)
        outer_model.fit(dataset.x[outer_fit], dataset.y[outer_fit])
        outer_probabilities = outer_model.predict_proba(dataset.x[outer_valid])[:, 1]
        window_folds.append(binary_metrics(dataset.y[outer_valid], outer_probabilities, .5))
        for strategy in AGGREGATION_STRATEGIES:
            inner_y, inner_scores, _ = video_scores(groups[outer_fit], inner_oof, video_targets, strategy)
            threshold = choose_video_threshold(inner_y, inner_scores)
            outer_y, outer_scores, _ = video_scores(groups[outer_valid], outer_probabilities, video_targets, strategy)
            thresholds[strategy].append(threshold)
            metrics[strategy].append(binary_metrics(outer_y, outer_scores, threshold))
    return NestedCVResult(window_folds, metrics, thresholds)

def final_video_threshold(dataset: FeatureDataset, supervised_mask: np.ndarray, video_targets: dict[str, int], classifier: str, strategy: str) -> float:
    """Choose deployment threshold on training-only grouped OOF video scores."""
    indices = np.flatnonzero(supervised_mask); groups = dataset.groups
    oof = np.full(len(indices), np.nan)
    for fit_rel, valid_rel in _splits(indices, groups, video_targets, SETTINGS.cv_folds):
        model = make_classifier(classifier)
        model.fit(dataset.x[indices[fit_rel]], dataset.y[indices[fit_rel]])
        oof[valid_rel] = model.predict_proba(dataset.x[indices[valid_rel]])[:, 1]
    y, scores, _ = video_scores(groups[indices], oof, video_targets, strategy)
    return choose_video_threshold(y, scores)
