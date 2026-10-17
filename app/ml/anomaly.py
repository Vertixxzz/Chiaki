"""Small, local normal-behaviour anomaly models.

All estimators are fitted exclusively on windows from normal training videos.
Their public scores always have one direction: larger means more anomalous.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from app.config import SETTINGS

REPRESENTATIONS = ('absolute', 'temporal', 'combined')
DETECTORS = ('knn', 'isolation_forest', 'one_class_svm')
AGGREGATIONS = ('max', 'top3_mean', 'top5_mean')

def feature_view(x: np.ndarray, representation: str) -> np.ndarray:
    """Select absolute, delta-based, or combined parts of the fixed feature vector."""
    if representation == 'combined': return x
    embedding_dim = (x.shape[1] - 2) // 8
    if embedding_dim <= 0 or embedding_dim * 8 + 2 != x.shape[1]:
        raise ValueError(f'Unexpected temporal feature width: {x.shape[1]}')
    if representation == 'absolute': return x[:, :embedding_dim * 6]
    if representation == 'temporal': return x[:, embedding_dim * 6:]
    raise ValueError(f'Unknown representation: {representation}')

def aggregate_scores(scores: np.ndarray, strategy: str) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if strategy == 'max': return float(values[-1])
    if strategy == 'top3_mean': return float(values[-min(3, len(values)):].mean())
    if strategy == 'top5_mean': return float(values[-min(5, len(values)):].mean())
    raise ValueError(f'Unknown score aggregation: {strategy}')

def smooth_scores(scores: np.ndarray, width: int = 3) -> np.ndarray:
    """Centered rolling mean for visualization/localization only."""
    if len(scores) < 2 or width <= 1: return np.asarray(scores, dtype=float)
    pad = width // 2
    padded = np.pad(scores, (pad, pad), mode='edge')
    return np.convolve(padded, np.ones(width) / width, mode='valid')

@dataclass
class NormalAnomalyModel:
    representation: str
    detector_name: str
    scaler: StandardScaler
    pca: PCA
    detector: object
    neighbors: NearestNeighbors | None = None

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(feature_view(x, self.representation)))

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self._transform(x)
        if self.detector_name == 'knn':
            distances, _ = self.neighbors.kneighbors(z)
            return distances.mean(axis=1)
        if self.detector_name == 'isolation_forest': return -self.detector.score_samples(z)
        if self.detector_name == 'one_class_svm': return -self.detector.decision_function(z)
        raise ValueError(f'Unknown detector: {self.detector_name}')

def fit_normal_model(x_normal: np.ndarray, representation: str, detector_name: str, pca_components: int = 32) -> NormalAnomalyModel:
    """Fit transformations and detector using normal windows only."""
    if detector_name not in DETECTORS or representation not in REPRESENTATIONS:
        raise ValueError('Unsupported anomaly configuration')
    x = feature_view(x_normal, representation)
    scaler = StandardScaler().fit(x)
    n_components = min(pca_components, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components, random_state=SETTINGS.random_seed).fit(scaler.transform(x))
    z = pca.transform(scaler.transform(x))
    if detector_name == 'knn':
        neighbors = NearestNeighbors(n_neighbors=min(5, len(z))).fit(z)
        return NormalAnomalyModel(representation, detector_name, scaler, pca, neighbors, neighbors)
    if detector_name == 'isolation_forest':
        detector = IsolationForest(n_estimators=300, random_state=SETTINGS.random_seed, contamination='auto').fit(z)
    else:
        detector = OneClassSVM(kernel='rbf', gamma='scale', nu=.1).fit(z)
    return NormalAnomalyModel(representation, detector_name, scaler, pca, detector)

def percentile_threshold(normal_video_scores: np.ndarray, percentile: float = 95.0) -> float:
    """Threshold policy fitted only from held-out normal validation video scores."""
    return float(np.percentile(normal_video_scores, percentile))
