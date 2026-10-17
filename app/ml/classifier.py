from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from app.config import SETTINGS

def classifier_names() -> tuple[str, ...]:
    return ('logistic_regression', 'linear_svc_calibrated', 'random_forest')

def make_classifier(name: str) -> Pipeline | RandomForestClassifier:
    if name == 'logistic_regression':
        return Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SETTINGS.random_seed))])
    if name == 'linear_svc_calibrated':
        svc = LinearSVC(class_weight='balanced', random_state=SETTINGS.random_seed)
        return Pipeline([('scaler', StandardScaler()), ('classifier', CalibratedClassifierCV(svc, cv=3, method='sigmoid'))])
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=SETTINGS.random_seed)
    raise ValueError(f'Unknown classifier: {name}')
