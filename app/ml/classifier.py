from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_classifier() -> Pipeline:
    return Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))])
