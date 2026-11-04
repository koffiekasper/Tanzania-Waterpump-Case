"""XGBoost with readable target labels and probability columns."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from .config import TrainingConfig
from .data_loader import TARGET_CLASSES


class WaterWellClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, random_seed=42):
        self.params = params
        self.random_seed = random_seed

    def fit(self, X, y):
        if set(y) != set(TARGET_CLASSES):
            raise ValueError(f"Training requires all three status labels: {TARGET_CLASSES}")
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        params = TrainingConfig().xgb_params.copy()
        params.update(self.params or {})
        params.setdefault("random_state", self.random_seed)
        params["objective"] = "multi:softprob"
        params["num_class"] = 3
        self.model_ = XGBClassifier(**params)
        self.model_.fit(X, self.label_encoder_.transform(y))
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.label_encoder_.inverse_transform(self.model_.predict(X).astype(int))

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def probability_frame(self, X, index=None):
        return pd.DataFrame(self.predict_proba(X), columns=self.classes_, index=index)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_

    def save(self, path: str | Path):
        check_is_fitted(self, "model_")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        """Load only trusted artifacts: joblib can execute arbitrary code."""
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError("Artifact is not a WaterWellClassifier")
        return artifact
