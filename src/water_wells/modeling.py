"""XGBoost with readable target labels and probability columns."""
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

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_
