"""One composed pipeline shared by experiments and inference."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from .config import TrainingConfig
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .population_features import PopulationFeatures


class TrainingPipeline(ClassifierMixin, BaseEstimator):
    probability_columns = {
        "functional": "probability_functional",
        "functional needs repair": "probability_needs_repair",
        "non functional": "probability_non_functional",
    }

    def __init__(self, config=None, population_raster=None,
                 cache_dir="data/processed/population", data_mode="auto"):
        self.config = config
        self.population_raster = population_raster
        self.cache_dir = cache_dir
        self.data_mode = data_mode

    def prepare_population_data(self, X, output_path=None):
        """Populate/reuse the geo cache before splitting; optionally export CSV.

        Only external raster features are computed here. Training-dependent
        neighbor counts and encoders remain inside fit, after splitting.
        """
        data = X.copy()
        if self.population_raster is not None:
            data = PopulationFeatures(
                self.population_raster, self.cache_dir, self.data_mode,
            ).add_features(data)
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False)
        return data

    def split(self, X, y):
        """Split raw rows before any learned transformation (stratified)."""
        config = self.config or TrainingConfig()
        return train_test_split(X, y, test_size=config.validation_fraction,
                                random_state=config.random_seed, stratify=y)

    def fit(self, X, y):
        """Fit exactly the supplied rows; use split() first for validation."""
        config = self.config or TrainingConfig()
        self.label_encoder_ = LabelEncoder().fit(y)
        params = TrainingConfig().xgb_params.copy()
        params.update(config.xgb_params)
        params.setdefault("random_state", config.random_seed)
        params.update(objective="multi:softprob", num_class=3)
        self.pipeline_ = Pipeline([
            ("features", FeatureEngineer(
                population_raster=self.population_raster,
                cache_dir=self.cache_dir, data_mode=self.data_mode,
            )),
            ("preprocessing", DataPreprocessor(config.min_frequency, config.max_categories)),
            ("classifier", XGBClassifier(**params)),
        ])
        self.pipeline_.fit(X, self.label_encoder_.transform(y))
        self.classes_ = self.label_encoder_.classes_
        return self

    @property
    def classifier(self):
        return self.pipeline_.named_steps["classifier"]

    @property
    def preprocessor(self):
        return self.pipeline_.named_steps["preprocessing"]

    def predict(self, X):
        return self.label_encoder_.inverse_transform(self.pipeline_.predict(X).astype(int))

    def predict_proba(self, X):
        """Return probabilities with readable class names, in input row order."""
        return pd.DataFrame(self.pipeline_.predict_proba(X), index=X.index,
                            columns=self.classes_)

    def predict_wells(self, X):
        probabilities = self.predict_proba(X)
        result = X[["id", "latitude", "longitude"]].copy()
        result["predicted_status"] = self.classes_[probabilities.to_numpy().argmax(axis=1)]
        for label, column in self.probability_columns.items():
            result[column] = probabilities[label].to_numpy()
        return result

    def save(self, path: str | Path | None = None):
        path = Path(path or (self.config or TrainingConfig()).model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        """Load only trusted artifacts; joblib deserialization executes code."""
        artifact = joblib.load(path)
        return artifact
