"""Opt-in full-data training; run with WATER_WELLS_FULL_TRAIN=1."""
import json
import os
from pathlib import Path
import pytest
from water_wells import DataConfig, DataLoader, ModelEvaluator, TrainingPipeline


@pytest.mark.skipif(os.environ.get("WATER_WELLS_FULL_TRAIN") != "1",
                    reason="Full 500-tree fit is opt-in")
def test_full_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    X, y = DataLoader(DataConfig(data_dir=root)).load_training()
    pipeline = TrainingPipeline()
    X_train, X_valid, X_holdout, y_train, y_valid, y_holdout = pipeline.split(X, y)
    pipeline.fit(X_train, y_train)
    metrics = ModelEvaluator().metrics(y_valid, pipeline.predict(X_valid))
    metrics["encoded_features"] = len(pipeline.preprocessor.get_feature_names_out())
    print(json.dumps(metrics, indent=2))
    # Verify a useful model without asserting the leaked notebook's exact score.
    assert metrics["accuracy"] > y_valid.value_counts(normalize=True).max()
    pipeline.save(tmp_path / "full_pipeline.joblib")
    reloaded = TrainingPipeline.load(tmp_path / "full_pipeline.joblib")
    assert (pipeline.predict(X_valid.head()) == reloaded.predict(X_valid.head())).all()
