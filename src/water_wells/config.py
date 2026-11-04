"""Small, explicit configuration objects."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path(".")
    train_file: str = "4910797b-ee55-40a7-8668-10efd5c1b960.csv"
    labels_file: str = "0bf8bc6e-30d0-4c50-956a-603fc693d966.csv"


@dataclass
class TrainingConfig:
    random_seed: int = 42
    validation_fraction: float = 0.25
    model_path: Path = Path("models/water_well_model.joblib")
    min_frequency: int | float = 20
    max_categories: int | None = 50
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "eval_metric": "mlogloss", "n_jobs": -1,
    })
