"""Load competition CSV files without relying on row alignment."""
from pathlib import Path
import pandas as pd
from .config import DataConfig


RAW_COLUMNS = tuple("""
    id amount_tsh date_recorded funder gps_height installer longitude latitude
    wpt_name num_private basin subvillage region region_code district_code lga
    ward population public_meeting recorded_by scheme_management scheme_name
    permit construction_year extraction_type extraction_type_group
    extraction_type_class management management_group payment payment_type
    water_quality quality_group quantity quantity_group source source_type
    source_class waterpoint_type waterpoint_type_group
""".split())
TARGET_CLASSES = ("functional", "functional needs repair", "non functional")


class DataLoader:
    """Paths are relative to config.data_dir; explicit paths override defaults."""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    @staticmethod
    def validate_features(data: pd.DataFrame) -> None:
        missing = sorted(set(RAW_COLUMNS) - set(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if data.columns.duplicated().any():
            raise ValueError("Duplicate column names are not supported")

    def load_features(self, path: str | Path) -> pd.DataFrame:
        data = pd.read_csv(Path(self.config.data_dir) / path)
        self.validate_features(data)
        if data["id"].isna().any() or data["id"].duplicated().any():
            raise ValueError("Well IDs must be present and unique in dataset files")
        return data

    def load_training(self) -> tuple[pd.DataFrame, pd.Series]:
        features = self.load_features(self.config.train_file)
        labels = pd.read_csv(Path(self.config.data_dir) / self.config.labels_file)
        if not {"id", "status_group"}.issubset(labels.columns):
            raise ValueError("Labels require id and status_group columns")
        if labels["id"].isna().any() or labels["id"].duplicated().any():
            raise ValueError("Label IDs must be present and unique")
        target = features[["id"]].merge(
            labels[["id", "status_group"]], on="id", how="left",
            sort=False, validate="one_to_one",
        )["status_group"]
        if not target.isin(TARGET_CLASSES).all():
            raise ValueError("Missing or unrecognized training labels")
        target.index = features.index
        return features, target
