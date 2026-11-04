"""Load competition CSV files without relying on row alignment."""
from pathlib import Path
import pandas as pd
from .config import DataConfig


TARGET_CLASSES = ("functional", "functional needs repair", "non functional")


class DataLoader:
    """Paths are relative to config.data_dir; explicit paths override defaults."""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    def load_features(self, path: str | Path) -> pd.DataFrame:
        return pd.read_csv(Path(self.config.data_dir) / path)

    def load_training(self) -> tuple[pd.DataFrame, pd.Series]:
        features = self.load_features(self.config.train_file)
        labels = pd.read_csv(Path(self.config.data_dir) / self.config.labels_file)
        target = features[["id"]].merge(
            labels[["id", "status_group"]], on="id", how="left",
            sort=False,
        )["status_group"]
        target.index = features.index
        return features, target
