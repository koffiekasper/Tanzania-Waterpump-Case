"""Well age, training-fitted neighbor counts, and cached population features."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import BallTree
from sklearn.utils.validation import check_is_fitted
from .data_loader import DataLoader
from .population_features import PopulationFeatures


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Count neighbors in the fitted training reference, excluding the same ID.

    Unknown/invalid coordinates yield NaN counts. Longitude zero is retained,
    as in the notebook; no new geographic sentinel policy is introduced.
    """

    def __init__(self, radii_km=(3, 5, 10, 20), population_raster=None,
                 cache_dir="data/processed/population", data_mode="auto"):
        self.radii_km = radii_km
        self.population_raster = population_raster
        self.cache_dir = cache_dir
        self.data_mode = data_mode

    def prepare_population(self, X):
        """Enrich/cache external population sums without fitting well references.

        Safe to call before splitting: this step learns nothing from well labels
        or neighboring training rows. All other features are built in transform.
        """
        if getattr(self, "population_raster", None) is None:
            return X.copy()
        return PopulationFeatures(
            self.population_raster, self.cache_dir, self.data_mode,
        ).fit_transform(X)

    @staticmethod
    def _valid(data):
        return (data["latitude"].between(-90, 90)
                & data["longitude"].between(-180, 180)).to_numpy()

    def fit(self, X: pd.DataFrame, y=None):
        DataLoader.validate_features(X)
        if X["id"].isna().any() or X["id"].duplicated().any():
            raise ValueError("Training reference IDs must be present and unique")
        valid = self._valid(X)
        self.reference_ = X.loc[valid, ["id", "latitude", "longitude"]].copy()
        self.tree_ = (BallTree(np.radians(self.reference_[["latitude", "longitude"]]),
                               metric="haversine") if valid.any() else None)
        self.reference_by_id_ = self.reference_.set_index("id")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "reference_")
        DataLoader.validate_features(X)
        data = self.prepare_population(X)
        valid = self._valid(data)
        # Exclude a reference well only when ID and coordinates match. This
        # keeps single-row and batch inference identical and avoids subtracting
        # a fictitious self from unseen wells.
        matching = self.reference_by_id_.reindex(data["id"])
        same = (matching["latitude"].to_numpy() == data["latitude"].to_numpy()) & (
            matching["longitude"].to_numpy() == data["longitude"].to_numpy())
        for radius in self.radii_km:
            counts = np.full(len(data), np.nan)
            if self.tree_ is not None and valid.any():
                counts[valid] = self.tree_.query_radius(
                    np.radians(data.loc[valid, ["latitude", "longitude"]]),
                    r=radius / 6371.0, count_only=True,
                ) - same[valid].astype(int)
            data[f"nearby_wells_{radius}km"] = counts
        years = pd.to_datetime(data["date_recorded"], errors="coerce").dt.year
        age = years - data["construction_year"]
        data["well_age"] = age.mask((data["construction_year"] == 0) | (age < 0)).astype(float)
        data["construction_year_missing"] = (data["construction_year"] == 0).astype(int)
        data["construction_year"] = data["construction_year"].replace(0, np.nan).astype(float)
        data["wpt_name"] = data["wpt_name"].fillna("none")
        return data.drop(columns=["id", "date_recorded", "subvillage", "status_group"],
                         errors="ignore")
