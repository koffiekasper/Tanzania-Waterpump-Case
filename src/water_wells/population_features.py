"""Population pixel-center sums with a coordinate-based disk cache."""
from pathlib import Path
import hashlib
import os
import tempfile

import joblib
import numpy as np
import pandas as pd

POPULATION_COLUMNS = {
    1: "Population_1km", 2: "Population_2km", 3: "Population_3km",
    5: "population_5km", 10: "population_10km", 20: "population_20km",
}
METRIC_CRS = "+proj=aeqd +lat_0=-6 +lon_0=35 +datum=WGS84 +units=m +no_defs"


class PopulationFeatures:
    """Enrich raw wells, preserving population, order, index and other columns.

    auto: reuse cached coordinates and write missing ones; load: require cache;
    write: recompute and save requested coordinates; off: compute without disk.
    Raster path/size/mtime and algorithm version identify each cache. No labels
    or training-dependent neighbor features are cached. Cache files are trusted
    local joblib artifacts. Distances use an approximate local metric projection.
    """

    def __init__(self, raster_path, cache_dir="data/processed/population",
                 data_mode="auto", chunk_size=500):
        self.raster_path = raster_path
        self.cache_dir = cache_dir
        self.data_mode = data_mode
        self.chunk_size = chunk_size

    def _cache_path(self):
        path = Path(self.raster_path).resolve()
        stat = path.stat()
        signature = f"v1|{path}|{stat.st_size}|{stat.st_mtime_ns}|{METRIC_CRS}"
        key = hashlib.sha256(signature.encode()).hexdigest()
        return Path(self.cache_dir) / f"{key}.joblib"

    def add_features(self, X):
        if self.data_mode not in {"auto", "load", "write", "off"}:
            raise ValueError("data_mode must be auto, load, write, or off")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        coordinates = X[["longitude", "latitude"]]
        valid = (coordinates.longitude.between(-180, 180)
                 & coordinates.latitude.between(-90, 90)).to_numpy()
        keys = pd.MultiIndex.from_frame(coordinates.iloc[np.flatnonzero(valid)])
        unique = keys.unique()
        columns = list(POPULATION_COLUMNS.values())
        cache = pd.DataFrame(index=unique[:0], columns=columns, dtype=float)
        path = None
        if self.data_mode != "off":
            path = self._cache_path()
            if path.exists():
                cache = joblib.load(path)
        missing = unique if self.data_mode in {"write", "off"} else unique.difference(cache.index)
        if len(missing) and self.data_mode == "load":
            raise ValueError(f"Population cache missing {len(missing)} coordinates; use data_mode='auto' or 'write'.")
        if len(missing):
            computed = self._compute(missing)
            cache = pd.concat([cache.loc[~cache.index.isin(missing)], computed])
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic replacement prevents partially written cache files.
                fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
                os.close(fd)
                try:
                    joblib.dump(cache, temporary)
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        result = X.copy()
        values = np.full((len(X), len(columns)), np.nan)
        values[valid] = cache.reindex(keys).to_numpy()
        result[columns] = values
        return result

    def _compute(self, coordinates):
        import geopandas as gpd
        import rasterio

        with rasterio.open(self.raster_path) as raster:
            band = raster.read(1, masked=True)
            rows, cols = np.where(~np.ma.getmaskarray(band) & np.isfinite(band.data))
            x, y = rasterio.transform.xy(raster.transform, rows, cols)
            pixels = gpd.GeoDataFrame(
                {"pixel_population": band.data[rows, cols].astype(float)},
                geometry=gpd.points_from_xy(x, y), crs=raster.crs,
            ).to_crs(METRIC_CRS)
        wells = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                coordinates.get_level_values("longitude"),
                coordinates.get_level_values("latitude"),
            ), crs="EPSG:4326",
        ).to_crs(METRIC_CRS)
        totals = pd.DataFrame(0.0, index=coordinates, columns=list(POPULATION_COLUMNS.values()))
        for radius, column in POPULATION_COLUMNS.items():
            for start in range(0, len(wells), self.chunk_size):
                chunk = wells.iloc[start:start + self.chunk_size]
                buffers = gpd.GeoDataFrame(
                    {"well_row": chunk.index.to_numpy()},
                    geometry=chunk.geometry.buffer(radius * 1000, quad_segs=64).to_numpy(),
                    crs=METRIC_CRS,
                )
                # Query the reusable pixel index with buffers, rather than scanning
                # the full raster against a new buffer index for every chunk.
                matches = gpd.sjoin(buffers, pixels, how="inner", predicate="intersects")
                sums = matches.groupby("well_row")["pixel_population"].sum()
                totals.iloc[sums.index.to_numpy(dtype=int), totals.columns.get_loc(column)] = sums.to_numpy()
        return totals
