import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from sklearn.neighbors import BallTree

from water_wells import DataConfig, DataLoader, TrainingConfig, TrainingPipeline
from water_wells.kpi_metrics import (
    EARTH_RADIUS_M, assign_population_regions, build_population_access_df,
)


REGION_FILES = ("predictions.csv", "validation_predictions.csv", "splits.csv",
                "baseline_access.csv", "feature_importance.csv")


def write_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def cache_exists(directory, filenames, refresh=False):
    return not refresh and all((directory / name).exists() for name in filenames)


def prepare_population(raster_path, boundaries_path, output_dir, refresh=False):
    directory = output_dir / "shared"
    directory.mkdir(parents=True, exist_ok=True)
    if cache_exists(directory, ["population.csv", "boundaries.csv"], refresh):
        print("Loading shared population cache", flush=True)
        return pd.read_csv(directory / "population.csv")
    print("Reading raster and assigning population pixels to regions...", flush=True)
    with rasterio.open(raster_path) as raster:
        band = raster.read(1, masked=True)
        rows, cols = np.where(~np.ma.getmaskarray(band) & np.isfinite(band.data))
        x, y = rasterio.transform.xy(raster.transform, rows, cols)
        longitude, latitude = transform(raster.crs, "EPSG:4326", x, y)
        population = pd.DataFrame({
            "pixel_id": rows.astype(np.int64) * raster.width + cols,
            "x": x, "y": y, "population": band.data[rows, cols].astype(float),
            "longitude": longitude, "latitude": latitude,
        })
        population_crs = raster.crs.to_string()
    boundaries = gpd.read_file(boundaries_path)
    population = assign_population_regions(population, boundaries, population_crs)
    boundary_table = pd.DataFrame({
        "region": boundaries["shapeName"],
        "geometry_wkt": boundaries.to_crs("EPSG:4326").geometry.to_wkt(),
    })
    write_csv(population, directory / "population.csv")
    write_csv(boundary_table, directory / "boundaries.csv")
    return population


def prepare_region(region, X, y, population, population_tree, raster_path,
                   output_dir, config, nearby_radius_m,
                   population_crs, refresh=False):
    slug = re.sub(r"[^a-z0-9]+", "_", region.lower()).strip("_")
    directory = output_dir / "regions" / slug
    directory.mkdir(parents=True, exist_ok=True)
    if cache_exists(directory, REGION_FILES, refresh):
        print(f"{region}: reusing CSV cache", flush=True)
        return
    print(f"{region}: fitting geographic-holdout model...", flush=True)
    pipeline = TrainingPipeline(config, population_raster=raster_path,
                                cache_dir=output_dir.parent / "population", data_mode="auto")
    X_train, X_valid, X_holdout, y_train, y_valid, y_holdout = pipeline.split(
        X, y, holdout_regions=[region],
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict_wells(X_holdout)
    predictions["actual_status"] = y_holdout
    predictions["region"] = X_holdout["region"]
    validation = pipeline.predict_wells(X_valid)
    validation["actual_status"] = y_valid
    validation["region"] = X_valid["region"]
    splits = pd.concat([
        X_train[["id"]].assign(split="train"),
        X_valid[["id"]].assign(split="validation"),
        X_holdout[["id"]].assign(split="geographic_holdout"),
    ])


    neighbors = population_tree.query_radius(
        np.radians(predictions[["latitude", "longitude"]].to_numpy()),
        r=nearby_radius_m / EARTH_RADIUS_M,
    )
    keep = population["region"].eq(region).to_numpy().copy()
    keep[np.unique(np.concatenate(neighbors))] = True
    scope_population = population.iloc[np.flatnonzero(keep)]
    baseline = build_population_access_df(
        scope_population, predictions, population_crs=population_crs,
        nearby_radius_m=nearby_radius_m, selected_well_ids=[], region_names=[region],
    )


    baseline = baseline.drop(columns=["nearest_functional_well_id",
                                       "distance_to_nearest_functional_well_m"])
    importance = pd.DataFrame({
        "feature": pipeline.preprocessor.get_feature_names_out(),
        "importance": pipeline.classifier.feature_importances_,
    }).sort_values("importance", ascending=False)
    for name, frame in zip(REGION_FILES, [predictions, validation, splits, baseline, importance]):
        write_csv(frame, directory / name)
    print(f"{region}: saved {len(predictions):,} candidates and {len(baseline):,} population pixels", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cache dashboard CSVs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--regions", nargs="+", help="Region names; default: all regions. Quote names with spaces.")
    parser.add_argument("--nearby-radius-m", type=float, default=2500)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = args.output_dir or root / "data/processed/dashboard"
    output.mkdir(parents=True, exist_ok=True)
    raster = root / "tza_pop_2026_CN_1km_R2025A_UA_v1.tif"
    boundaries = root / "data/raw/geoBoundaries-TZA-ADM1.geojson"
    loader = DataLoader(DataConfig(data_dir=root))
    X, y = loader.load_training()
    config = TrainingConfig(random_seed=args.random_seed)
    config.xgb_params["n_estimators"] = args.n_estimators
    population = prepare_population(raster, boundaries, output, args.refresh)
    with rasterio.open(raster) as dataset:
        population_crs = dataset.crs.to_string()
    wells = X.copy()
    wells["actual_status"] = y
    write_csv(wells, output / "shared/wells.csv")
    population_tree = BallTree(np.radians(population[["latitude", "longitude"]].to_numpy()), metric="haversine")
    all_regions = sorted(X.region.unique())
    regions = args.regions or all_regions
    for region in regions:
        prepare_region(region, X, y, population, population_tree, raster, output, config,
                       args.nearby_radius_m, population_crs, args.refresh)

    catalog = []
    for name in all_regions:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        directory = output / "regions" / slug
        if cache_exists(directory, REGION_FILES):
            catalog.append({"region": name, "directory": f"regions/{slug}"})
    write_csv(pd.DataFrame(catalog), output / "regions.csv")
    print(f"Dashboard CSV cache ready: {output}", flush=True)


if __name__ == "__main__":
    main()
