"""Population access under a ground-truth-informed successful-repair scenario."""
import numpy as np
from rasterio.warp import transform as transform_coordinates
from sklearn.neighbors import BallTree


EARTH_RADIUS_M = 6_371_000


def build_population_access_df(population_df, holdout_results_df,
                               population_crs="EPSG:4326", nearby_radius_m=5_000):
    """Return nearby population pixels and functional well IDs/distances.

    Filter by distance to ANY holdout well, counting each pixel once. Baseline
    availability uses actual functional wells. Intervention adds actual
    nonfunctional wells predicted nonfunctional, assumed successfully repaired.
    Other statuses do not qualify. Only supplied holdout wells are considered.

    x/y remain in population_crs; distances are spherical straight-line meters
    from pixel centers. Inputs and original pixel indices are preserved.
    With no available functional wells, distance is infinity and well ID is NA.
    """
    longitude, latitude = transform_coordinates(
        population_crs, "EPSG:4326", population_df["x"].to_numpy(),
        population_df["y"].to_numpy(),
    )
    coordinates = np.radians(np.column_stack([latitude, longitude]))
    holdout_tree = BallTree(
        np.radians(holdout_results_df[["latitude", "longitude"]].to_numpy()),
        metric="haversine",
    )
    distance, _ = holdout_tree.query(coordinates, k=1)
    nearby = distance[:, 0] * EARTH_RADIUS_M <= nearby_radius_m
    access_df = population_df.loc[nearby, ["x", "y", "population"]].copy()
    coordinates = coordinates[nearby]

    already_functional = holdout_results_df["actual_status"].eq("functional")
    repaired = (holdout_results_df["actual_status"].eq("non functional")
                & holdout_results_df["predicted_status"].eq("non functional"))
    scenarios = [
        (already_functional | repaired,
         "nearest_functional_well_id", "distance_to_nearest_functional_well_m"),
        (already_functional,
         "nearest_functional_well_without_intervention_id",
         "distance_to_nearest_functional_well_without_intervention_m"),
    ]
    for available, id_column, distance_column in scenarios:
        wells = holdout_results_df.loc[available]
        if wells.empty or access_df.empty:
            access_df[id_column] = np.nan
            access_df[distance_column] = np.inf
            continue
        tree = BallTree(np.radians(wells[["latitude", "longitude"]].to_numpy()),
                        metric="haversine")
        distance, position = tree.query(coordinates, k=1)
        access_df[id_column] = wells["id"].to_numpy()[position[:, 0]]
        access_df[distance_column] = distance[:, 0] * EARTH_RADIUS_M
    return access_df


def calculate_access_kpis(access_df, access_distance_m=2_500):
    """Summarize population-weighted access on a fixed set of raster pixels.

    At or within the distance threshold counts as access. Coverage percentages
    use only the population in access_df, not the whole administrative region.
    Gains assume the successful-repair scenario used to build the dataframe.
    Changing this threshold does not recalculate nearest wells.
    """
    population = access_df["population"].astype(float)
    before = access_df["distance_to_nearest_functional_well_without_intervention_m"].le(access_distance_m)
    after = access_df["distance_to_nearest_functional_well_m"].le(access_distance_m)
    total = float(population.sum())
    with_access_before = float(population.loc[before].sum())
    with_access_after = float(population.loc[after].sum())
    coverage_before = 100 * with_access_before / total if total else np.nan
    coverage_after = 100 * with_access_after / total if total else np.nan
    return {
        "population_total": total,
        "population_with_access_before": with_access_before,
        "population_without_access_before": total - with_access_before,
        "population_with_access_after": with_access_after,
        "population_without_access_after": total - with_access_after,
        "population_gaining_access": float(population.loc[after & ~before].sum()),
        "coverage_before_pct": coverage_before,
        "coverage_after_pct": coverage_after,
        "coverage_improvement_pp": coverage_after - coverage_before,
    }
