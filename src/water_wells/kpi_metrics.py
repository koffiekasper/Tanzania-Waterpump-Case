"""Population access under a ground-truth-informed successful-repair scenario."""
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.warp import transform as transform_coordinates
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree


EARTH_RADIUS_M = 6_371_000


def select_interventions(pred_proba, budget_n=None, budget_fraction=None,
                         strategy="confidence", random_state=42):
    """Select pump checks by confidence or random sampling without replacement.

    strategy="random" samples all candidates with a reproducible seed; neither
    probabilities nor ground truth influence the random selection. Sorting IDs
    first keeps the random selection stable if input rows are reordered.
    Confidence selection uses id and probability_non_functional. Pass either budget_n
    or budget_fraction (0 to 1). Fractions use ALL candidate rows and round down;
    counts larger than the candidate pool select all rows. Ties use ID ascending.
    Selection never uses actual_status or predicted_status. The returned rows
    retain their original columns and indices; input predictions are unchanged.
    """
    n = budget_n if budget_n is not None else int(len(pred_proba) * budget_fraction)
    ranked = (pred_proba.sort_values("id").sample(frac=1, random_state=random_state)
              if strategy == "random" else pred_proba.sort_values(
                  ["probability_non_functional", "id"], ascending=[False, True],
              ))
    return ranked.head(n).copy()


def select_access_interventions(population_df, candidates_df, functional_wells_df,
                                budget_n=None, budget_fraction=None,
                                population_crs="EPSG:4326", access_distance_m=2_500):
    """Greedily maximize expected additional population access within a budget.

    Candidates need id, latitude, longitude, probability_non_functional only.
    functional_wells_df is an explicit KNOWN working-well inventory, not inferred
    from candidate labels. The caller supplies the population area to optimize.
    No candidate ground truth or predicted_status is read.

    Score each candidate as probability_non_functional times still-unserved
    population it can reach. After selecting a candidate, multiply residual
    population weights in its catchment by (1 - probability_non_functional).
    This accounts for overlap and uncertain repair benefit, assuming independent
    well outcomes, usable probabilities, and successful repair when nonfunctional.
    Coverage relationships are computed once and reused across greedy steps.
    Ties use ID ascending. Zero-gain checks fill any remaining budget.
    Returns selection order, marginal and cumulative expected population gains;
    these are estimates, not the realized ground-truth KPI.
    """
    candidates = candidates_df.sort_values("id")
    n = budget_n if budget_n is not None else int(len(candidates) * budget_fraction)
    n = min(n, len(candidates))
    if n == 0:
        return candidates.iloc[:0].assign(selection_rank=np.array([], dtype=int),
                                           expected_population_gain=np.array([], dtype=float),
                                           cumulative_expected_population_gain=np.array([], dtype=float))

    longitude, latitude = transform_coordinates(
        population_crs, "EPSG:4326", population_df["x"].to_numpy(),
        population_df["y"].to_numpy(),
    )
    coordinates = np.radians(np.column_stack([latitude, longitude]))
    baseline_distance = np.full(len(population_df), np.inf)
    if not functional_wells_df.empty and len(population_df):
        baseline_tree = BallTree(
            np.radians(functional_wells_df[["latitude", "longitude"]].to_numpy()),
            metric="haversine",
        )
        baseline_distance = baseline_tree.query(coordinates, k=1)[0][:, 0] * EARTH_RADIUS_M
    unserved = (baseline_distance > access_distance_m) & (population_df["population"].to_numpy() > 0)
    remaining_population = population_df["population"].to_numpy(dtype=float)[unserved].copy()
    coverage = csr_matrix((len(candidates), len(remaining_population)))
    if len(remaining_population):
        population_tree = BallTree(coordinates[unserved], metric="haversine")
        neighborhoods = population_tree.query_radius(
            np.radians(candidates[["latitude", "longitude"]].to_numpy()),
            r=access_distance_m / EARTH_RADIUS_M,
        )
        indptr = np.r_[0, np.cumsum([len(points) for points in neighborhoods])]
        indices = np.concatenate(neighborhoods)
        coverage = csr_matrix((np.ones(len(indices)), indices, indptr),
                              shape=(len(candidates), len(remaining_population)))

    probabilities = candidates["probability_non_functional"].to_numpy(dtype=float)
    chosen = np.zeros(len(candidates), dtype=bool)
    positions, gains = [], []
    for _ in range(n):
        scores = probabilities * (coverage @ remaining_population)
        scores[chosen] = -np.inf
        position = int(np.argmax(scores))
        positions.append(position)
        gains.append(float(scores[position]))
        chosen[position] = True
        pixels = coverage.indices[coverage.indptr[position]:coverage.indptr[position + 1]]
        remaining_population[pixels] *= 1 - probabilities[position]
    result = candidates.iloc[positions].copy()
    result["selection_rank"] = np.arange(1, n + 1)
    result["expected_population_gain"] = gains
    result["cumulative_expected_population_gain"] = np.cumsum(gains)
    return result


def assign_population_regions(population_df, boundaries, population_crs="EPSG:4326",
                              region_column="shapeName"):
    """Annotate pixels without dropping/duplicating rows or changing x/y values.

    boundaries must contain Tanzania's regional polygons. Pixels on a shared
    boundary are assigned to the alphabetically first matching region. Unmatched
    pixels have missing region and within_tanzania=False. Original indices,
    including repeated indices, are preserved.
    """
    points = gpd.GeoDataFrame(
        {"pixel_position": np.arange(len(population_df))},
        geometry=gpd.points_from_xy(population_df["x"], population_df["y"]),
        crs=population_crs,
    ).to_crs(boundaries.crs)
    polygons = boundaries[[region_column, "geometry"]].rename(columns={region_column: "region"})
    matches = gpd.sjoin(points, polygons, how="left", predicate="intersects")
    matches = matches.sort_values(["pixel_position", "region"]).drop_duplicates("pixel_position")
    regions = matches.set_index("pixel_position")["region"].reindex(range(len(population_df)))
    result = population_df.copy()
    result["region"] = regions.to_numpy()
    result["within_tanzania"] = regions.notna().to_numpy()
    return result


def build_population_access_df(population_df, holdout_results_df,
                               population_crs="EPSG:4326", nearby_radius_m=5_000,
                               selected_well_ids=None, region_names=None,
                               random_selected_well_ids=None):
    """Calculate access scenarios once, with nearby and regional scope flags.

    With region_names, pass population annotated by assign_population_regions:
    retain every pixel and mark within_region for the requested names. Without
    region_names, keep the original nearby-only behavior. within_candidate_radius
    always marks distance to ANY candidate well, independent of selected repairs.
    Nearby scope can extend past the regional boundary, as in the original KPI.
    Baseline
    availability uses actual functional wells. Intervention adds actual
    nonfunctional wells selected for intervention, assumed successfully repaired.
    Pass selected_well_ids from select_interventions for a budget-based scenario;
    without it, use the original predicted_status rule. An empty selection means
    no repairs. False positives consume selection budget but add no repaired well.
    Other statuses do not qualify. Only supplied holdout wells are considered.
    Optional random_selected_well_ids adds a random-repair scenario with the same
    baseline and pixel scopes. Select the same number of candidates for fairness;
    random checks of wells not actually nonfunctional still consume the budget.

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
    keep = nearby if region_names is None else np.ones(len(population_df), dtype=bool)
    access_df = population_df.loc[keep].copy()
    coordinates = coordinates[keep]
    access_df["distance_to_nearest_candidate_well_m"] = distance[keep, 0] * EARTH_RADIUS_M
    access_df["within_candidate_radius"] = nearby[keep]
    if region_names is not None:
        access_df["within_region"] = access_df["region"].isin(region_names)

    already_functional = holdout_results_df["actual_status"].eq("functional")
    selected = (holdout_results_df["predicted_status"].eq("non functional")
                if selected_well_ids is None
                else holdout_results_df["id"].isin(selected_well_ids))
    repaired = holdout_results_df["actual_status"].eq("non functional") & selected
    scenarios = [
        (already_functional | repaired,
         "nearest_functional_well_id", "distance_to_nearest_functional_well_m"),
        (already_functional,
         "nearest_functional_well_without_intervention_id",
         "distance_to_nearest_functional_well_without_intervention_m"),
    ]
    if random_selected_well_ids is not None:
        random_repaired = (holdout_results_df["actual_status"].eq("non functional")
                           & holdout_results_df["id"].isin(random_selected_well_ids))
        scenarios.append((
            already_functional | random_repaired,
            "nearest_functional_well_with_random_interventions_id",
            "distance_to_nearest_functional_well_with_random_interventions_m",
        ))
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
    """Summarize access without repeating spatial calculations.

    With within_region, return nearby_population and full_region summaries.
    Nearby scope preserves the original candidate catchment definition; regional
    scope includes remote pixels inside the requested administrative boundaries.
    Both use the same access threshold (inclusive) and intervention scenario.
    Without regional scope, return the original single-population summary.
    Coverage is undefined (NaN) when a scope has zero population. If random
    intervention distances are present, also report random access/gains and the
    population gain difference between model-selected and random interventions.
    A seeded random run is one realization, not an average over repeated trials.
    """
    regional = "within_region" in access_df
    scopes = {
        "nearby_population": access_df.loc[access_df["within_candidate_radius"]],
        "full_region": access_df.loc[access_df["within_region"]],
    } if regional else {"nearby_population": access_df}
    results = {}
    for scope, frame in scopes.items():
        population = frame["population"].astype(float)
        before = frame["distance_to_nearest_functional_well_without_intervention_m"].le(access_distance_m)
        after = frame["distance_to_nearest_functional_well_m"].le(access_distance_m)
        total = float(population.sum())
        with_access_before = float(population.loc[before].sum())
        with_access_after = float(population.loc[after].sum())
        coverage_before = 100 * with_access_before / total if total else np.nan
        coverage_after = 100 * with_access_after / total if total else np.nan
        results[scope] = {
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
        random_column = "distance_to_nearest_functional_well_with_random_interventions_m"
        if random_column in frame:
            random_access = frame[random_column].le(access_distance_m)
            with_access_random = float(population.loc[random_access].sum())
            random_gain = float(population.loc[random_access & ~before].sum())
            coverage_random = 100 * with_access_random / total if total else np.nan
            results[scope].update({
                "population_with_access_random": with_access_random,
                "population_without_access_random": total - with_access_random,
                "population_gaining_access_random": random_gain,
                "coverage_random_pct": coverage_random,
                "coverage_improvement_random_pp": coverage_random - coverage_before,
                "population_gain_over_random": results[scope]["population_gaining_access"] - random_gain,
            })
    return results if regional else results["nearby_population"]
