"""Population-weighted access scenarios, independent of model training."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from rasterio.warp import transform
from sklearn.metrics.pairwise import haversine_distances

from water_wells.kpi_metrics import build_population_access_df, calculate_access_kpis


def sample_data():
    wells = pd.DataFrame({
        "id": [901, 502, 703, 104], "latitude": [-6.] * 4,
        "longitude": [35., 35.01, 35.02, 35.03],
        "actual_status": ["functional", "non functional", "non functional", "functional needs repair"],
        "predicted_status": ["non functional", "non functional", "functional", "non functional"],
    }, index=[30, 5, 99, 2])
    population = pd.DataFrame({
        "x": [35., 35.01, 35.02, 35.03, 36.], "y": [-6.] * 5,
        "population": [10., 20., 30., 40., 500.],
    }, index=[9, 3, 41, 12, 50])
    return population, wells


def test_access_dataframe_matches_direct_distances_and_preserves_inputs():
    population, wells = sample_data()
    original_population, original_wells = population.copy(), wells.copy()
    access = build_population_access_df(population, wells)
    assert access.index.tolist() == [9, 3, 41, 12]
    assert access.nearest_functional_well_id.tolist() == [901, 502, 502, 502]
    assert access.nearest_functional_well_without_intervention_id.tolist() == [901] * 4
    coords = np.radians(access[["y", "x"]])
    expected = haversine_distances(coords, np.radians(wells.iloc[:2][["latitude", "longitude"]])) * 6371000
    np.testing.assert_allclose(access.distance_to_nearest_functional_well_m, expected.min(axis=1))
    np.testing.assert_allclose(access.distance_to_nearest_functional_well_without_intervention_m, expected[:, 0])
    assert_frame_equal(population, original_population)
    assert_frame_equal(wells, original_wells)
    # No overlapping well buffers duplicate population pixels.
    assert access.population.sum() == 100


def test_access_kpis_use_population_weights_and_inclusive_threshold():
    access = pd.DataFrame({
        "population": [10., 20., 30., 40.],
        "distance_to_nearest_functional_well_without_intervention_m": [1000., 3000., 3000., np.inf],
        "distance_to_nearest_functional_well_m": [1000., 2000., 2500., np.inf],
    }, index=[3, 3, 10, 50])
    original = access.copy()
    metrics = calculate_access_kpis(access)
    assert metrics == {
        "population_total": 100.,
        "population_with_access_before": 10., "population_without_access_before": 90.,
        "population_with_access_after": 60., "population_without_access_after": 40.,
        "population_gaining_access": 50., "coverage_before_pct": 10.,
        "coverage_after_pct": 60., "coverage_improvement_pp": 50.,
    }
    assert calculate_access_kpis(access, access_distance_m=2000)["population_gaining_access"] == 20
    assert_frame_equal(access, original)


def test_access_reprojects_population_coordinates():
    population, wells = sample_data()
    expected = build_population_access_df(population, wells)
    projected = population.copy()
    projected["x"], projected["y"] = transform("EPSG:4326", "EPSG:3857", population.x.to_numpy(), population.y.to_numpy())
    actual = build_population_access_df(projected, wells, population_crs="EPSG:3857")
    assert actual.index.equals(expected.index)
    assert actual.nearest_functional_well_id.equals(expected.nearest_functional_well_id)
    np.testing.assert_allclose(actual.distance_to_nearest_functional_well_m,
                               expected.distance_to_nearest_functional_well_m, atol=1e-7)


def test_no_baseline_functional_wells_and_no_repaired_wells():
    population, wells = sample_data()
    wells["actual_status"] = "non functional"
    wells["predicted_status"] = "functional"
    unavailable = build_population_access_df(population, wells)
    assert np.isinf(unavailable.distance_to_nearest_functional_well_m).all()
    assert unavailable.nearest_functional_well_id.isna().all()
    assert calculate_access_kpis(unavailable)["population_without_access_after"] == 100
    wells.loc[wells.index[0], "predicted_status"] = "non functional"
    repaired = build_population_access_df(population, wells)
    assert np.isinf(repaired.distance_to_nearest_functional_well_without_intervention_m).all()
    assert repaired.nearest_functional_well_without_intervention_id.isna().all()
    assert calculate_access_kpis(repaired)["population_gaining_access"] > 0


def test_no_nearby_population_returns_empty_table_and_undefined_coverage():
    population, wells = sample_data()
    access = build_population_access_df(population.iloc[[-1]], wells)
    assert access.empty and len(access.columns) == 7
    metrics = calculate_access_kpis(access)
    assert metrics["population_total"] == 0
    assert metrics["population_gaining_access"] == 0
    assert np.isnan(metrics["coverage_before_pct"])


def test_budget_selection_counts_fractions_ties_and_ground_truth_independence():
    from water_wells.kpi_metrics import select_interventions

    candidates = pd.DataFrame({
        "id": [40, 30, 20, 10, 50],
        "probability_non_functional": [.2, .8, .8, .95, .1],
        "actual_status": ["functional"] * 5,
    }, index=[8, 2, 2, 7, 9])
    original = candidates.copy(deep=True)
    assert select_interventions(candidates, budget_n=2).id.tolist() == [10, 20]
    assert select_interventions(candidates, budget_fraction=.5).id.tolist() == [10, 20]
    assert select_interventions(candidates, budget_fraction=0).empty
    assert select_interventions(candidates, budget_n=0).empty
    assert len(select_interventions(candidates, budget_fraction=1)) == 5
    assert len(select_interventions(candidates, budget_n=100)) == 5
    shuffled = candidates.iloc[::-1].copy()
    shuffled["actual_status"] = "non functional"
    assert select_interventions(shuffled, budget_n=2).id.tolist() == [10, 20]
    assert_frame_equal(candidates, original)


def test_budget_false_positives_consume_slots_and_baseline_stays_fixed():
    from water_wells.kpi_metrics import select_interventions

    population, wells = sample_data()
    # Highest score is a repair-needed well: it consumes a slot but cannot count
    # as a successful repair under the existing nonfunctional-only scenario.
    wells["probability_non_functional"] = [.1, .8, .7, .95]
    baseline = build_population_access_df(population, wells, selected_well_ids=[])
    one_selected = select_interventions(wells, budget_n=1)
    assert one_selected.id.tolist() == [104]
    one = build_population_access_df(population, wells, selected_well_ids=one_selected.id)
    assert_frame_equal(one, baseline)
    assert calculate_access_kpis(one)["population_gaining_access"] == 0
    two_selected = select_interventions(wells, budget_n=2)
    two = build_population_access_df(population, wells, selected_well_ids=two_selected.id)
    assert_frame_equal(two[["x", "y", "population"]], baseline[["x", "y", "population"]])
    np.testing.assert_array_equal(
        two.distance_to_nearest_functional_well_without_intervention_m,
        baseline.distance_to_nearest_functional_well_without_intervention_m,
    )
    assert calculate_access_kpis(two, access_distance_m=500)["population_gaining_access"] == 20
    # Selection overrides the original multiclass prediction, including false negatives.
    selected_false_negative = build_population_access_df(population, wells, selected_well_ids=[703])
    assert 703 in selected_false_negative.nearest_functional_well_id.to_numpy()
    assert (two.distance_to_nearest_functional_well_m <= baseline.distance_to_nearest_functional_well_m).all()
