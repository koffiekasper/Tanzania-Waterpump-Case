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
    assert access.empty and len(access.columns) == 9
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


def test_population_region_assignment_preserves_pixels_and_resolves_shared_boundaries():
    import geopandas as gpd
    from shapely.geometry import box
    from water_wells.kpi_metrics import assign_population_regions

    population = pd.DataFrame({"x": [.5, 1., 1.5, 3.], "y": [.5] * 4,
                               "population": [10., 20., 30., 40.]}, index=[5, 5, 9, 2])
    original = population.copy(deep=True)
    boundaries = gpd.GeoDataFrame({"shapeName": ["Beta", "Alpha"]},
                                  geometry=[box(1, 0, 2, 1), box(0, 0, 1, 1)], crs="EPSG:4326")
    result = assign_population_regions(population, boundaries)
    assert_frame_equal(result[original.columns], original)
    assert result.region.iloc[:3].tolist() == ["Alpha", "Alpha", "Beta"]
    assert pd.isna(result.region.iloc[3])
    assert result.within_tanzania.tolist() == [True, True, True, False]
    assert result.population.sum() == 100
    assert_frame_equal(result, assign_population_regions(population, boundaries.iloc[::-1]))
    assert_frame_equal(result, assign_population_regions(population, boundaries.to_crs("EPSG:3857")))


def test_dual_scope_preserves_nearby_kpi_and_counts_remote_population(monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    from water_wells.kpi_metrics import assign_population_regions
    import water_wells.kpi_metrics as module

    population, wells = sample_data()
    # The distant 500 people are INSIDE Morogoro; add people outside Tanzania.
    population = pd.concat([population, pd.DataFrame(
        {"x": [38.], "y": [-6.], "population": [900.]}, index=[80],
    )])
    boundaries = gpd.GeoDataFrame({"shapeName": ["Morogoro"]},
                                  geometry=[box(34.9, -6.1, 36.1, -5.9)], crs="EPSG:4326")
    annotated = assign_population_regions(population, boundaries)
    old = build_population_access_df(population, wells, selected_well_ids=[502], nearby_radius_m=2500)
    original_tree = module.BallTree
    queries = []

    class CountingTree:
        def __init__(self, *args, **kwargs):
            self.tree = original_tree(*args, **kwargs)

        def query(self, coordinates, **kwargs):
            queries.append(len(coordinates))
            return self.tree.query(coordinates, **kwargs)

    monkeypatch.setattr(module, "BallTree", CountingTree)
    access = build_population_access_df(annotated, wells, region_names=["Morogoro"],
                                        selected_well_ids=[502], nearby_radius_m=2500)
    # Candidate, before, after: one query each, not one set per scope.
    assert queries == [len(population)] * 3
    assert_frame_equal(access[annotated.columns], annotated)
    summaries = calculate_access_kpis(access)
    assert summaries["nearby_population"] == calculate_access_kpis(old)
    assert summaries["nearby_population"]["population_total"] == 100
    assert summaries["full_region"]["population_total"] == 600
    assert summaries["full_region"]["population_without_access_after"] == 500
    assert summaries["full_region"]["coverage_after_pct"] < summaries["nearby_population"]["coverage_after_pct"]
    assert access.loc[80, "within_region"] == False


def test_random_selection_is_seeded_uniform_and_ignores_truth_and_probabilities():
    from water_wells.kpi_metrics import select_interventions

    candidates = pd.DataFrame({"id": range(20), "probability_non_functional": np.linspace(0, 1, 20),
                               "actual_status": ["functional"] * 20})
    selected = select_interventions(candidates, budget_fraction=.25, strategy="random", random_state=7)
    assert len(selected) == 5 and selected.id.is_unique
    changed = candidates.iloc[::-1].copy()
    changed["actual_status"] = "non functional"
    changed["probability_non_functional"] = 0
    repeat = select_interventions(changed, budget_n=5, strategy="random", random_state=7)
    assert selected.id.tolist() == repeat.id.tolist()
    assert selected.id.tolist() != select_interventions(candidates, budget_n=5, strategy="random", random_state=8).id.tolist()
    assert select_interventions(candidates, budget_n=0, strategy="random").empty
    assert len(select_interventions(candidates, budget_n=100, strategy="random")) == 20
    # Larger budgets extend the same seeded sequence rather than resampling it.
    assert selected.id.tolist() == select_interventions(candidates, budget_n=10, strategy="random", random_state=7).id.iloc[:5].tolist()


def test_random_intervention_metrics_share_baseline_and_can_outperform_model():
    population, wells = sample_data()
    access = build_population_access_df(population, wells, selected_well_ids=[502],
                                        random_selected_well_ids=[703])
    metrics = calculate_access_kpis(access, access_distance_m=500)
    assert metrics["population_with_access_before"] == 10
    assert metrics["population_gaining_access"] == 20
    assert metrics["population_gaining_access_random"] == 30
    assert metrics["population_with_access_random"] == 40
    assert metrics["population_without_access_random"] == 60
    assert metrics["coverage_random_pct"] == 40
    assert metrics["coverage_improvement_random_pp"] == 30
    assert metrics["population_gain_over_random"] == -10
    assert access.nearest_functional_well_with_random_interventions_id.tolist() == [901, 901, 703, 703]


def test_random_false_positives_and_zero_budget_add_no_access():
    population, wells = sample_data()
    for selected_ids in [[], [901], [104]]:
        access = build_population_access_df(population, wells, selected_well_ids=[502],
                                            random_selected_well_ids=selected_ids)
        metrics = calculate_access_kpis(access, access_distance_m=500)
        assert metrics["population_gaining_access_random"] == 0
        assert metrics["coverage_random_pct"] == metrics["coverage_before_pct"]
        np.testing.assert_array_equal(
            access.distance_to_nearest_functional_well_with_random_interventions_m,
            access.distance_to_nearest_functional_well_without_intervention_m,
        )


def test_random_and_model_share_full_budget_and_both_geographic_scopes(monkeypatch):
    import water_wells.kpi_metrics as module
    population, wells = sample_data()
    population["region"] = "Morogoro"
    original_tree = module.BallTree
    query_sizes = []

    class CountingTree:
        def __init__(self, *args, **kwargs):
            self.tree = original_tree(*args, **kwargs)

        def query(self, coordinates, **kwargs):
            query_sizes.append(len(coordinates))
            return self.tree.query(coordinates, **kwargs)

    monkeypatch.setattr(module, "BallTree", CountingTree)
    access = build_population_access_df(population, wells, region_names=["Morogoro"],
                                        selected_well_ids=wells.id, random_selected_well_ids=wells.id)
    assert query_sizes == [len(population)] * 4
    metrics = calculate_access_kpis(access, access_distance_m=500)
    assert metrics["nearby_population"]["population_total"] == 100
    assert metrics["full_region"]["population_total"] == 600
    for summary in metrics.values():
        assert summary["population_gain_over_random"] == 0
        assert summary["population_gaining_access_random"] == summary["population_gaining_access"]
        assert summary["coverage_random_pct"] == summary["coverage_after_pct"]


def test_access_selector_accounts_for_probability_overlap_and_known_baseline():
    from water_wells.kpi_metrics import select_access_interventions

    population = pd.DataFrame({"x": [35., 35.1, 35.2, 36.], "y": [-6.] * 4,
                               "population": [1000., 100., 100., 10000.]})
    candidates = pd.DataFrame({"id": [30, 10, 20], "latitude": [-6.] * 3,
                               "longitude": [35.2, 35.1, 35.1],
                               "probability_non_functional": [.7, .9, .8]})
    baseline = pd.DataFrame({"latitude": [-6.], "longitude": [35.]})
    original_population, original_candidates = population.copy(), candidates.copy()
    selected = select_access_interventions(population, candidates, baseline, budget_n=3)
    # After selecting ID 10, the overlapping ID 20 yields only 0.8 * 10,
    # while ID 30 serves another 100 with expected gain 70.
    assert selected.id.tolist() == [10, 30, 20]
    np.testing.assert_allclose(selected.expected_population_gain, [90., 70., 8.])
    np.testing.assert_allclose(selected.cumulative_expected_population_gain, [90., 160., 168.])
    assert selected.selection_rank.tolist() == [1, 2, 3]
    assert_frame_equal(population, original_population)
    assert_frame_equal(candidates, original_candidates)
    # Labels are ignored even if passed accidentally as additional columns.
    changed = candidates.iloc[::-1].assign(actual_status="non functional", predicted_status="functional")
    again = select_access_interventions(population, changed, baseline, budget_fraction=1.)
    assert again.id.tolist() == selected.id.tolist()
    np.testing.assert_allclose(again.expected_population_gain, selected.expected_population_gain)


def test_access_selector_prefers_population_benefit_over_raw_confidence():
    from water_wells.kpi_metrics import select_access_interventions

    population = pd.DataFrame({"x": [35., 35.1, 35.2], "y": [-6.] * 3,
                               "population": [1000., 10., 100.]})
    candidates = pd.DataFrame({"id": [1, 2, 3], "latitude": [-6.] * 3,
                               "longitude": [35., 35.1, 35.2],
                               "probability_non_functional": [.999, .99, .7]})
    baseline = pd.DataFrame({"latitude": [-6.], "longitude": [35.]})
    selected = select_access_interventions(population, candidates, baseline, budget_n=1)
    assert selected.id.tolist() == [3]
    assert selected.expected_population_gain.iloc[0] == 70


def test_access_selector_zero_budget_no_baseline_and_deterministic_ties():
    from water_wells.kpi_metrics import select_access_interventions

    population = pd.DataFrame({"x": [35.], "y": [-6.], "population": [100.]})
    candidates = pd.DataFrame({"id": [2, 1], "latitude": [-6.] * 2,
                               "longitude": [35.] * 2, "probability_non_functional": [.5, .5]})
    no_baseline = pd.DataFrame(columns=["latitude", "longitude"])
    selected = select_access_interventions(population, candidates, no_baseline, budget_n=100)
    assert selected.id.tolist() == [1, 2]
    np.testing.assert_allclose(selected.expected_population_gain, [50., 25.])
    zero = select_access_interventions(population, candidates, no_baseline, budget_n=0)
    assert zero.empty and "expected_population_gain" in zero
    baseline = pd.DataFrame({"latitude": [-6.], "longitude": [35.]})
    covered = select_access_interventions(population, candidates, baseline, budget_fraction=1)
    assert covered.id.tolist() == [1, 2]
    assert covered.expected_population_gain.eq(0).all()
