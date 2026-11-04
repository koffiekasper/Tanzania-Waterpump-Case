"""Small deterministic tests; no competition data required."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from water_wells import (
    DataConfig, DataLoader, DataPreprocessor, FeatureEngineer,
    ModelEvaluator, TrainingConfig, TrainingPipeline,
)
from water_wells.data_loader import TARGET_CLASSES
from water_wells.preprocessing import UnknownImputer


RAW_COLUMNS = tuple("""
    id amount_tsh date_recorded funder gps_height installer longitude latitude
    wpt_name num_private basin subvillage region region_code district_code lga
    ward population public_meeting recorded_by scheme_management scheme_name
    permit construction_year extraction_type extraction_type_group
    extraction_type_class management management_group payment payment_type
    water_quality quality_group quantity quantity_group source source_type
    source_class waterpoint_type waterpoint_type_group
""".split())
@pytest.fixture
def wells():
    n = 36
    data = pd.DataFrame({column: ["standard"] * n for column in RAW_COLUMNS})
    for column in ("amount_tsh", "gps_height", "num_private", "region_code",
                   "district_code", "population"):
        data[column] = np.arange(n)
    data["id"] = np.arange(n) + 100
    data["date_recorded"] = "2013-01-01"
    data["construction_year"] = 2000
    data["latitude"] = -6.0 - np.arange(n) * 0.001
    data["longitude"] = 35.0 + np.arange(n) * 0.001
    data["public_meeting"] = pd.Series([True, False, None] * 12, dtype=object)
    data["permit"] = data["public_meeting"].copy()
    data.loc[0, ["funder", "installer", "scheme_name", "scheme_management"]] = None
    data.loc[0, "construction_year"] = 0
    data.loc[1, "construction_year"] = 2020
    data.loc[2, "wpt_name"] = None
    data.index = np.arange(n) * 3 + 7
    return data


@pytest.fixture
def config():
    return TrainingConfig(min_frequency=2, max_categories=5,
                          xgb_params={"n_estimators": 3, "max_depth": 2, "n_jobs": 1})


@pytest.fixture
def target(wells):
    return pd.Series(list(TARGET_CLASSES) * 12, index=wells.index)


def test_features_and_input_preservation(wells):
    original = wells.copy(deep=True)
    engineer = FeatureEngineer().fit(wells.iloc[:3])
    transformed = engineer.transform(wells)
    assert_frame_equal(wells, original)
    assert transformed.index.equals(wells.index)
    assert transformed.iloc[0]["construction_year_missing"] == 1
    assert np.isnan(transformed.iloc[0]["construction_year"])
    assert np.isnan(transformed.iloc[1]["well_age"])
    assert transformed.iloc[2]["well_age"] == 13
    assert transformed.iloc[2]["wpt_name"] == "none"
    assert not {"id", "date_recorded", "subvillage"}.intersection(transformed.columns)
    assert "quantity_group" in transformed
    assert transformed.iloc[0]["nearby_wells_3km"] == 2
    assert transformed.iloc[3]["nearby_wells_3km"] == 3
    assert_frame_equal(engineer.transform(wells.iloc[[3]]), transformed.iloc[[3]])
    assert len(engineer.reference_) == 3


def test_invalid_and_empty_geographic_reference(wells):
    invalid = wells.iloc[:2].copy()
    invalid["latitude"] = np.nan
    engineer = FeatureEngineer().fit(invalid)
    assert engineer.transform(wells)["nearby_wells_3km"].isna().all()
    engineer.fit(wells)
    invalid["latitude"] = 100
    assert engineer.transform(invalid)["nearby_wells_5km"].isna().all()


def test_unknown_imputer():
    data = pd.DataFrame({"permit": [True, False, None]})
    imputer = UnknownImputer(stringify=True)
    result = imputer.fit_transform(data)
    assert result["permit"].tolist() == ["True", "False", "Unknown"]
    assert data["permit"].iloc[2] is None
    assert imputer.get_feature_names_out().tolist() == ["permit"]


def test_preprocessor_unseen_missing_and_dimensions(wells):
    engineered = FeatureEngineer().fit_transform(wells)
    processor = DataPreprocessor(min_frequency=2, max_categories=5)
    training = processor.fit_transform(engineered)
    new = engineered.iloc[[5]].copy()
    new["funder"] = "never seen before"
    new["permit"] = None
    transformed = processor.transform(new)
    assert transformed.shape == (1, training.shape[1])
    assert len(processor.get_feature_names_out()) == training.shape[1]
    encoder = processor.column_transformer_.named_transformers_["funder"].named_steps["encode"]
    assert "never seen before" not in encoder.categories_[0]
    # Numerical missing values intentionally remain for XGBoost.
    assert np.isnan(engineered.iloc[0]["construction_year"])


def test_predictions_serialization_and_duplicate_indices(wells, target, config, tmp_path):
    pipeline = TrainingPipeline(config).fit(wells, target)
    new = wells.iloc[[8, 2, 7]].copy()
    new.index = [4, 4, 1]
    new["funder"] = "unseen funder"
    original = new.copy(deep=True)
    probabilities = pipeline.predict_proba(new)
    assert list(probabilities.columns) == list(TARGET_CLASSES)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert set(pipeline.predict(new)).issubset(TARGET_CLASSES)
    results = pipeline.predict_wells(new)
    assert results["id"].tolist() == new["id"].tolist()
    assert results.index.equals(new.index)
    assert results["predicted_status"].tolist() == pipeline.predict(new).tolist()
    np.testing.assert_allclose(results["probability_needs_repair"],
                               probabilities["functional needs repair"])
    path = tmp_path / "pipeline.joblib"
    pipeline.save(path)
    loaded = TrainingPipeline.load(path)
    assert_frame_equal(results, loaded.predict_wells(new))
    assert_frame_equal(original, new)
    # Batch composition cannot change geographic features or predictions.
    assert_frame_equal(loaded.predict_wells(new.iloc[[0]]), results.iloc[[0]])
    assert len(ModelEvaluator().feature_importance(pipeline)) == len(
        pipeline.preprocessor.get_feature_names_out())


def test_split_and_sklearn_cross_validation(wells, target, config):
    pipeline = clone(TrainingPipeline(config))
    X_train, X_valid, X_holdout, y_train, y_valid, y_holdout = pipeline.split(wells, target)
    pipeline.fit(X_train, y_train)
    reference = pipeline.pipeline_.named_steps["features"].reference_
    assert set(reference["id"]).isdisjoint(X_valid["id"])
    assert not hasattr(TrainingPipeline(config), "pipeline_")
    scores = cross_val_score(TrainingPipeline(config), wells, target,
                            cv=StratifiedKFold(2), scoring="accuracy")
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    metrics = ModelEvaluator().metrics(y_valid, pipeline.predict(X_valid))
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}


def test_loader_aligns_labels(wells, target, tmp_path):
    wells.to_csv(tmp_path / "values.csv", index=False)
    labels = pd.DataFrame({"id": wells["id"], "status_group": target}).iloc[::-1]
    labels.to_csv(tmp_path / "labels.csv", index=False)
    loader = DataLoader(DataConfig(tmp_path, "values.csv", "labels.csv"))
    X, y = loader.load_training()
    assert X["id"].tolist() == wells["id"].tolist()
    assert y.tolist() == target.tolist()
    assert_frame_equal(loader.load_features("values.csv"), X)


@pytest.fixture
def population_raster(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / "population.tif"
    values = np.array([[1, 2, -9999], [4, 8, 16], [32, 64, 128]], dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=3, width=3,
                       count=1, dtype=values.dtype, crs="EPSG:4326",
                       transform=from_origin(34.995, -5.995, 0.01, 0.01),
                       nodata=-9999) as raster:
        raster.write(values, 1)
    return path


def test_population_sums_cache_and_coordinate_changes(population_raster, tmp_path, monkeypatch):
    from water_wells import PopulationFeatures, POPULATION_COLUMNS
    data = pd.DataFrame({"longitude": [35.0, 35.0, np.nan],
                         "latitude": [-6.0, -6.0, -6.0],
                         "population": [99, 77, 55]}, index=[5, 5, 2])
    original = data.copy(deep=True)
    cache = tmp_path / "cache"
    processor = PopulationFeatures(population_raster, cache, "auto", chunk_size=1)
    result = processor.add_features(data)
    columns = list(POPULATION_COLUMNS.values())
    assert_frame_equal(data, original)
    assert result.index.equals(data.index)
    assert result.population.equals(data.population)
    assert result.iloc[0]["Population_1km"] == 1
    assert result.iloc[0]["population_20km"] == 255  # excludes NoData
    assert result.iloc[2][columns].isna().all()
    assert (np.diff(result.iloc[0][columns].to_numpy(dtype=float)) >= 0).all()
    assert len(list(cache.glob("*.joblib"))) == 1
    refreshed = PopulationFeatures(population_raster, cache, "write").add_features(data)
    assert_frame_equal(refreshed, result)
    uncached_dir = tmp_path / "no-cache"
    uncached = PopulationFeatures(population_raster, uncached_dir, "off").add_features(data)
    assert_frame_equal(uncached, result)
    assert not uncached_dir.exists()
    outside = data.iloc[[0]].copy()
    outside["longitude"] = 40.0
    outside_result = PopulationFeatures(population_raster, cache).add_features(outside)
    assert (outside_result[columns] == 0).all().all()

    def no_compute(*args):
        raise AssertionError("Cached coordinates should not recompute geospatial sums")

    monkeypatch.setattr(PopulationFeatures, "_compute", no_compute)
    reordered = data.iloc[[1, 0]]
    loaded = PopulationFeatures(population_raster, cache, "load").add_features(reordered)
    assert_frame_equal(loaded, result.iloc[[1, 0]])
    changed = data.iloc[[0]].copy()
    changed["longitude"] += 0.01
    with pytest.raises(KeyError):
        PopulationFeatures(population_raster, cache, "load").add_features(changed)
    # A changed raster cannot silently reuse old features.
    import os
    stat = population_raster.stat()
    os.utime(population_raster, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    with pytest.raises(KeyError):
        PopulationFeatures(population_raster, cache, "load").add_features(data)


def test_population_pipeline_cache_export_and_reload(wells, target, config,
                                                       population_raster, tmp_path, monkeypatch):
    from water_wells import PopulationFeatures, POPULATION_COLUMNS
    cache = tmp_path / "cache"
    pipeline = TrainingPipeline(config, population_raster, cache, "auto")
    output = tmp_path / "processed.csv"
    prepared = pipeline.prepare_population_data(wells, output_path=output)
    saved = pd.read_csv(output)
    assert saved.id.tolist() == wells.id.tolist()
    assert all(column in saved for column in POPULATION_COLUMNS.values())
    assert_frame_equal(prepared[wells.columns], wells)

    def no_compute(*args):
        raise AssertionError("Prewarmed full-data cache must serve splits and inference")

    monkeypatch.setattr(PopulationFeatures, "_compute", no_compute)
    pipeline = clone(TrainingPipeline(config, population_raster, cache, "load"))
    X_train, X_valid, X_holdout, y_train, y_valid, y_holdout = pipeline.split(wells, target)
    pipeline.fit(X_train, y_train)
    assert list(pipeline.pipeline_.named_steps) == ["features", "preprocessing", "classifier"]
    engineer = pipeline.pipeline_.named_steps["features"]
    engineered = engineer.transform(X_valid)
    assert all(column in engineered for column in POPULATION_COLUMNS.values())
    assert engineered.population.equals(X_valid.population)
    names = pipeline.preprocessor.get_feature_names_out()
    assert all(f"numerical__{column}" in names for column in POPULATION_COLUMNS.values())
    reference = pipeline.pipeline_.named_steps["features"].reference_
    assert set(reference.id).isdisjoint(X_valid.id)
    results = pipeline.predict_wells(X_valid)
    pipeline.save(tmp_path / "model.joblib")
    restored = TrainingPipeline.load(tmp_path / "model.joblib")
    assert_frame_equal(results, restored.predict_wells(X_valid))



def test_feature_engineer_population_in_standard_cycle(wells, population_raster, tmp_path):
    from water_wells import POPULATION_COLUMNS
    original = wells.copy(deep=True)
    engineer = clone(FeatureEngineer(
        population_raster=population_raster, cache_dir=tmp_path / "cache",
    ))
    result = engineer.fit_transform(wells)
    assert_frame_equal(wells, original)
    assert result.index.equals(wells.index)
    assert result.population.equals(wells.population)
    assert all(column in result for column in POPULATION_COLUMNS.values())
    assert "well_age" in result and "nearby_wells_3km" in result
    assert_frame_equal(engineer.transform(wells.iloc[[5]]), result.iloc[[5]])


@pytest.mark.parametrize("average", ["weighted", "macro", "micro", None])
def test_evaluator_summary_preserves_sklearn_metrics(average):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Include an absent true class and a predicted-only class.
    actual = [TARGET_CLASSES[0], TARGET_CLASSES[0], TARGET_CLASSES[2]]
    predicted = [TARGET_CLASSES[0], TARGET_CLASSES[1], TARGET_CLASSES[0]]
    expected = precision_recall_fscore_support(
        actual, predicted, labels=TARGET_CLASSES, average=average, zero_division=0,
    )
    result = ModelEvaluator().metrics(actual, predicted, average=average)
    assert result["accuracy"] == accuracy_score(actual, predicted)
    for name, value in zip(("precision", "recall", "f1"), expected[:3]):
        np.testing.assert_allclose(result[name], value)


def test_pipeline_owns_target_encoding_and_xgboost(wells, target, config):
    from xgboost import XGBClassifier

    pipeline = TrainingPipeline(config).fit(wells, target)
    assert isinstance(pipeline.classifier, XGBClassifier)
    assert pipeline.classes_.tolist() == list(TARGET_CLASSES)
    encoded = pipeline.pipeline_.predict(wells)
    assert set(encoded).issubset({0, 1, 2})
    np.testing.assert_array_equal(
        pipeline.predict(wells), pipeline.label_encoder_.inverse_transform(encoded.astype(int)),
    )
    probabilities = pipeline.predict_proba(wells)
    np.testing.assert_allclose(probabilities.to_numpy(), pipeline.classifier.predict_proba(
        pipeline.pipeline_[:-1].transform(wells),
    ))
    assert probabilities.columns.tolist() == list(TARGET_CLASSES)
    # Partial overrides retain the model defaults and configured seed.
    params = pipeline.classifier.get_params()
    assert params["n_estimators"] == 3
    assert params["learning_rate"] == 0.05
    assert params["random_state"] == config.random_seed
    assert params["objective"] == "multi:softprob"


@pytest.mark.parametrize("strategy", ["stratified", "uniform", "most_frequent"])
def test_evaluator_baseline_predictions_match_dummy_classifier(strategy):
    from sklearn.dummy import DummyClassifier
    from pandas.testing import assert_series_equal

    X_train = pd.DataFrame({"unused_feature": range(10)})
    y_train = pd.Series([TARGET_CLASSES[0]] * 7 + [TARGET_CLASSES[1]] * 2 + [TARGET_CLASSES[2]])
    X_valid = pd.DataFrame({"unused_feature": [100, 200, 300, 400]}, index=[7, 7, 2, 9])
    evaluator = ModelEvaluator()
    actual = evaluator.baseline_predictions(X_train, y_train, X_valid, strategy=strategy)
    expected = DummyClassifier(strategy=strategy, random_state=42).fit(X_train, y_train).predict(X_valid)
    np.testing.assert_array_equal(actual.to_numpy(), expected)
    assert actual.index.equals(X_valid.index)
    assert_series_equal(actual, evaluator.baseline_predictions(
        X_train, y_train, X_valid, strategy=strategy,
    ))
    # Baselines depend on training labels and validation row count, not feature values.
    assert_series_equal(actual, evaluator.baseline_predictions(
        X_train * -1, y_train, X_valid * -1, strategy=strategy,
    ))
    if strategy == "most_frequent":
        assert actual.eq(TARGET_CLASSES[0]).all()


@pytest.mark.parametrize("holdout_regions", [["Morogoro"], ["Morogoro", "Tanga"]])
def test_geographic_split_is_disjoint_aligned_and_train_only(wells, target, config, holdout_regions):
    data = wells.copy()
    data.loc[data.index[:9], "region"] = "Morogoro"
    data.loc[data.index[9:15], "region"] = "Tanga"
    original = data.copy(deep=True)
    pipeline = TrainingPipeline(config)
    X_train, X_valid, X_holdout, y_train, y_valid, y_holdout = pipeline.split(data, target, holdout_regions=holdout_regions)
    expected = data.region.isin(holdout_regions)
    assert_frame_equal(X_holdout, data.loc[expected])
    pd.testing.assert_series_equal(y_holdout, target.loc[expected])
    partitions = [X_train, X_valid, X_holdout]
    ids = [set(frame.id) for frame in partitions]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert set.union(*ids) == set(data.id)
    assert sum(map(len, partitions)) == len(data)
    assert len(X_valid) == int(np.ceil((~expected).sum() * config.validation_fraction))
    for frame, labels in [(X_train, y_train),
                          (X_valid, y_valid),
                          (X_holdout, y_holdout)]:
        pd.testing.assert_series_equal(labels, target.loc[frame.index])
    pipeline.fit(X_train, y_train)
    reference_ids = set(pipeline.pipeline_.named_steps['features'].reference_.id)
    assert reference_ids == ids[0]
    assert pipeline.predict_wells(X_holdout).index.equals(X_holdout.index)
    repeat_X_train, *_ = pipeline.split(data, target, holdout_regions=holdout_regions)
    assert_frame_equal(repeat_X_train, X_train)
    assert_frame_equal(data, original)


def test_split_without_holdout_preserves_original_split(wells, target, config):
    from sklearn.model_selection import train_test_split
    expected = train_test_split(wells, target, test_size=config.validation_fraction,
                                random_state=config.random_seed, stratify=target)
    X_train, X_valid, X_holdout, y_train, y_valid, y_holdout = TrainingPipeline(config).split(wells, target)
    assert_frame_equal(X_train, expected[0])
    assert_frame_equal(X_valid, expected[1])
    pd.testing.assert_series_equal(y_train, expected[2])
    pd.testing.assert_series_equal(y_valid, expected[3])
    assert_frame_equal(X_holdout, wells.iloc[:0])
    pd.testing.assert_series_equal(y_holdout, target.iloc[:0])
