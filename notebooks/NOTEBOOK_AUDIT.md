# Original notebook audit

Inspected every source cell and the saved training/evaluation outputs in `tanzania_water_notebook.ipynb` before implementing the package. Original files remain in place.

| Cells | Logic and dependencies | Refactored treatment |
|---|---|---|
| 0–19 | Imports; hardcoded root CSV filenames; global `data`, `labels`, `drop_columns`; schema/missing-value exploration; missing well names filled with `none`. Longitude/latitude string feature is commented out. `quantity_group` appended to drop list. | DataLoader/config; well-name fill in FeatureEngineer. Exploratory displays excluded. Drop list was never applied. |
| 20–24 | `count_nearby_wells` builds BallTree over all available coordinates; four global assignments create radius counts before splitting. Plot distributions. | Stateful FeatureEngineer fit/transform using training reference; same radii and haversine metric. Invalid/missing coordinates produce NaN counts. Plots remain reference analysis only. |
| 25–29 | ID checks; append ID to unused drop list; merge target on ID. | Loader validates unique IDs and aligns labels; ID removed from model features. |
| 30–32 | Parse recorded date; age = recorded year minus construction year; zero construction years/negative age yield missing age; zero-year indicator; convert zero years to NaN. | FeatureEngineer preserves formulas and indicator. |
| 33–45 | Missing-value inspection, global `geo_cols`; `fill_unknown_funders` and `fill_unknown_installers` return filled series but exploratory calls do not assign them. | UnknownImputer implements the same fills in the actual preprocessing pipeline. |
| 46–51 | KNN geographic subvillage imputation, `data_new`, and holdout accuracy helper; depends on valid coordinate rows and known subvillage labels. K sweep commented out. Narrative concludes inference unreliable. | Experimental branch excluded: `data_new` never used by final training. |
| 52–61 | Name `impute_subvillage` redefined as constant fill. Its cell calls `impute_public_meeting` before definition, so a clean sequential execution fails. Remaining `impute_*` helpers fill `Unknown`; permit demo calls scheme-name helper, with equivalent fill behavior. | Constant imputer replaces the six active helpers. Subvillage ultimately dropped, so its fill is unnecessary. No notebook execution order dependencies. |
| 62–72 | Global date sort/reset; date and subvillage dropped; categorical exploration. | Drop date after age derivation and drop subvillage; preserve row order instead of global sort. |
| 73–77 | Lambda FunctionTransformers depend on global helper functions; booleans converted to strings only for public meeting/permit; OHE factory with frequency 20/category cap 50; six custom branches, other categoricals and numeric passthrough; LabelEncoder; fit ColumnTransformer on ALL rows; feature-name displays. | Serializable class-based transformers; same fills, OHE settings and numeric passthrough. Train-only fit. Label mapping in classifier. Exposed fitted ColumnTransformer and feature names. |
| 78–80 | Split already transformed data 75/25, shuffle, seed 42; 500-tree XGBoost multiclass model; classification report and confusion matrix use global encoder. | Raw stratified split first; composed fitted pipeline; preserved XGBoost parameters; separate evaluator using readable labels. |
| 81–85 | Importance plotting helper depends on passed model/names; invoked with 500 features; target inspection and population distribution plot. | Evaluator returns importance DataFrame and configurable top-N plot; descriptive population exploration excluded from production. |

## Behavioral implications

The notebook's saved transformed shape is `(59400, 491)`. Its classification report rounds accuracy to 0.79. No exact unrounded score or independent untouched holdout was saved.

Two unsupervised operations see validation inputs in the notebook: encoder fitting and geographic reference construction. Neither uses target labels, but both violate the requested training-only boundary and overstate independence of validation. The refactoring fits both on training only. Geographical counts for validation therefore use a smaller reference. The full-data final model can use all labeled wells as its reference when refitted after evaluation.

`id` and `quantity_group` were never dropped despite the global list: retain quantity group; remove ID as an arbitrary identifier with no stable inference meaning. Construction year, missing indicator and age all remain. No new geographic correction is guessed for zero longitude. Numerical NaNs go directly to XGBoost as before. The final code adds no KNN, median fill, scaling or feature selection.

Only public meeting/permit stringify to reconcile mixed boolean and string values, exactly as the original active pipeline does. Unknown categorical values remain governed by sklearn's `infrequent_if_exist`: if no infrequent bucket was learned for a feature they encode as all zero, sometimes with a sklearn warning.

Probability names use the original target strings in `predict_proba` and stable output column names in `predict_wells`. Artifacts include all inference state; they do not reference notebook globals or lambdas.

## Verification status

- Available data: 59,400 training wells, two supplied feature/label CSVs; no separate unlabeled competition test CSV.
- Original saved notebook output inspected; original notebook was not executed or modified.
- All new Python files and experimentation code cells compile successfully (stdlib checks).
- Unit tests and opt-in full-data integration test supplied.
- Runtime dependency installation was attempted in both `.venv` and the user's root `bin/python` environment. PyPI DNS/network access is unavailable in the execution sandbox. Both environments initially lacked pandas, numpy, sklearn, XGBoost, matplotlib, joblib and pytest.
- Confirmed using the user-created root virtual environment (`bin/python`): `python -m pytest -q` fails with `No module named pytest`. Editable installation fails resolving setuptools from PyPI. Full training cannot execute without the modeling dependencies; no corrected validation score or saved trained artifact is claimed.
- Final stdlib verification: syntax valid for all 12 Python source/test files and 11 notebook code cells; pyproject TOML and src-layout declaration valid. Both supplied CSVs contain 59,400 unique IDs and their ID sets match; targets contain exactly the three expected classes.
