# Tanzania water access: prioritizing pump repairs

\<write here on the client's repair-planning decision, the limited budget, and the objective of increasing population access to working water points>

## The proposal at a glance

\<write here on what the proof of concept delivers, who would use it, the main finding, and the decision it helps the client make>

\<image placeholder: dashboard overview showing the regional map, scenario controls, and KPI comparison>

## From a repair budget to an actionable plan

\<write here on selecting a region, entering a number of pump checks or a repair budget, setting service and repair costs, comparing scenarios, and downloading the ranked repair-plan CSV>

\<write here on the difference between checks and successful repairs, and the CSV's well identifiers, locations, priority ranks, and expected additional people gaining access>

## Why predicting broken pumps is only part of the problem

\<write here on the initial confidence-first strategy, its comparison against equal-budget random selection, and why identifying failures does not necessarily maximize additional water access>

\<write here on prioritizing failure probability multiplied by currently unserved population, accounting for overlapping catchments and earlier selections>

## Evidence and results

### Classification performance on unseen wells and an unseen region

\<write here on current validation and geographic-holdout results, random and majority classification baselines, and nonfunctional-class precision–recall performance>

\<image or table placeholder: selected results from notebooks/02_model_evaluation.ipynb, with split names, metric definitions, and evaluation settings>

### Water-access gains under the same repair constraints

\<write here on confidence-first, random, and expected-access strategies under the same number of checks, the evaluated region, access threshold, and population scope>

\<write here on realized additional people gaining access, total people with access, improvement over random, and the distinction between expected ranking scores and realized simulated outcomes>

\<image placeholder: strategy comparison from notebook 03 or the dashboard, showing gains and total access with matching settings>

\<write here on whether the observed improvement is consistent beyond this example, the use of one seeded random realization, and what cannot yet be concluded>

## How water access is estimated

### Access definition and travel-time context

\<write here on why collection time matters, the relevance of walking, and the evidence required before making claims about vehicle or bicycle availability>

\<write here on the distinction between a 30-minute collection round trip and the configurable 2.5 km one-way straight-line distance proxy; include the implied walking time, return journey, queues, terrain, and routes before equating them>

\<source placeholder: UNICEF collection-time context — https://www.unicef.org/press-releases/unicef-collecting-water-often-colossal-waste-time-women-and-girls>

\<image placeholder: access-distance illustration or notebook distance distribution with the chosen threshold clearly marked>

### Geographic holdout and the Morogoro demonstration

\<write here on reserving a complete region from model fitting and cross-validation, evaluating generalization to an unseen region, and keeping training-dependent preprocessing inside each training split>

\<write here on the Morogoro choice, the PCA visualization, and equal-weight regional centroid similarity calculated in the full transformed feature space rather than only the two plotted components>

\<image placeholder: PCA of well features colored by region from notebook 03>

\<image placeholder: highlighted regional profile and full-feature similarity ranking from notebook 03>

\<write here on why an incomplete pump inventory in a randomized subset can distort nearest-working-well distances; distinguish this inventory-coverage problem from training/test leakage, and explain the remaining cross-boundary inventory limitation>

### Population scopes and before/after scenarios

\<write here on raster pixel centers and population weights, the fixed 2.5 km catchment around any candidate pump, the alternative full administrative-region population, and avoiding duplicate population counting where catchments overlap>

\<write here on the no-repair baseline, adding selected truly nonfunctional pumps under an assumed successful repair, treating false positives as checks without additional access, and comparing against random checks on the same candidate pool>

\<write here on nearest functional well IDs and distances before and after intervention, summing population within the inclusive access threshold, and keeping the population denominator fixed within each comparison>

\<image placeholder: before/after nearest-functional-well distance distribution for Morogoro from notebook 03; identify whether the vertical axis counts pixels or people>

## Data and geographic context

### Pump records, population raster, and administrative boundaries

\<write here on the pump dataset and target classes, population-raster provider and version, reference years, spatial resolution, units, boundary source, licenses, and how each input is used>

\<source placeholder: pump dataset and license>

\<source placeholder: population TIFF provider, download location, methodology, and license>

\<source placeholder: pinned Tanzania ADM1 boundaries — https://github.com/wmgeolab/geoBoundaries/blob/9469f09/releaseData/gbOpen/TZA/ADM1/geoBoundaries-TZA-ADM1.geojson>

### Population plausibility check: Dar es Salaam

\<write here on the approximately 5.6 million population sum within 20 km of the selected pump, the coordinates and raster year, and its use as a plausibility check rather than a validated regional total>

\<write here on the external city-population estimate, its source and reference year, the difference between a circular catchment and administrative or metropolitan boundaries, and why the totals need not match>

\<image placeholder: cited city population and density reference, with date and geographic definition visible>

\<image placeholder: GPS map of the selected Dar es Salaam location and its 20 km catchment>

\<image placeholder: satellite view of the same location and catchment>

\<image placeholder: street-level photograph of the same area, with location and image attribution>

## Technical implementation

### Modeling and feature engineering

\<write here on CSV label alignment by well ID, well age and missingness handling, training-reference neighbor counts, categorical encoding, XGBoost, and external population sums at 1, 2, 3, 5, 10, and 20 km>

### Intervention selection and evaluation

\<write here on greedy expected marginal access gains, the known-working-well inventory assumption, overlap handling, seeded random selection, spatial nearest-neighbor queries, and reusable population-access dataframes>

### Precomputation and interactive scenarios

\<write here on separate geographic-holdout predictions for each region, the CSV cache, Streamlit's scenario-result cache, and which calculations rerun when a threshold or repair count changes>

\<write here on manually refreshing disk caches when inputs or model settings change, and distinguish cached predictions from recalculated intervention selections>

### Project layout and notebook reading order

\<write here on src/water_wells for modeling and KPIs, src/dashboard for caching and the application, tests for validation, and the role of externally supplied data and generated artifacts>

\<write here on notebooks/01_early_exploration_draft.ipynb as the historical pre-implementation prototype and its evaluation limitations>

\<write here on notebooks/02_model_evaluation.ipynb as the evaluation of the current reusable pipeline>

\<write here on notebooks/03_geographic_features_and_access_experiments.ipynb as the population, regional-holdout, and intervention-strategy experiments>

## Run the project

### 1. Prepare the Python environment

\<instruction placeholder: supported Python version, creating and activating a virtual environment, installing the editable package with dashboard and development extras, and running commands from the repository root>

### 2. Obtain and place the input data

\<instruction placeholder: source links and required local paths for training features, labels, population TIFF, and Tanzania regional boundaries; identify files excluded from version control>

### 3. Build or refresh dashboard caches

\<instruction placeholder: run the dashboard.cache_data module, demonstrate caching one named region before all regions, describe the refresh option, expected runtime, output location, and when existing CSVs are reused>

### 4. Launch the dashboard and generate a repair plan

\<instruction placeholder: the Streamlit launch command for src/dashboard/app.py, opening the local app, selecting a cached region, setting repair constraints, interpreting the views, and downloading the plan>

### 5. Reproduce the notebook analyses

\<instruction placeholder: start Jupyter, select the project environment, follow the numbered notebook order, run cells from a fresh session, and identify the historical prototype versus current evaluation>

### 6. Run verification

\<instruction placeholder: run the standard pytest suite, explain the opt-in full-training test and its environment variable, and state the dependencies and compute needed for full-data execution>

## Assumptions, limitations, and next steps

### Interpretation of the proof of concept

\<write here on simulated rather than observed repair outcomes, assumed repair success, known functional status as an input, probability calibration, and geographic evaluation beyond a single region>

\<write here on straight-line distance versus practical access, pump capacity and reliability, seasonality and water quality, population and pump-record reference-year differences, and boundary or coordinate uncertainty>

\<write here on the current reservation of repair cost plus service-call fee for every selected pump, its distinction from charging repair costs only on confirmed failures, and the absence of routing or sequential budget reallocation>

### Path to operational use

\<write here on the client's next validation step, field verification, a complete current pump inventory, realistic travel-time modeling, repair outcomes and costs, repeated random comparisons, and expansion to more regions>

\<write here on the decision this proof of concept supports today and the evidence required before using its estimates as operational commitments>
