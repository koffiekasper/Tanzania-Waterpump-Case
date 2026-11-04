from water_wells.kpi_metrics import (
    build_population_access_df, select_access_interventions, select_interventions,
)


def compare_strategies(population, predictions, n_repairs, access_distance_m=2500):
    working_wells = predictions.loc[
        predictions["actual_status"].eq("functional"), ["id", "latitude", "longitude"],
    ]
    candidates = predictions[["id", "latitude", "longitude", "probability_non_functional"]]
    population = population.assign(x=population["longitude"], y=population["latitude"])
    selected = select_access_interventions(
        population.loc[population["within_region"]], candidates, working_wells,
        budget_n=n_repairs, access_distance_m=access_distance_m,
    )
    random_selected = select_interventions(
        predictions, budget_n=len(selected), strategy="random", random_state=42,
    )
    access = build_population_access_df(
        population, predictions, selected_well_ids=selected["id"],
        random_selected_well_ids=random_selected["id"],
        region_names=predictions["region"].unique().tolist(), nearby_radius_m=2500,
    )
    return access, selected, random_selected
