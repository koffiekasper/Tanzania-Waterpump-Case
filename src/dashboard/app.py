import importlib
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio
from rasterio.warp import transform
from shapely import wkt
import streamlit as st

from dashboard import scenario

importlib.reload(scenario)
from water_wells.kpi_metrics import calculate_access_kpis


st.set_page_config(page_title="Water access | Tanzania", page_icon="💧", layout="wide")
st.html((Path(__file__).with_name("style.css")).read_text())
st.session_state.setdefault("repairs_input", 100)
st.session_state.setdefault("repairs_slider", 100)
st.session_state.setdefault("map_collapsed", False)
st.session_state.setdefault("kpi_collapsed", False)
for key, default in [("strategy_view", "Before strategy"), ("map_color_mode", "Threshold bands")]:
    st.session_state[key] = st.session_state.get(key, default)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data/processed/dashboard"


@st.cache_data
def load_map_regions():
    regions = pd.read_csv(CACHE_DIR / "regions.csv")["region"].tolist()
    boundaries = pd.read_csv(CACHE_DIR / "shared/boundaries.csv").set_index("region")
    geometries = {region: wkt.loads(boundaries.loc[region, "geometry_wkt"])
                  for region in regions}
    return regions, geometries


@st.cache_data
def load_population_points(region, n_repairs, access_distance_m):
    catalog = pd.read_csv(CACHE_DIR / "regions.csv").set_index("region")
    directory = CACHE_DIR / catalog.loc[region, "directory"]
    population = pd.read_csv(directory / "baseline_access.csv")
    predictions = pd.read_csv(directory / "predictions.csv")
    access, selected, random_selected = scenario.compare_strategies(
        population, predictions, n_repairs, access_distance_m=access_distance_m,
    )
    raster_path = Path(__file__).resolve().parents[2] / "tza_pop_2026_CN_1km_R2025A_UA_v1.tif"
    with rasterio.open(raster_path) as raster:
        pixel_ids = access["pixel_id"].to_numpy()
        rows = (pixel_ids // raster.width)[:, None] + [0, 0, 1, 1]
        cols = (pixel_ids % raster.width)[:, None] + [0, 1, 1, 0]
        affine = raster.transform
        x = affine.a * cols + affine.b * rows + affine.c
        y = affine.d * cols + affine.e * rows + affine.f
        longitude, latitude = transform(raster.crs, "EPSG:4326", x.ravel(), y.ravel())
    access["cell_polygon"] = np.column_stack([longitude, latitude]).reshape(-1, 4, 2).tolist()
    return access, selected, random_selected


def sync_repairs(source):
    target = "repairs_input" if source == "repairs_slider" else "repairs_slider"
    st.session_state[target] = st.session_state[source]


def minimize_panel(panel):
    other = "kpi" if panel == "map" else "map"
    st.session_state[f"{panel}_collapsed"] = True
    st.session_state[f"{other}_collapsed"] = False


def maximize_panel(panel):
    other = "kpi" if panel == "map" else "map"
    st.session_state[f"{panel}_collapsed"] = False
    st.session_state[f"{other}_collapsed"] = not st.session_state[f"{other}_collapsed"]


def panel_header(title, panel):
    other = "kpi" if panel == "map" else "map"
    with st.container(key=f"{panel}_header"):
        heading, minimize, maximize = st.columns([4, 1.3, 1.3], gap="small", vertical_alignment="center")
        with heading:
            st.html(f'<div class="panel-title"><span class="section-dot"></span>{title}</div>')
        with minimize:
            st.button("Minimize", key=f"minimize_{panel}", help=f"Collapse {title.lower()} to the side",
                      on_click=minimize_panel, args=(panel,), width="stretch", type="tertiary")
        with maximize:
            expanded = st.session_state[f"{other}_collapsed"]
            st.button("Restore" if expanded else "Maximize", key=f"maximize_{panel}",
                      help="Restore split view" if expanded else f"Give {title.lower()} the available space",
                      on_click=maximize_panel, args=(panel,), width="stretch", type="tertiary")


def panel_rail(title, panel):
    with st.container(key=f"{panel}_rail"):
        st.button("›" if panel == "map" else "‹", key=f"restore_{panel}",
                  help=f"Restore {title.lower()}", width="stretch",
                  on_click=st.session_state.update, args=({f"{panel}_collapsed": False},))
        st.html(f'<div class="rail-label">{title}</div>')


def comparison_chart(random_value, strategy_value):
    data = pd.DataFrame({"Approach": ["Random", "Our approach"],
                         "Population": [random_value, strategy_value]})
    chart = alt.Chart(data).encode(
        x=alt.X("Approach:N", sort=["Random", "Our approach"], title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=10)),
        y=alt.Y("Population:Q", title="People", axis=alt.Axis(format="~s"),
                scale=alt.Scale(domain=[0, max(1, max(random_value, strategy_value) * 1.2)])),
        color=alt.Color("Approach:N", legend=None,
                        scale=alt.Scale(domain=["Random", "Our approach"],
                                        range=["#99AAB4", "#087F8C"])),
        tooltip=["Approach:N", alt.Tooltip("Population:Q", format=",.0f")],
    )
    bars = chart.mark_bar(size=36, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = chart.mark_text(dy=-9, fontSize=11).encode(
        text=alt.Text("Population:Q", format=",.0f"), color=alt.value("#334C59"),
    )
    return (bars + labels).properties(height=210).configure_view(strokeWidth=0)


def workspace(title="Water access overview"):
    regions, geometries = load_map_regions()
    st.session_state.setdefault("region", "Morogoro" if "Morogoro" in regions else regions[0])
    with st.container(key="intro"):
        st.caption("TANZANIA  /  WATER INFRASTRUCTURE")
        st.title(title)
        st.write("Explore a region and plan your next intervention.")

    widths = ([0.55, 9.65, 2.7] if st.session_state["map_collapsed"] else
              [9.65, 0.55, 2.7] if st.session_state["kpi_collapsed"] else [5.5, 4.7, 2.7])
    map_column, kpi_column, controls_column = st.columns(widths, gap="medium")
    with controls_column, st.container(key="controls_panel"):
        st.html('<div class="controls-eyebrow">YOUR SCENARIO</div>')
        st.subheader("Scenario settings")
        st.caption("Choose a region and set your repair budget.")
        region = st.selectbox("Region", regions, key="region")
        threshold_km = st.number_input(
            "Water access threshold (KM)", min_value=0.1, value=2.5, step=0.5,
            key="water_access_threshold_km",
            help="People at or within this distance of a working well have access. Also updates strategy selection. Choose threshold bands or a distance gradient above the map.",
        )
        population_scope = st.segmented_control(
            "Population scope", ["Full population", "Population near wells"],
            default="Full population", selection_mode="single", required=True,
            key="population_scope",
        )
        if population_scope == "Population near wells":
            st.caption("Fixed study area: within 2.5 km of any regional well, including across its border.")
        st.divider()
        st.number_input("N_repairs", min_value=0, max_value=1000,
                        step=1, key="repairs_input", on_change=sync_repairs,
                        args=("repairs_input",))
        st.slider("Number of repairs", min_value=0, max_value=1000,
                  key="repairs_slider", on_change=sync_repairs,
                  args=("repairs_slider",), label_visibility="collapsed")
        cost_per_repair = st.number_input(
            "Estimated cost per repair (TZS)", min_value=0., value=150_000.,
            step=10_000., format="%.0f", key="cost_per_repair_tzs",
        )
        st.html('<div class="controls-note">Retrospective repair simulation using cached predictions and known working-well records.</div>')

    access, selected, random_selected = load_population_points(
        region, st.session_state["repairs_input"], threshold_km * 1000,
    )
    scope_key = "full_region" if population_scope == "Full population" else "nearby_population"
    metrics = calculate_access_kpis(access, access_distance_m=threshold_km * 1000)[scope_key]
    scope_column = "within_region" if population_scope == "Full population" else "within_candidate_radius"
    points = access.loc[access[scope_column]].copy()
    west, south, east, north = geometries[region].bounds
    view = pdk.data_utils.compute_view([[west, south], [east, north]])
    view.zoom -= 0.5

    with map_column:
        if st.session_state["map_collapsed"]:
            panel_rail("Map view", "map")
        else:
            with st.container(key="map_panel"):
                panel_header("Map view", "map")
                with st.container(key="map_settings"):
                    strategy_view = st.segmented_control(
                        "Strategy view", ["Before strategy", "After strategy"],
                        selection_mode="single", required=True,
                        key="strategy_view",
                    )
                    color_mode = st.segmented_control(
                        "Map colors", ["Threshold bands", "Gradient"],
                        selection_mode="single", required=True,
                        key="map_color_mode",
                    )
                    if strategy_view == "Before strategy":
                        st.caption("Random sampling of pumps using the same repair budget (seed 42).")
                    else:
                        st.caption("Pumps prioritized for expected access gains across the full region.")
                distance_column = (
                    "distance_to_nearest_functional_well_with_random_interventions_m"
                    if strategy_view == "Before strategy" else "distance_to_nearest_functional_well_m"
                )
                points["distance_m"] = points[distance_column]
                points["distance_km"] = (points["distance_m"] / 1000).round(2)
                threshold_m = threshold_km * 1000
                distance_scale = np.clip((points["distance_m"].fillna(np.inf).to_numpy() - threshold_m) / (3 * threshold_m), 0, 1)
                if color_mode == "Threshold bands":
                    distance_scale = points["distance_m"].fillna(np.inf).gt(threshold_m).to_numpy(dtype=float)
                palette = np.array([[78, 112, 139], [255, 0, 0]])
                rgb = np.column_stack([
                    np.interp(distance_scale, [0, 1], palette[:, channel])
                    for channel in range(3)
                ]).round().astype(int)
                points["color"] = np.column_stack([rgb, np.full(len(points), 255)]).tolist()
                population_layer = pdk.Layer(
                    "PolygonLayer", id="population-distance", data=points,
                    get_polygon="cell_polygon", get_fill_color="color", opacity=1,
                    filled=True, stroked=False, pickable=True,
                )
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[population_layer], initial_view_state=view,
                        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                        map_provider="carto", tooltip={
                            "html": "<b>Population:</b> {population}<br/><b>Nearest functional well:</b> {distance_km} km",
                        },
                    ),
                    height=560, key=f"region_map_{region}",
                    alt=f"{population_scope} for {region}: raster cells colored by distance to a functional well",
                )
                if color_mode == "Threshold bands":
                    st.html(f'''<div class="map-legend">
                        <span>Distance to a functional well · {strategy_view.lower()}</span>
                        <div class="threshold-legend">
                            <span><i style="background:#4e708b"></i>Within {threshold_km:g} km</span>
                            <span><i style="background:#ff0000"></i>Above {threshold_km:g} km</span>
                        </div>
                    </div>''')
                else:
                    st.html(f'''<div class="map-legend">
                        <span>Distance to a functional well · {strategy_view.lower()}</span>
                        <div class="distance-gradient"></div>
                        <div class="distance-labels"><span>0 km</span><span>{threshold_km:g} km</span><span>{threshold_km * 4:g}+ km</span></div>
                        <p class="opacity-note">Grey-blue through {threshold_km:g} km; grey-blue to red up to {threshold_km * 4:g} km. Each cell is one population pixel.</p>
                    </div>''')


    with kpi_column:
        if st.session_state["kpi_collapsed"]:
            panel_rail("KPI overview", "kpi")
        else:
            with st.container(key="kpi_panel"):
                panel_header("KPI overview", "kpi")
                with st.container(key="kpi_content"):
                    st.caption(f"{population_scope} · {metrics['population_total']:,.0f} people · {len(selected):,} pump checks")
                    gain_chart, total_chart = st.columns(2, gap="small")
                    with gain_chart:
                        st.markdown("**Population gaining access**")
                        st.altair_chart(comparison_chart(
                            metrics["population_gaining_access_random"], metrics["population_gaining_access"],
                        ), width="stretch", key="gain_comparison")
                    with total_chart:
                        st.markdown("**Total population with access**")
                        st.altair_chart(comparison_chart(
                            metrics["population_with_access_random"], metrics["population_with_access_after"],
                        ), width="stretch", key="total_comparison")
                    st.caption(f"Already had access before any intervention: {metrics['population_with_access_before']:,.0f} people.")
                    random_gain = metrics["population_gaining_access_random"]
                    strategy_gain = metrics["population_gaining_access"]
                    relative_gain = f"{(strategy_gain / random_gain - 1) * 100:+.1f}% vs random" if random_gain > 0 else None
                    st.metric("Improvement over random baseline", f"{strategy_gain - random_gain:+,.0f} people",
                              delta=relative_gain)
                    st.divider()
                    st.markdown("**Intervention costs**")
                    total_cost = len(selected) * cost_per_repair
                    costs = pd.DataFrame({
                        "Random": [len(random_selected) * cost_per_repair, cost_per_repair,
                                   total_cost * 1000 / random_gain if random_gain > 0 else np.nan],
                        "Our approach": [total_cost, cost_per_repair,
                                         total_cost * 1000 / strategy_gain if strategy_gain > 0 else np.nan],
                    }, index=["Total TZS spent", "Cost per repair (TZS)", "TZS per 1,000 people gaining access"])
                    st.table(costs.style.format("{:,.0f}", na_rep="—"))
                    st.caption("Cost is applied to every pump check, including checks without a successful repair. Random uses seed 42.")


def page_one():
    workspace()


def page_two():
    workspace("Page 2")


def page_three():
    workspace("Page 3")


pages = [
    st.Page(page_one, title="Page 1", default=True),
    st.Page(page_two, title="Page 2", url_path="page-2"),
    st.Page(page_three, title="Page 3", url_path="page-3"),
]
page = st.navigation(pages, position="hidden")
with st.container(key="topbar"):
    brand, links = st.columns([3, 2], vertical_alignment="center")
    with brand:
        st.html('<div class="brand"><span class="brand-mark">◈</span><span>Water access<span class="brand-country">TANZANIA</span></span></div>')
    with links:
        link_columns = st.columns(3)
        for column, destination in zip(link_columns, pages):
            with column:
                st.page_link(destination, label=destination.title, width="stretch")
page.run()
