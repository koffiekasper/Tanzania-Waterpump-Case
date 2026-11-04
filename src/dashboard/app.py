from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio
from rasterio.warp import transform
from shapely import wkt
import streamlit as st

from dashboard.scenario import compare_strategies


st.set_page_config(page_title="Water access | Tanzania", page_icon="💧", layout="wide")
st.html((Path(__file__).with_name("style.css")).read_text())
st.session_state.setdefault("repairs_input", 100)
st.session_state.setdefault("repairs_slider", 100)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data/processed/dashboard"


@st.cache_data
def load_map_regions():
    regions = pd.read_csv(CACHE_DIR / "regions.csv")["region"].tolist()
    boundaries = pd.read_csv(CACHE_DIR / "shared/boundaries.csv").set_index("region")
    geometries = {region: wkt.loads(boundaries.loc[region, "geometry_wkt"])
                  for region in regions}
    return regions, geometries


@st.cache_data
def load_population_points(region, n_repairs):
    catalog = pd.read_csv(CACHE_DIR / "regions.csv").set_index("region")
    directory = CACHE_DIR / catalog.loc[region, "directory"]
    population = pd.read_csv(directory / "baseline_access.csv")
    predictions = pd.read_csv(directory / "predictions.csv")
    access, _, _ = compare_strategies(population, predictions, n_repairs)
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
    return access


def sync_repairs(source):
    target = "repairs_input" if source == "repairs_slider" else "repairs_slider"
    st.session_state[target] = st.session_state[source]


def workspace(title="Water access overview"):
    regions, geometries = load_map_regions()
    st.session_state.setdefault("region", "Morogoro" if "Morogoro" in regions else regions[0])
    region = st.session_state["region"]
    boundary = geometries[region]
    west, south, east, north = boundary.bounds
    view = pdk.data_utils.compute_view([[west, south], [east, north]])
    view.zoom -= 0.5
    with st.container(key="intro"):
        st.caption("TANZANIA  /  WATER INFRASTRUCTURE")
        st.title(title)
        st.write("Explore a region and plan your next intervention.")

    map_column, kpi_column, controls_column = st.columns([5.8, 3.2, 2.6], gap="medium")
    with map_column, st.container(key="map_panel"):
        st.html('<div class="panel-heading"><span class="section-dot"></span>Map view</div>')
        with st.container(key="map_settings"):
            distance_control, strategy_control = st.columns([1, 1.4])
            with distance_control:
                threshold_km = st.number_input(
                    "Distance threshold (km)", min_value=0.1, value=5.0, step=0.5,
                    key="map_distance_threshold_km", width=220,
                    help="Points at or beyond this distance are orange. White is halfway. This changes only the map color scale.",
                )
            with strategy_control:
                strategy_view = st.segmented_control(
                    "Strategy view", ["Before strategy", "After strategy"],
                    default="Before strategy", selection_mode="single", required=True,
                    key="strategy_view",
                )
            population_scope = st.segmented_control(
                "Population scope", ["Full population", "Population near wells"],
                default="Full population", selection_mode="single", required=True,
                key="population_scope",
            )
            if population_scope == "Population near wells":
                st.caption("Population within 2.5 km of any regional well, including across the regional border.")
            if strategy_view == "Before strategy":
                st.caption("Random sampling of pumps using the same repair budget (seed 42).")
            else:
                st.caption("Pumps selected to maximize expected population access within the same repair budget.")
        access = load_population_points(region, st.session_state["repairs_input"])
        scope_column = "within_region" if population_scope == "Full population" else "within_candidate_radius"
        points = access.loc[access[scope_column]].copy()
        distance_column = (
            "distance_to_nearest_functional_well_with_random_interventions_m"
            if strategy_view == "Before strategy" else "distance_to_nearest_functional_well_m"
        )
        points["distance_m"] = points[distance_column]
        points["distance_km"] = (points["distance_m"] / 1000).round(2)
        distance_scale = np.clip(points["distance_m"].fillna(np.inf).to_numpy() / (threshold_km * 1000), 0, 1)
        palette = np.array([[0, 0, 255], [247, 247, 247], [230, 126, 34]])
        rgb = np.column_stack([
            np.interp(distance_scale, [0, 0.5, 1], palette[:, channel])
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
        st.html(f'''<div class="map-legend">
            <span>Distance to a functional well · {strategy_view.lower()}</span>
            <div class="distance-gradient"></div>
            <div class="distance-labels"><span>0 km</span><span>{threshold_km / 2:g} km</span><span>{threshold_km:g}+ km</span></div>
            <p class="opacity-note">Each cell matches one population raster pixel. Hover for population and distance.</p>
        </div>''')
    with kpi_column, st.container(key="kpi_panel"):
        st.html('<div class="panel-heading"><span class="section-dot"></span>KPI overview</div>')
        st.html('''<div class="kpi-placeholder">
            <div class="placeholder-symbol">▥</div>
            <h3>Impact at a glance</h3>
            <p>Regional indicators will appear here.</p>
            <span class="placeholder-label">KEY INDICATORS</span>
        </div>''')
    with controls_column, st.container(key="controls_panel"):
        st.html('<div class="controls-eyebrow">YOUR SCENARIO</div>')
        st.subheader("Scenario settings")
        st.caption("Choose a region and set your repair budget.")
        st.selectbox("Region", regions, key="region")
        st.divider()
        st.number_input("N_repairs", min_value=0, max_value=1000,
                        step=1, key="repairs_input", on_change=sync_repairs,
                        args=("repairs_input",))
        st.slider("Number of repairs", min_value=0, max_value=1000,
                  key="repairs_slider", on_change=sync_repairs,
                  args=("repairs_slider",), label_visibility="collapsed")
        st.text_input("Estimated cost per repair", placeholder="e.g. 150,000 TZS", key="repair_cost")
        st.html('<div class="controls-note">A starting point for your next water-access scenario.</div>')


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
