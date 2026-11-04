from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Water access | Tanzania", page_icon="💧", layout="wide")
st.html((Path(__file__).with_name("style.css")).read_text())
st.session_state.setdefault("repairs_input", 100)
st.session_state.setdefault("repairs_slider", 100)


def sync_repairs(source):
    target = "repairs_input" if source == "repairs_slider" else "repairs_slider"
    st.session_state[target] = st.session_state[source]


def workspace(title="Water access overview"):
    with st.container(key="intro"):
        st.caption("TANZANIA  /  WATER INFRASTRUCTURE")
        st.title(title)
        st.write("Explore a region and plan your next intervention.")

    map_column, kpi_column, controls_column = st.columns([5.8, 3.2, 2.6], gap="medium")
    with map_column, st.container(key="map_panel"):
        st.html('<div class="panel-heading"><span class="section-dot"></span>Map view</div>')
        st.html('''<div class="map-placeholder">
            <div class="placeholder-symbol">⌖</div>
            <h3>A place to explore</h3>
            <p>The regional map will appear here.</p>
            <span class="placeholder-label">MAP VIEW</span>
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
        st.selectbox("Region", ["Morogoro", "Arusha", "Dodoma", "Dar es Salaam"], key="region")
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
