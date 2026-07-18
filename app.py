"""
DRISHTI — AI-Driven Crime Analytics & Visualization Platform
============================================================
KSP Datathon 2026 prototype.  Run:  streamlit run app.py

All analytics are zone-level (no individual profiling). Demo data is
fully SYNTHETIC — the pipeline works unchanged on a real FIR export
with the same columns.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from ml_engine import (HotspotPredictor, DemandForecaster, ClusterDetector,
                       SHIFT_NAMES, DOW_NAMES)
from nlq_engine import parse_query, apply_filters, EXAMPLE_QUERIES
from patrol import allocate

# ---------------------------------------------------------------- theme ----
C = dict(bg="#0B1220", panel="#141E33", brass="#F0B429", blue="#4C8DFF",
         red="#E4572E", text="#E9EEF7", dim="#8FA0BC", green="#3ECF8E")
CRIME_COLORS = px.colors.qualitative.Bold

st.set_page_config(page_title="DRISHTI — Crime Analytics",
                   page_icon="🛡️", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family:'IBM Plex Sans',sans-serif; }}
.block-container {{ padding-top:1.1rem; }}
.cmd-strip {{ border-top:3px solid {C['brass']};
  background:linear-gradient(90deg,{C['panel']} 0%,#0E1730 100%);
  border-radius:0 0 10px 10px; padding:14px 22px 12px 22px;
  margin-bottom:6px; }}
.cmd-strip h1 {{ font-size:1.65rem; letter-spacing:.16em; margin:0;
  color:{C['text']}; }}
.cmd-strip h1 span {{ color:{C['brass']}; }}
.cmd-sub {{ color:{C['dim']}; font-family:'IBM Plex Mono',monospace;
  font-size:.78rem; letter-spacing:.08em; }}
.chip {{ display:inline-block; font-family:'IBM Plex Mono',monospace;
  font-size:.70rem; padding:2px 10px; border-radius:20px; margin-right:8px;
  border:1px solid {C['dim']}44; color:{C['dim']}; }}
.chip.warn {{ border-color:{C['brass']}; color:{C['brass']}; }}
[data-testid="stMetric"] {{ background:{C['panel']}; border:1px solid #22304D;
  border-left:3px solid {C['brass']}; border-radius:8px; padding:10px 14px; }}
[data-testid="stMetricLabel"] p {{ color:{C['dim']} !important;
  font-size:.74rem !important; letter-spacing:.06em;
  text-transform:uppercase; }}
.stTabs [data-baseweb="tab"] {{ letter-spacing:.05em; }}
footer {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data ----
@st.cache_data(show_spinner="Loading crime records…")
def load_data() -> pd.DataFrame:
    import os
    if not os.path.exists("data/crime_records.csv"):
        import generate_data
        generate_data.main()
    df = pd.read_csv("data/crime_records.csv", parse_dates=["DateTime"])
    df["DateOnly"] = df["DateTime"].dt.normalize()
    return df


@st.cache_resource(show_spinner="Training models (one-time)…")
def get_models():
    df = load_data()
    return HotspotPredictor(df), DemandForecaster(df), ClusterDetector(df)


df = load_data()
hp, fc, cd = get_models()
DATA_START, DATA_END = df["DateTime"].min(), df["DateTime"].max()


# ------------------------------------------------------------- helpers ----
def dark_fig(fig, h=340):
    fig.update_layout(template="plotly_dark", height=h,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans", color=C["text"]),
                      margin=dict(l=10, r=10, t=42, b=10),
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor="#22304D")
    fig.update_yaxes(gridcolor="#22304D")
    return fig


def base_map(center=(12.9629, 77.6100), zoom=11):
    return folium.Map(location=center, zoom_start=zoom,
                      tiles="CartoDB dark_matter", control_scale=True)


def show_map(m, height=520, key=None):
    st_folium(m, height=height, use_container_width=True,
              returned_objects=[], key=key)


def kpi_delta(cur: pd.DataFrame, full: pd.DataFrame, start, end):
    span = end - start
    prev = full[(full["DateTime"] >= start - span) &
                (full["DateTime"] < start)]
    if len(prev) == 0:
        return None
    return f"{(len(cur) - len(prev)) / len(prev) * 100:+.1f}% vs prev period"


# ------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.markdown(f"### 🛡️ DRISHTI <span style='color:{C['brass']}'>"
                f"CONTROL</span>", unsafe_allow_html=True)
    st.caption("Global filters apply to Overview, Hotspot Map & Trends.")
    d1, d2 = st.date_input(
        "Date range", (DATA_END - pd.Timedelta(days=180), DATA_END),
        min_value=DATA_START.date(), max_value=DATA_END.date())
    sel_crimes = st.multiselect("Crime types", sorted(df["Crime_Type"]
                                .unique()), default=[])
    sel_areas = st.multiselect("Areas", sorted(df["Area"].unique()),
                               default=[])
    st.divider()
    st.markdown(f"<span class='chip warn'>SYNTHETIC DEMO DATA</span>",
                unsafe_allow_html=True)
    st.caption("Modelled on realistic Bengaluru patterns; no real FIR data. "
               "Pipeline runs unchanged on a real FIR CSV export.")
    st.caption("Zone-level analytics only — no individual profiling.")

mask = ((df["DateTime"].dt.date >= d1) & (df["DateTime"].dt.date <= d2))
if sel_crimes:
    mask &= df["Crime_Type"].isin(sel_crimes)
if sel_areas:
    mask &= df["Area"].isin(sel_areas)
fdf = df[mask]

# -------------------------------------------------------------- header ----
st.markdown(f"""
<div class="cmd-strip">
  <h1>DRISHTI <span>ದೃಷ್ಟಿ</span></h1>
  <div class="cmd-sub">AI-DRIVEN CRIME ANALYTICS &amp; VISUALIZATION ·
  KSP DATATHON 2026</div>
  <div style="margin-top:8px">
    <span class="chip">RECORDS {len(df):,}</span>
    <span class="chip">WINDOW {d1:%d %b %Y} → {d2:%d %b %Y}</span>
    <span class="chip">MODEL MAE {hp.val_mae:.3f}/slot</span>
    <span class="chip warn">SYNTHETIC DATA</span>
  </div>
</div>""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("No records match the current filters — widen the date "
               "range or clear a filter in the sidebar.")
    st.stop()

TABS = st.tabs(["📊 Command Overview", "🗺️ Hotspot Map",
                "🔮 Prediction & Forecast", "📈 Trends & Patterns",
                "💬 Ask DRISHTI", "🚓 Patrol Allocation"])

# ======================================================= TAB 1: OVERVIEW ==
with TABS[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    start_ts, end_ts = pd.Timestamp(d1), pd.Timestamp(d2)
    c1.metric("Incidents", f"{len(fdf):,}",
              kpi_delta(fdf, df, start_ts, end_ts))
    c2.metric("Daily average", f"{len(fdf) / max((d2 - d1).days, 1):.1f}")
    c3.metric("Most affected area", fdf["Area"].mode().iloc[0])
    c4.metric("Top crime type", fdf["Crime_Type"].mode().iloc[0])
    c5.metric("Clearance rate",
              f"{(fdf['Status'].str.startswith(('Chargesheeted', 'Closed - Convicted'))).mean() * 100:.0f}%",
              help="Chargesheeted + convicted share of filtered FIRs")

    l, r = st.columns([3, 2])
    with l:
        wk = (fdf.set_index("DateTime").resample("W").size()
              .rename("incidents").reset_index())
        fig = px.area(wk, x="DateTime", y="incidents",
                      title="Weekly incident volume")
        fig.update_traces(line_color=C["brass"],
                          fillcolor="rgba(240,180,41,0.18)")
        st.plotly_chart(dark_fig(fig), width="stretch")
    with r:
        by_type = (fdf["Crime_Type"].value_counts().reset_index())
        by_type.columns = ["Crime_Type", "count"]
        fig = px.bar(by_type, x="count", y="Crime_Type", orientation="h",
                     title="Distribution by crime type",
                     color="Crime_Type",
                     color_discrete_sequence=CRIME_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(dark_fig(fig), width="stretch")

    l, r = st.columns([3, 2])
    with l:
        hm = (fdf.assign(DOW=fdf["DateTime"].dt.day_name())
              .groupby(["DOW", "Hour"]).size().rename("n").reset_index())
        pivot = (hm.pivot(index="DOW", columns="Hour", values="n")
                 .reindex(DOW_NAMES).fillna(0))
        fig = px.imshow(pivot, aspect="auto",
                        color_continuous_scale=["#0B1220", "#4C8DFF",
                                                "#F0B429", "#E4572E"],
                        title="When crime happens — hour × weekday",
                        labels=dict(color="incidents"))
        st.plotly_chart(dark_fig(fig, 360), width="stretch")
    with r:
        stat = fdf["Status"].value_counts().reset_index()
        stat.columns = ["Status", "n"]
        fig = px.pie(stat, names="Status", values="n", hole=0.55,
                     title="Case status",
                     color_discrete_sequence=[C["blue"], C["green"],
                                              C["brass"], C["red"]])
        st.plotly_chart(dark_fig(fig, 360), width="stretch")

# ==================================================== TAB 2: HOTSPOT MAP ==
with TABS[1]:
    l, r = st.columns([3, 1])
    with r:
        layer = st.radio("Layer", ["Heatmap", "Incident clusters"],
                         key="maplayer")
        st.caption(f"{len(fdf):,} incidents plotted for the current "
                   "filters. Use the sidebar to narrow crime type, "
                   "area or dates.")
        top_areas = (fdf["Area"].value_counts().head(8)
                     .rename_axis("Area").reset_index(name="incidents"))
        st.dataframe(top_areas, hide_index=True, width="stretch")
    with l:
        m = base_map()
        if layer == "Heatmap":
            HeatMap(fdf[["Latitude", "Longitude"]].to_numpy().tolist(),
                    radius=13, blur=18, min_opacity=0.25).add_to(m)
        else:
            mc = MarkerCluster().add_to(m)
            sample = fdf.sample(min(len(fdf), 2500), random_state=42)
            for _, rrow in sample.iterrows():
                folium.CircleMarker(
                    [rrow["Latitude"], rrow["Longitude"]], radius=3,
                    color=C["brass"], fill=True, fill_opacity=0.7,
                    popup=(f"{rrow['FIR_ID']}<br>{rrow['Crime_Type']}"
                           f"<br>{rrow['Area']} · "
                           f"{rrow['DateTime']:%d %b %Y %H:%M}")
                ).add_to(mc)
        show_map(m, key="hotspot_map")

# ============================================== TAB 3: PREDICTION ========
with TABS[2]:
    st.markdown("##### Risk prediction — gradient-boosted model over "
                "historical incident rates")
    a, b, c3_, d_ = st.columns(4)
    shift = a.selectbox("Shift", SHIFT_NAMES, index=0)
    dow_n = b.selectbox("Day", DOW_NAMES, index=5)
    month = c3_.selectbox("Month", list(range(1, 13)),
                          index=DATA_END.month - 1,
                          format_func=lambda m_:
                          pd.Timestamp(2026, m_, 1).strftime("%B"))
    pc = d_.multiselect("Restrict to crime types (optional)",
                        hp.crimes, default=[])
    risk = hp.area_risk(shift, DOW_NAMES.index(dow_n), month,
                        crime_types=pc or None)

    l, r = st.columns([2, 3])
    with l:
        fig = px.bar(risk.head(12).iloc[::-1], x="risk_score", y="Area",
                     orientation="h", title="Predicted risk score (0-100)",
                     color="risk_score",
                     color_continuous_scale=["#4C8DFF", "#F0B429",
                                             "#E4572E"],
                     hover_data=["dominant_crime", "expected_incidents"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(dark_fig(fig, 430), width="stretch")
    with r:
        coords = df.groupby("Area")[["Latitude", "Longitude"]].mean()
        m = base_map()
        for _, rw in risk.iterrows():
            la, lo = coords.loc[rw["Area"]]
            col = (C["red"] if rw["risk_score"] >= 70 else
                   C["brass"] if rw["risk_score"] >= 40 else C["blue"])
            folium.Circle(
                [la, lo], radius=250 + 14 * rw["risk_score"], color=col,
                fill=True, fill_opacity=0.28, weight=1,
                popup=(f"<b>{rw['Area']}</b><br>Risk {rw['risk_score']}"
                       f"<br>Dominant: {rw['dominant_crime']}")
            ).add_to(m)
        show_map(m, height=430, key="risk_map")

    st.markdown("##### 14-day citywide demand forecast")
    hist = fc.history.iloc[-60:]
    fut = fc.forecast(14)
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Observed",
                    line=dict(color=C["blue"]))
    fig.add_scatter(x=fut.index, y=fut.values, name="Forecast",
                    line=dict(color=C["brass"], dash="dash"))
    fig.add_vrect(x0=fut.index[0], x1=fut.index[-1],
                  fillcolor="rgba(240,180,41,0.07)", line_width=0)
    fig.update_layout(title="Daily incidents — last 60 days + next 14")
    st.plotly_chart(dark_fig(fig, 320), width="stretch")

    with st.expander("Model card — how these predictions are made"):
        st.markdown(f"""
| Component | Detail |
|---|---|
| Hotspot model | `HistGradientBoostingRegressor`, {hp.n_train:,} training rows |
| Features | area, crime type, shift, day-of-week, month, weekend flag |
| Target | historical incidents **per day-slot** (true zeros included) |
| Validation MAE | **{hp.val_mae:.4f}** incidents / day-slot (15% holdout) |
| Forecast model | autoregressive GBM — lags 1/7/14, 7-day rolling mean, calendar features |
| Forecast MAE | **{fc.val_mae:.2f}** incidents / day (last-10% holdout) |
| Scope | zone-level only; no personal data, no individual prediction |
""")

# ================================================ TAB 4: TRENDS ==========
with TABS[3]:
    l, r = st.columns(2)
    with l:
        mo = (fdf.set_index("DateTime")
              .groupby([pd.Grouper(freq="MS"), "Crime_Type"]).size()
              .rename("n").reset_index())
        fig = px.line(mo, x="DateTime", y="n", color="Crime_Type",
                      title="Monthly trend by crime type",
                      color_discrete_sequence=CRIME_COLORS)
        st.plotly_chart(dark_fig(fig, 380), width="stretch")
    with r:
        yr = (df.assign(Year=df["DateTime"].dt.year)
              .groupby(["Year", "Crime_Type"]).size()
              .rename("n").reset_index())
        fig = px.bar(yr, x="Crime_Type", y="n", color="Year",
                     barmode="group", title="Year-over-year (full data)",
                     color_discrete_sequence=[C["dim"], C["blue"],
                                              C["brass"]])
        st.plotly_chart(dark_fig(fig, 380), width="stretch")

    st.markdown("##### Emerging clusters — DBSCAN, last 90 days vs "
                "previous 90")
    cl = cd.clusters
    if cl.empty:
        st.info("Not enough recent incidents to form clusters.")
    else:
        l, r = st.columns([2, 3])
        with l:
            view = cl[["nearest_area", "dominant_crime", "incidents_recent",
                       "growth_pct", "emerging"]].rename(columns={
                           "nearest_area": "Area",
                           "dominant_crime": "Dominant crime",
                           "incidents_recent": "Recent (90d)",
                           "growth_pct": "Growth %",
                           "emerging": "Emerging"})
            st.dataframe(view, hide_index=True, width="stretch",
                         height=330)
            st.caption("A cluster is flagged **emerging** when incident "
                       "density inside its own radius grew ≥15% vs the "
                       "previous 90-day window.")
        with r:
            m = base_map()
            for _, rw in cl.iterrows():
                col = C["red"] if rw["emerging"] else C["blue"]
                folium.Circle(
                    [rw["lat"], rw["lon"]],
                    radius=180 + rw["incidents_recent"] * 0.9,
                    color=col, fill=True, fill_opacity=0.3, weight=2,
                    popup=(f"<b>{rw['nearest_area']}</b><br>"
                           f"{rw['dominant_crime']}<br>"
                           f"{rw['incidents_recent']} incidents · "
                           f"growth {rw['growth_pct']}%")
                ).add_to(m)
            show_map(m, height=380, key="cluster_map")

# ================================================= TAB 5: ASK DRISHTI ====
with TABS[4]:
    st.markdown("##### Ask in plain English — fully offline parser, "
                "runs on air-gapped infrastructure")
    if "nlq" not in st.session_state:
        st.session_state.nlq = EXAMPLE_QUERIES[0]
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLE_QUERIES):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.nlq = ex
    q = st.text_input("Query", key="nlq",
                      label_visibility="collapsed",
                      placeholder="e.g. vehicle theft hotspots last 3 months")
    if q.strip():
        p = parse_query(q, df)
        sub = apply_filters(df, p)
        st.markdown(p["interpretation"])
        if sub.empty:
            st.warning("No incidents match that query — try a wider time "
                       "window or a different area.")
        elif p["intent"] == "count":
            a, b, c3_ = st.columns(3)
            a.metric("Matching incidents", f"{len(sub):,}")
            b.metric("Most affected area", sub["Area"].mode().iloc[0])
            days = max((p["end"] - p["start"]).days, 1)
            c3_.metric("Per day", f"{len(sub) / days:.1f}")
        elif p["intent"] == "trend":
            g = (sub.set_index("DateTime").resample("W").size()
                 .rename("incidents").reset_index())
            fig = px.line(g, x="DateTime", y="incidents",
                          title="Weekly trend for your query")
            fig.update_traces(line_color=C["brass"])
            st.plotly_chart(dark_fig(fig), width="stretch")
            hg = sub.groupby("Hour").size().rename("n").reset_index()
            fig = px.bar(hg, x="Hour", y="n", title="By hour of day")
            fig.update_traces(marker_color=C["blue"])
            st.plotly_chart(dark_fig(fig, 260), width="stretch")
        elif p["intent"] == "compare":
            g = (sub["Area"].value_counts().head(12).rename_axis("Area")
                 .reset_index(name="incidents"))
            fig = px.bar(g.iloc[::-1], x="incidents", y="Area",
                         orientation="h", title="Ranked by incidents",
                         color="incidents",
                         color_continuous_scale=["#4C8DFF", "#F0B429",
                                                 "#E4572E"])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(dark_fig(fig, 420), width="stretch")
        else:                                              # map
            l, r = st.columns([3, 1])
            with r:
                st.metric("Matching incidents", f"{len(sub):,}")
                st.dataframe(sub["Area"].value_counts().head(6)
                             .rename_axis("Area").reset_index(name="n"),
                             hide_index=True, width="stretch")
            with l:
                m = base_map()
                HeatMap(sub[["Latitude", "Longitude"]].to_numpy().tolist(),
                        radius=13, blur=18, min_opacity=0.3).add_to(m)
                show_map(m, height=430, key="nlq_map")

# ============================================== TAB 6: PATROL ============
with TABS[5]:
    st.markdown("##### Data-driven patrol allocation — largest-remainder "
                "method over predicted risk")
    a, b, c3_ = st.columns(3)
    units = a.number_input("Patrol units available", 20, 400, 60, step=5)
    pshift = b.selectbox("Shift", SHIFT_NAMES, index=0, key="pshift")
    pday = c3_.selectbox("Day", DOW_NAMES, index=5, key="pday")
    prisk = hp.area_risk(pshift, DOW_NAMES.index(pday), DATA_END.month)
    alloc = allocate(prisk, int(units))

    l, r = st.columns([2, 3])
    with l:
        st.dataframe(
            alloc[["Area", "risk_score", "dominant_crime", "patrol_units",
                   "priority"]].rename(columns={
                       "risk_score": "Risk", "dominant_crime": "Focus crime",
                       "patrol_units": "Units", "priority": "Priority"}),
            hide_index=True, width="stretch", height=430)
        st.download_button("⬇ Download deployment order (CSV)",
                           alloc.to_csv(index=False),
                           file_name="patrol_allocation.csv")
    with r:
        coords = df.groupby("Area")[["Latitude", "Longitude"]].mean()
        m = base_map()
        for _, rw in alloc.iterrows():
            la, lo = coords.loc[rw["Area"]]
            col = {"CRITICAL": C["red"], "ELEVATED": C["brass"],
                   "ROUTINE": C["blue"]}[str(rw["priority"])]
            folium.Circle(
                [la, lo], radius=220 + 130 * int(rw["patrol_units"]),
                color=col, fill=True, fill_opacity=0.3, weight=2,
                popup=(f"<b>{rw['Area']}</b><br>{rw['patrol_units']} units "
                       f"· {rw['priority']}<br>Focus: "
                       f"{rw['dominant_crime']}")).add_to(m)
        show_map(m, height=470, key="patrol_map")
    st.caption(f"{int(alloc['patrol_units'].sum())} units allocated across "
               f"{len(alloc)} areas for {pday} · {pshift}. Minimum 1 unit "
               "per area guaranteed; remainder distributed by predicted "
               "risk.")

st.markdown(f"<div style='text-align:center;color:{C['dim']};"
            f"font-size:.72rem;padding-top:18px'>DRISHTI ದೃಷ್ಟಿ · KSP "
            "Datathon 2026 prototype · synthetic demo data · zone-level "
            "analytics only</div>", unsafe_allow_html=True)
