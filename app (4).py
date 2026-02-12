"""
Lapstone LLC — Philadelphia Construction & Real Estate Intelligence Dashboard
==============================================================================
Pulls live data from FRED, Census ACS, and BLS to provide a comprehensive
view of the Philadelphia metro area's construction economy, demographics,
and rental market dynamics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
import os

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

def _get_secret(key: str) -> str:
    """Try st.secrets first (Streamlit Cloud), then env vars."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

FRED_API_KEY = _get_secret("FRED_API_KEY")
CENSUS_API_KEY = _get_secret("CENSUS_API_KEY")

# FIPS Codes
PHILLY_COUNTY_FIPS = "42101"  # Philadelphia County
PA_STATE_FIPS = "42"

# Surrounding counties
COUNTY_FIPS = {
    "Philadelphia": "42101",
    "Montgomery": "42091",
    "Bucks": "42017",
    "Delaware": "42045",
    "Chester": "42029",
    "Berks (Reading)": "42011",
    "Camden (NJ)": "34007",
    "Burlington (NJ)": "34005",
    "Gloucester (NJ)": "34015",
}

# FRED Series
FRED_SERIES = {
    "Unemployment Rate (Philly County)": "PAPHIL5URN",
    "Unemployment Rate (Philly MSA)": "PHIL942URN",
    "Building Permits (MSA, Total)": "PHIL942BPPRIV",
    "Building Permits (MSA, 1-Unit)": "PHIL942BP1FH",
    "All Employees: Construction (MSA)": "PHIL942CONS",
    "All Employees: Total Nonfarm (MSA)": "PHIL942NA",
    "Labor Force (Philly County)": "LAUCN421010000000006",
    "Homeownership Rate (Philly)": "HOWNRATEACS042101",
    "Resident Population (Philly County)": "PAPHIL5POP",
    "GDP (Philly MSA)": "NGMP37964",
    "Median HH Income (Philly County)": "MHIPA42101A052NCEN",
    "CPI: Shelter (Philly MSA)": "CUURS12ASAH",
    "CPI: All Items (Philly MSA)": "CUURS12ASA0",
}

# Plotly color palette — dark theme with gold accent (Lapstone branding)
COLORS = {
    "gold": "#C8A951",
    "slate": "#4A6274",
    "teal": "#2EC4B6",
    "coral": "#E76F51",
    "lavender": "#9B8EC7",
    "sky": "#48A9A6",
    "sand": "#D4A373",
    "steel": "#7F8C9B",
    "bg": "#0E1117",
    "card": "#1A1D26",
    "text": "#E8E8E8",
    "muted": "#7F8C9B",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="DM Sans, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor=COLORS["card"],
        font_size=12,
        font_family="DM Sans",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 6)
def fetch_fred_series(series_id: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch a single FRED series. Returns DataFrame with date + value."""
    if not FRED_API_KEY:
        return pd.DataFrame()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return df[["date", "value"]].reset_index(drop=True)
    except Exception as e:
        st.warning(f"FRED fetch error for {series_id}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 12)
def fetch_census_acs(
    variables: list[str],
    geo: str = "county:*",
    state: str = "42",
    year: int = 2023,
    dataset: str = "acs/acs5",
) -> pd.DataFrame:
    """Fetch variables from Census ACS API. Returns raw DataFrame."""
    if not CENSUS_API_KEY:
        return pd.DataFrame()
    base = f"https://api.census.gov/data/{year}/{dataset}"
    get_vars = ",".join(["NAME"] + variables)
    params = {
        "get": get_vars,
        "for": geo,
        "key": CENSUS_API_KEY,
    }
    if geo.startswith("tract") or geo.startswith("county"):
        params["in"] = f"state:{state}"
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except Exception as e:
        st.warning(f"Census ACS error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 12)
def fetch_census_acs_county_multi(
    variables: list[str],
    county_fips: dict,
    year: int = 2023,
    dataset: str = "acs/acs5",
) -> pd.DataFrame:
    """Fetch ACS data for multiple specific counties."""
    if not CENSUS_API_KEY:
        return pd.DataFrame()
    frames = []
    for name, fips in county_fips.items():
        state = fips[:2]
        county = fips[2:]
        base = f"https://api.census.gov/data/{year}/{dataset}"
        get_vars = ",".join(["NAME"] + variables)
        params = {
            "get": get_vars,
            "for": f"county:{county}",
            "in": f"state:{state}",
            "key": CENSUS_API_KEY,
        }
        try:
            r = requests.get(base, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if len(data) >= 2:
                row = pd.DataFrame(data[1:], columns=data[0])
                row["county_label"] = name
                frames.append(row)
        except Exception:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


@st.cache_data(ttl=3600 * 12)
def fetch_census_tract_data(
    variables: list[str],
    state: str = "42",
    county: str = "101",
    year: int = 2023,
) -> pd.DataFrame:
    """Fetch tract-level ACS data for a county."""
    if not CENSUS_API_KEY:
        return pd.DataFrame()
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = ",".join(["NAME"] + variables)
    params = {
        "get": get_vars,
        "for": "tract:*",
        "in": f"state:{state}&in=county:{county}",
        "key": CENSUS_API_KEY,
    }
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except Exception as e:
        st.warning(f"Census tract error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 12)
def fetch_bls_qcew(
    area_fips: str = "42101",
    year: str = "2024",
    qtr: str = "1",
    industry: str = "1012",  # Construction
) -> pd.DataFrame:
    """Fetch QCEW data from BLS API (CSV files)."""
    url = f"https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area_fips}.csv"
    try:
        df = pd.read_csv(url, dtype=str, timeout=20)
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def make_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def format_number(n, decimals=1):
    if pd.isna(n):
        return "N/A"
    if abs(n) >= 1e9:
        return f"${n/1e9:,.{decimals}f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:,.{decimals}f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:,.{decimals}f}K"
    return f"{n:,.{decimals}f}"


def build_time_series_chart(
    df: pd.DataFrame,
    title: str,
    y_label: str = "",
    color: str = COLORS["gold"],
    show_trend: bool = False,
    y_format: str = None,
):
    """Build a clean time-series line chart."""
    if df.empty:
        return go.Figure().update_layout(title=title, **CHART_LAYOUT)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{y_label}: %{{y:,.1f}}<extra></extra>",
        )
    )
    if show_trend and len(df) > 12:
        df_sorted = df.sort_values("date")
        rolling = df_sorted["value"].rolling(12, min_periods=6).mean()
        fig.add_trace(
            go.Scatter(
                x=df_sorted["date"],
                y=rolling,
                mode="lines",
                line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
                name="12-mo avg",
                hoverinfo="skip",
            )
        )
    layout_kwargs = {**CHART_LAYOUT, "title": dict(text=title, font=dict(size=16))}
    if y_format:
        layout_kwargs["yaxis"] = {**CHART_LAYOUT.get("yaxis", {}), "tickformat": y_format}
    fig.update_layout(**layout_kwargs)
    return fig


def build_bar_chart(
    names: list,
    values: list,
    title: str,
    color: str = COLORS["gold"],
    horizontal: bool = False,
    y_label: str = "",
):
    """Build a clean bar chart."""
    fig = go.Figure()
    if horizontal:
        fig.add_trace(
            go.Bar(
                y=names,
                x=values,
                orientation="h",
                marker_color=color,
                hovertemplate=f"<b>%{{y}}</b><br>{y_label}: %{{x:,.1f}}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y:,.1f}}<extra></extra>",
            )
        )
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(size=16)))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lapstone Intel — Philadelphia Construction Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header banner */
    .dashboard-header {
        background: linear-gradient(135deg, #1A1D26 0%, #0E1117 100%);
        border: 1px solid rgba(200, 169, 81, 0.2);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .dashboard-header h1 {
        color: #C8A951;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        color: #7F8C9B;
        font-size: 0.95rem;
        margin: 0.25rem 0 0 0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1A1D26;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricLabel"] {
        color: #7F8C9B !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #E8E8E8 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Space Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        color: #7F8C9B;
    }
    .stTabs [aria-selected="true"] {
        color: #C8A951 !important;
        border-bottom: 2px solid #C8A951;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #12151C;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #C8A951;
    }

    /* Section dividers */
    .section-label {
        color: #C8A951;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin: 2rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(200,169,81,0.15);
    }

    /* Info boxes */
    .info-box {
        background: rgba(200,169,81,0.06);
        border-left: 3px solid #C8A951;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #A8ADB5;
        margin: 0.5rem 0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    div.block-container {
        padding-top: 1.5rem;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏗️ Lapstone Intel")
    st.markdown(
        '<p style="color:#7F8C9B; font-size:0.85rem;">Philadelphia Construction & Real Estate Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # API Key Management
    st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)

    if not FRED_API_KEY:
        fred_key_input = st.text_input(
            "FRED API Key",
            type="password",
            help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        if fred_key_input:
            FRED_API_KEY = fred_key_input
            os.environ["FRED_API_KEY"] = fred_key_input
            st.rerun()
    else:
        st.success("✓ FRED API connected", icon="✅")

    if not CENSUS_API_KEY:
        census_key_input = st.text_input(
            "Census API Key",
            type="password",
            help="Get a free key at https://api.census.gov/data/key_signup.html",
        )
        if census_key_input:
            CENSUS_API_KEY = census_key_input
            os.environ["CENSUS_API_KEY"] = census_key_input
            st.rerun()
    else:
        st.success("✓ Census API connected", icon="✅")

    st.divider()

    # Date range
    st.markdown('<div class="section-label">Time Range</div>', unsafe_allow_html=True)
    start_year = st.slider("Start year", 2010, 2024, 2015)
    start_date = f"{start_year}-01-01"

    st.divider()

    # County selection for comparisons
    st.markdown('<div class="section-label">Regional Comparison</div>', unsafe_allow_html=True)
    selected_counties = st.multiselect(
        "Compare counties",
        options=list(COUNTY_FIPS.keys()),
        default=["Philadelphia", "Montgomery", "Bucks", "Berks (Reading)"],
    )

    st.divider()

    st.markdown(
        """
        <div style="color:#5A6270; font-size:0.75rem; margin-top:1rem;">
        <b>Data Sources</b><br>
        • Federal Reserve (FRED)<br>
        • U.S. Census Bureau (ACS 5-Year)<br>
        • Bureau of Labor Statistics (QCEW)<br><br>
        Built for <a href="https://www.lapstonellc.com" target="_blank" style="color:#C8A951;">Lapstone LLC</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="dashboard-header">
        <h1>Philadelphia Metro Intelligence</h1>
        <p>Construction economy, rental demand, and demographic analytics for the greater Philadelphia region</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_overview, tab_construction, tab_rental, tab_demographics, tab_regional = st.tabs(
    ["📊 Overview", "🏗️ Construction", "🏠 Rental Demand", "👥 Demographics", "🗺️ Regional"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    if not FRED_API_KEY:
        st.info(
            "👈 Enter your free **FRED API key** in the sidebar to load live economic data. "
            "Get one at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)"
        )

    # Key metrics row
    st.markdown('<div class="section-label">Key Indicators — Latest Available</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    # Unemployment
    unemp_df = fetch_fred_series("PAPHIL5URN", start_date)
    with col1:
        if not unemp_df.empty:
            latest = unemp_df.iloc[-1]["value"]
            prev = unemp_df.iloc[-13]["value"] if len(unemp_df) > 13 else None
            delta = f"{latest - prev:+.1f}pp YoY" if prev else None
            make_metric_card("Unemployment Rate", f"{latest:.1f}%", delta, "inverse")
        else:
            make_metric_card("Unemployment Rate", "—")

    # Construction Employment
    const_emp_df = fetch_fred_series("PHIL942CONS", start_date)
    with col2:
        if not const_emp_df.empty:
            latest = const_emp_df.iloc[-1]["value"]
            prev_yr = const_emp_df.iloc[-13]["value"] if len(const_emp_df) > 13 else None
            delta = f"{((latest/prev_yr)-1)*100:+.1f}% YoY" if prev_yr else None
            make_metric_card("Construction Jobs (MSA)", f"{latest:,.0f}K", delta)
        else:
            make_metric_card("Construction Jobs", "—")

    # Building Permits
    permits_df = fetch_fred_series("PHIL942BPPRIV", start_date)
    with col3:
        if not permits_df.empty:
            # Sum last 12 months vs prior 12
            recent_12 = permits_df.tail(12)["value"].sum()
            prior_12 = permits_df.iloc[-24:-12]["value"].sum() if len(permits_df) >= 24 else None
            delta = f"{((recent_12/prior_12)-1)*100:+.1f}% YoY" if prior_12 and prior_12 > 0 else None
            make_metric_card("Permits (12-mo, MSA)", f"{recent_12:,.0f}", delta)
        else:
            make_metric_card("Building Permits", "—")

    # GDP
    gdp_df = fetch_fred_series("NGMP37964", start_date)
    with col4:
        if not gdp_df.empty:
            latest = gdp_df.iloc[-1]["value"]
            prev = gdp_df.iloc[-2]["value"] if len(gdp_df) > 1 else None
            delta = f"{((latest/prev)-1)*100:+.1f}% YoY" if prev else None
            make_metric_card("GDP (Philly MSA)", format_number(latest * 1e6), delta)
        else:
            make_metric_card("GDP", "—")

    # Homeownership Rate
    homeown_df = fetch_fred_series("HOWNRATEACS042101", "2010-01-01")
    with col5:
        if not homeown_df.empty:
            latest = homeown_df.iloc[-1]["value"]
            prev = homeown_df.iloc[-2]["value"] if len(homeown_df) > 1 else None
            delta = f"{latest - prev:+.1f}pp" if prev else None
            make_metric_card("Homeownership Rate", f"{latest:.1f}%", delta, "off")
        else:
            make_metric_card("Homeownership Rate", "—")

    st.markdown("")

    # Time-series charts
    col_left, col_right = st.columns(2)

    with col_left:
        fig = build_time_series_chart(
            unemp_df,
            "Unemployment Rate — Philadelphia County",
            y_label="Rate",
            color=COLORS["coral"],
            show_trend=True,
            y_format=".1f",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = build_time_series_chart(
            const_emp_df,
            "Construction Employment — Philadelphia MSA (Thousands)",
            y_label="Employees (K)",
            color=COLORS["teal"],
            show_trend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        fig = build_time_series_chart(
            permits_df,
            "Building Permits Issued — Philadelphia MSA (Monthly)",
            y_label="Permits",
            color=COLORS["gold"],
            show_trend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        fig = build_time_series_chart(
            gdp_df,
            "Gross Domestic Product — Philadelphia MSA ($M)",
            y_label="GDP ($M)",
            color=COLORS["lavender"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Shelter CPI
    st.markdown('<div class="section-label">Inflation Tracking</div>', unsafe_allow_html=True)

    col_cpi1, col_cpi2 = st.columns(2)
    shelter_cpi = fetch_fred_series("CUURS12ASAH", start_date)
    all_cpi = fetch_fred_series("CUURS12ASA0", start_date)

    with col_cpi1:
        if not shelter_cpi.empty and not all_cpi.empty:
            # Calculate YoY % change
            for df_cpi in [shelter_cpi, all_cpi]:
                df_cpi_sorted = df_cpi.sort_values("date")
                df_cpi["yoy"] = df_cpi_sorted["value"].pct_change(12) * 100

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=shelter_cpi["date"],
                    y=shelter_cpi["yoy"],
                    mode="lines",
                    name="Shelter CPI (YoY %)",
                    line=dict(color=COLORS["coral"], width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=all_cpi["date"],
                    y=all_cpi["yoy"],
                    mode="lines",
                    name="All Items CPI (YoY %)",
                    line=dict(color=COLORS["steel"], width=2),
                )
            )
            fig.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Philadelphia MSA — CPI Year-over-Year Change (%)", font=dict(size=16)),
                yaxis_ticksuffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_cpi2:
        median_income_df = fetch_fred_series("MHIPA42101A052NCEN", "2010-01-01")
        fig = build_time_series_chart(
            median_income_df,
            "Median Household Income — Philadelphia County",
            y_label="Income",
            color=COLORS["sky"],
            y_format="$,.0f",
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_construction:
    st.markdown('<div class="section-label">Construction Sector Deep Dive</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Permits: 1-unit vs total
    permits_1unit = fetch_fred_series("PHIL942BP1FH", start_date)

    with col1:
        if not permits_df.empty and not permits_1unit.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=permits_df["date"],
                    y=permits_df["value"],
                    mode="lines",
                    name="Total Permits",
                    line=dict(color=COLORS["gold"], width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=permits_1unit["date"],
                    y=permits_1unit["value"],
                    mode="lines",
                    name="Single-Family (1-Unit)",
                    line=dict(color=COLORS["teal"], width=2),
                )
            )
            # Multi-family = total - 1-unit
            merged = permits_df.merge(permits_1unit, on="date", suffixes=("_total", "_1unit"))
            merged["multi"] = merged["value_total"] - merged["value_1unit"]
            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged["multi"],
                    mode="lines",
                    name="Multi-Family (2+ Unit)",
                    line=dict(color=COLORS["coral"], width=2),
                )
            )
            fig.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Building Permits by Structure Type — Philadelphia MSA", font=dict(size=16)),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Construction employment vs total nonfarm
        nonfarm_df = fetch_fred_series("PHIL942NA", start_date)
        if not const_emp_df.empty and not nonfarm_df.empty:
            # Calculate construction as % of total
            merged_emp = const_emp_df.merge(nonfarm_df, on="date", suffixes=("_const", "_total"))
            merged_emp["pct"] = (merged_emp["value_const"] / merged_emp["value_total"]) * 100

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=merged_emp["date"],
                    y=merged_emp["pct"],
                    mode="lines",
                    line=dict(color=COLORS["gold"], width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(200,169,81,0.08)",
                    hovertemplate="<b>%{x|%b %Y}</b><br>Construction share: %{y:.2f}%<extra></extra>",
                )
            )
            fig.update_layout(
                **CHART_LAYOUT,
                title=dict(
                    text="Construction as % of Total Employment — Philadelphia MSA",
                    font=dict(size=16),
                ),
                yaxis_ticksuffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)

    # QCEW Construction data
    st.markdown('<div class="section-label">Construction Industry Breakdown (BLS QCEW)</div>', unsafe_allow_html=True)

    qcew_years = ["2024", "2023", "2022", "2021", "2020"]
    selected_qcew_year = st.selectbox("QCEW Year", qcew_years, index=0)

    qcew_data = fetch_bls_qcew(area_fips="42101", year=selected_qcew_year, qtr="1")

    if not qcew_data.empty and "industry_code" in qcew_data.columns:
        # Filter to construction-related NAICS codes
        construction_codes = {
            "1012": "Construction (Total)",
            "1013": "Construction of Buildings",
            "1014": "Heavy & Civil Engineering",
            "1015": "Specialty Trade Contractors",
        }
        const_rows = qcew_data[
            (qcew_data["industry_code"].isin(construction_codes.keys()))
            & (qcew_data["own_code"] == "5")  # Private
        ].copy()

        if not const_rows.empty:
            for col_name in ["annual_avg_emplvl", "avg_annual_pay", "annual_avg_estabs_count"]:
                if col_name in const_rows.columns:
                    const_rows[col_name] = pd.to_numeric(const_rows[col_name], errors="coerce")

            const_rows["industry_label"] = const_rows["industry_code"].map(construction_codes)

            col_q1, col_q2, col_q3 = st.columns(3)
            total_row = const_rows[const_rows["industry_code"] == "1012"]

            if not total_row.empty:
                with col_q1:
                    val = total_row.iloc[0].get("annual_avg_emplvl", 0)
                    make_metric_card("Construction Employment", f"{val:,.0f}" if pd.notna(val) else "N/A")
                with col_q2:
                    val = total_row.iloc[0].get("avg_annual_pay", 0)
                    make_metric_card("Avg Annual Pay", f"${val:,.0f}" if pd.notna(val) else "N/A")
                with col_q3:
                    val = total_row.iloc[0].get("annual_avg_estabs_count", 0)
                    make_metric_card("Establishments", f"{val:,.0f}" if pd.notna(val) else "N/A")

            # Sub-industry breakdown
            sub_rows = const_rows[const_rows["industry_code"] != "1012"]
            if not sub_rows.empty and "annual_avg_emplvl" in sub_rows.columns:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    fig = build_bar_chart(
                        sub_rows["industry_label"].tolist(),
                        sub_rows["annual_avg_emplvl"].tolist(),
                        f"Construction Employment by Sub-Industry — {selected_qcew_year}",
                        color=COLORS["teal"],
                        y_label="Employees",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col_s2:
                    if "avg_annual_pay" in sub_rows.columns:
                        fig = build_bar_chart(
                            sub_rows["industry_label"].tolist(),
                            sub_rows["avg_annual_pay"].tolist(),
                            f"Avg Annual Pay by Sub-Industry — {selected_qcew_year}",
                            color=COLORS["gold"],
                            y_label="Avg Pay ($)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            '<div class="info-box">QCEW data loads directly from BLS — no API key required. '
            "If data doesn't appear, the selected year/quarter may not be published yet.</div>",
            unsafe_allow_html=True,
        )

    # Permit trends — annual aggregation
    st.markdown('<div class="section-label">Annual Permit Volume Trend</div>', unsafe_allow_html=True)

    if not permits_df.empty:
        annual = permits_df.copy()
        annual["year"] = annual["date"].dt.year
        annual_sum = annual.groupby("year")["value"].sum().reset_index()
        annual_sum = annual_sum[annual_sum["year"] >= start_year]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=annual_sum["year"],
                y=annual_sum["value"],
                marker_color=[
                    COLORS["gold"] if y == annual_sum["year"].max() else COLORS["slate"]
                    for y in annual_sum["year"]
                ],
                hovertemplate="<b>%{x}</b><br>Total Permits: %{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Annual Building Permits — Philadelphia MSA", font=dict(size=16)),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: RENTAL DEMAND INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab_rental:
    st.markdown('<div class="section-label">Rental Market — Young Professional Demand Signals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        "<b>Target Renter Avatar:</b> Young professionals (25–34), employed, responsible renters who "
        "can't yet afford to buy. This section identifies where these renters cluster and tracks "
        "affordability dynamics."
        "</div>",
        unsafe_allow_html=True,
    )

    if not CENSUS_API_KEY:
        st.info(
            "👈 Enter your free **Census API key** in the sidebar to load demographic data. "
            "Get one at [api.census.gov](https://api.census.gov/data/key_signup.html)"
        )
    else:
        # ── County-level comparison: Rent burden, young adults, income ──
        county_vars = [
            "B01001_011E",  # Male 25-29
            "B01001_012E",  # Male 30-34
            "B01001_035E",  # Female 25-29
            "B01001_036E",  # Female 30-34
            "B01003_001E",  # Total population
            "B25064_001E",  # Median gross rent
            "B19013_001E",  # Median household income
            "B25003_001E",  # Total tenure
            "B25003_003E",  # Renter-occupied
            "B25070_007E",  # Rent 25-29.9% of income
            "B25070_008E",  # Rent 30-34.9%
            "B25070_009E",  # Rent 35-39.9%
            "B25070_010E",  # Rent 40-49.9%
            "B25070_011E",  # Rent 50%+
            "B25070_001E",  # Total rent burden universe
        ]

        selected_fips = {k: v for k, v in COUNTY_FIPS.items() if k in selected_counties}
        county_data = fetch_census_acs_county_multi(county_vars, selected_fips)

        if not county_data.empty:
            # Compute metrics
            for col in county_vars:
                county_data[col] = pd.to_numeric(county_data[col], errors="coerce")

            county_data["pop_25_34"] = (
                county_data["B01001_011E"]
                + county_data["B01001_012E"]
                + county_data["B01001_035E"]
                + county_data["B01001_036E"]
            )
            county_data["pop_25_34_pct"] = county_data["pop_25_34"] / county_data["B01003_001E"] * 100
            county_data["renter_pct"] = county_data["B25003_003E"] / county_data["B25003_001E"] * 100
            county_data["rent_burdened"] = (
                county_data["B25070_008E"]
                + county_data["B25070_009E"]
                + county_data["B25070_010E"]
                + county_data["B25070_011E"]
            )
            county_data["rent_burden_pct"] = county_data["rent_burdened"] / county_data["B25070_001E"] * 100

            # Summary metrics
            philly_row = county_data[county_data["county_label"] == "Philadelphia"]
            if not philly_row.empty:
                pr = philly_row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    make_metric_card("Young Adults (25–34)", f"{pr['pop_25_34']:,.0f}")
                with c2:
                    make_metric_card("% Population 25–34", f"{pr['pop_25_34_pct']:.1f}%")
                with c3:
                    make_metric_card("Median Gross Rent", f"${pr['B25064_001E']:,.0f}")
                with c4:
                    make_metric_card("Renter-Occupied %", f"{pr['renter_pct']:.1f}%")

            st.markdown("")

            # Charts
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                sorted_df = county_data.sort_values("pop_25_34_pct", ascending=True)
                fig = build_bar_chart(
                    sorted_df["county_label"].tolist(),
                    sorted_df["pop_25_34_pct"].tolist(),
                    "Population Age 25–34 as % of Total",
                    color=COLORS["teal"],
                    horizontal=True,
                    y_label="% of pop",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_r2:
                sorted_df = county_data.sort_values("renter_pct", ascending=True)
                fig = build_bar_chart(
                    sorted_df["county_label"].tolist(),
                    sorted_df["renter_pct"].tolist(),
                    "Renter-Occupied Housing as % of Total",
                    color=COLORS["coral"],
                    horizontal=True,
                    y_label="Renter %",
                )
                st.plotly_chart(fig, use_container_width=True)

            col_r3, col_r4 = st.columns(2)

            with col_r3:
                sorted_df = county_data.sort_values("B25064_001E", ascending=True)
                fig = build_bar_chart(
                    sorted_df["county_label"].tolist(),
                    sorted_df["B25064_001E"].tolist(),
                    "Median Gross Rent by County",
                    color=COLORS["gold"],
                    horizontal=True,
                    y_label="Rent ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_r4:
                sorted_df = county_data.sort_values("rent_burden_pct", ascending=True)
                fig = build_bar_chart(
                    sorted_df["county_label"].tolist(),
                    sorted_df["rent_burden_pct"].tolist(),
                    "Rent-Burdened Households (30%+ of Income on Rent)",
                    color=COLORS["lavender"],
                    horizontal=True,
                    y_label="Burdened %",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Rent vs Income scatter
            st.markdown('<div class="section-label">Affordability Matrix</div>', unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=county_data["B19013_001E"],
                    y=county_data["B25064_001E"],
                    mode="markers+text",
                    text=county_data["county_label"],
                    textposition="top center",
                    textfont=dict(size=11, color=COLORS["text"]),
                    marker=dict(
                        size=county_data["pop_25_34_pct"] * 3,
                        color=county_data["renter_pct"],
                        colorscale=[[0, COLORS["slate"]], [1, COLORS["gold"]]],
                        colorbar=dict(title="Renter %"),
                        line=dict(width=1, color="rgba(255,255,255,0.2)"),
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Median HH Income: $%{x:,.0f}<br>"
                        "Median Rent: $%{y:,.0f}<br>"
                        "<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                **CHART_LAYOUT,
                title=dict(
                    text="Rent vs Income — Bubble Size = Young Adult Concentration",
                    font=dict(size=16),
                ),
                xaxis_title="Median Household Income ($)",
                yaxis_title="Median Gross Rent ($)",
                xaxis_tickprefix="$",
                yaxis_tickprefix="$",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Tract-level analysis for Philly ──
        st.markdown('<div class="section-label">Philadelphia Tract-Level Analysis — Young Professional Hotspots</div>', unsafe_allow_html=True)

        tract_vars = [
            "B01001_011E",  # Male 25-29
            "B01001_012E",  # Male 30-34
            "B01001_035E",  # Female 25-29
            "B01001_036E",  # Female 30-34
            "B01003_001E",  # Total pop
            "B25064_001E",  # Median gross rent
            "B19013_001E",  # Median HH income
            "B25003_003E",  # Renters
            "B25003_001E",  # Total tenure
        ]

        tract_data = fetch_census_tract_data(tract_vars, state="42", county="101")

        if not tract_data.empty:
            for col in tract_vars:
                tract_data[col] = pd.to_numeric(tract_data[col], errors="coerce")

            tract_data["pop_25_34"] = (
                tract_data["B01001_011E"]
                + tract_data["B01001_012E"]
                + tract_data["B01001_035E"]
                + tract_data["B01001_036E"]
            )
            tract_data["pop_25_34_pct"] = tract_data["pop_25_34"] / tract_data["B01003_001E"] * 100
            tract_data["renter_pct"] = tract_data["B25003_003E"] / tract_data["B25003_001E"] * 100
            tract_data["rent_to_income"] = (tract_data["B25064_001E"] * 12) / tract_data["B19013_001E"] * 100

            # Clean
            tract_clean = tract_data.dropna(subset=["pop_25_34_pct", "B25064_001E", "B19013_001E"])
            tract_clean = tract_clean[tract_clean["B01003_001E"] > 200]  # Meaningful population
            tract_clean = tract_clean[tract_clean["pop_25_34_pct"] < 100]  # Filter outliers

            # Compute a "Young Professional Demand Score"
            # Composite: high 25-34 concentration + high renter % + moderate rent-to-income
            scaler_cols = ["pop_25_34_pct", "renter_pct"]
            for sc in scaler_cols:
                min_v, max_v = tract_clean[sc].min(), tract_clean[sc].max()
                if max_v > min_v:
                    tract_clean[f"{sc}_norm"] = (tract_clean[sc] - min_v) / (max_v - min_v)
                else:
                    tract_clean[f"{sc}_norm"] = 0

            tract_clean["demand_score"] = (
                tract_clean["pop_25_34_pct_norm"] * 0.5
                + tract_clean["renter_pct_norm"] * 0.3
                + (1 - tract_clean["rent_to_income"].clip(0, 60) / 60) * 0.2
            ) * 100

            # Top tracts
            top_tracts = tract_clean.nlargest(20, "demand_score")

            # Simplify tract name
            top_tracts["tract_label"] = top_tracts["NAME"].str.replace(
                r"Census Tract (\d+\.?\d*),.*", r"Tract \1", regex=True
            )

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                fig = build_bar_chart(
                    top_tracts["tract_label"].tolist()[:15],
                    top_tracts["demand_score"].tolist()[:15],
                    "Top 15 Tracts — Young Professional Demand Score",
                    color=COLORS["gold"],
                    y_label="Score",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_t2:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=tract_clean["B25064_001E"],
                        y=tract_clean["pop_25_34_pct"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=tract_clean["demand_score"],
                            colorscale=[[0, COLORS["slate"]], [0.5, COLORS["teal"]], [1, COLORS["gold"]]],
                            colorbar=dict(title="Demand<br>Score"),
                            opacity=0.7,
                        ),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Rent: $%{x:,.0f}<br>"
                            "% Age 25-34: %{y:.1f}%<br>"
                            "Score: %{customdata[1]:.0f}<extra></extra>"
                        ),
                        customdata=np.column_stack(
                            (tract_clean["NAME"].values, tract_clean["demand_score"].values)
                        ),
                    )
                )
                fig.update_layout(
                    **CHART_LAYOUT,
                    title=dict(text="All Philly Tracts — Rent vs Young Adult %", font=dict(size=16)),
                    xaxis_title="Median Gross Rent ($)",
                    yaxis_title="% Population Age 25–34",
                    xaxis_tickprefix="$",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            with st.expander("📋 View Top 20 Tracts — Detailed Data"):
                display_cols = [
                    "tract_label",
                    "demand_score",
                    "pop_25_34",
                    "pop_25_34_pct",
                    "renter_pct",
                    "B25064_001E",
                    "B19013_001E",
                    "rent_to_income",
                ]
                display_names = {
                    "tract_label": "Tract",
                    "demand_score": "Demand Score",
                    "pop_25_34": "Pop 25-34",
                    "pop_25_34_pct": "% 25-34",
                    "renter_pct": "% Renters",
                    "B25064_001E": "Med. Rent ($)",
                    "B19013_001E": "Med. HH Income ($)",
                    "rent_to_income": "Rent/Income %",
                }
                display_df = top_tracts[display_cols].rename(columns=display_names)
                st.dataframe(
                    display_df.style.format(
                        {
                            "Demand Score": "{:.0f}",
                            "Pop 25-34": "{:,.0f}",
                            "% 25-34": "{:.1f}%",
                            "% Renters": "{:.1f}%",
                            "Med. Rent ($)": "${:,.0f}",
                            "Med. HH Income ($)": "${:,.0f}",
                            "Rent/Income %": "{:.1f}%",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_demographics:
    st.markdown('<div class="section-label">Demographic Trends — Philadelphia County</div>', unsafe_allow_html=True)

    # Population over time
    pop_df = fetch_fred_series("PAPHIL5POP", "2000-01-01")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig = build_time_series_chart(
            pop_df,
            "Resident Population — Philadelphia County (Thousands)",
            y_label="Population",
            color=COLORS["teal"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d2:
        labor_df = fetch_fred_series("LAUCN421010000000006", start_date)
        fig = build_time_series_chart(
            labor_df,
            "Civilian Labor Force — Philadelphia County",
            y_label="Labor Force",
            color=COLORS["lavender"],
        )
        st.plotly_chart(fig, use_container_width=True)

    if CENSUS_API_KEY:
        st.markdown('<div class="section-label">Age Distribution — County Comparison</div>', unsafe_allow_html=True)

        # Age breakdown
        age_vars = [
            "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",  # M <5, 5-9, 10-14, 15-17
            "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",  # M 18-19, 20, 21, 22-24
            "B01001_011E", "B01001_012E", "B01001_013E", "B01001_014E",  # M 25-29, 30-34, 35-39, 40-44
            "B01001_015E", "B01001_016E", "B01001_017E", "B01001_018E",  # M 45-49, 50-54, 55-59, 60-61
            "B01001_019E", "B01001_020E", "B01001_021E", "B01001_022E",  # M 62-64, 65-66, 67-69, 70-74
            "B01001_023E", "B01001_024E", "B01001_025E",                 # M 75-79, 80-84, 85+
            "B01001_027E", "B01001_028E", "B01001_029E", "B01001_030E",  # F <5, 5-9, 10-14, 15-17
            "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",  # F 18-19, 20, 21, 22-24
            "B01001_035E", "B01001_036E", "B01001_037E", "B01001_038E",  # F 25-29, 30-34, 35-39, 40-44
            "B01001_039E", "B01001_040E", "B01001_041E", "B01001_042E",  # F 45-49, 50-54, 55-59, 60-61
            "B01001_043E", "B01001_044E", "B01001_045E", "B01001_046E",  # F 62-64, 65-66, 67-69, 70-74
            "B01001_047E", "B01001_048E", "B01001_049E",                 # F 75-79, 80-84, 85+
            "B01003_001E",  # Total pop
        ]

        selected_fips = {k: v for k, v in COUNTY_FIPS.items() if k in selected_counties}
        age_data = fetch_census_acs_county_multi(age_vars, selected_fips)

        if not age_data.empty:
            for col in age_vars:
                age_data[col] = pd.to_numeric(age_data[col], errors="coerce")

            # Aggregate into age groups
            age_data["under_18"] = (
                age_data[["B01001_003E","B01001_004E","B01001_005E","B01001_006E",
                          "B01001_027E","B01001_028E","B01001_029E","B01001_030E"]].sum(axis=1)
            )
            age_data["age_18_24"] = (
                age_data[["B01001_007E","B01001_008E","B01001_009E","B01001_010E",
                          "B01001_031E","B01001_032E","B01001_033E","B01001_034E"]].sum(axis=1)
            )
            age_data["age_25_34"] = (
                age_data[["B01001_011E","B01001_012E","B01001_035E","B01001_036E"]].sum(axis=1)
            )
            age_data["age_35_44"] = (
                age_data[["B01001_013E","B01001_014E","B01001_037E","B01001_038E"]].sum(axis=1)
            )
            age_data["age_45_54"] = (
                age_data[["B01001_015E","B01001_016E","B01001_039E","B01001_040E"]].sum(axis=1)
            )
            age_data["age_55_64"] = (
                age_data[["B01001_017E","B01001_018E","B01001_019E",
                          "B01001_041E","B01001_042E","B01001_043E"]].sum(axis=1)
            )
            age_data["age_65_plus"] = (
                age_data[["B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
                          "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E"]].sum(axis=1)
            )

            total_pop = age_data["B01003_001E"]
            age_groups = ["under_18", "age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_64", "age_65_plus"]
            age_labels = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
            group_colors = [COLORS["steel"], COLORS["sky"], COLORS["gold"], COLORS["teal"],
                            COLORS["lavender"], COLORS["coral"], COLORS["sand"]]

            fig = go.Figure()
            for grp, lbl, clr in zip(age_groups, age_labels, group_colors):
                pcts = (age_data[grp] / total_pop * 100).tolist()
                fig.add_trace(
                    go.Bar(
                        x=age_data["county_label"],
                        y=pcts,
                        name=lbl,
                        marker_color=clr,
                        hovertemplate=f"<b>%{{x}}</b><br>{lbl}: %{{y:.1f}}%<extra></extra>",
                    )
                )
            fig.update_layout(
                **CHART_LAYOUT,
                barmode="stack",
                title=dict(text="Age Distribution by County (% of Population)", font=dict(size=16)),
                yaxis_ticksuffix="%",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Education
        st.markdown('<div class="section-label">Education Attainment (25+)</div>', unsafe_allow_html=True)

        edu_vars = [
            "B15003_001E",  # Total 25+
            "B15003_022E",  # Bachelor's
            "B15003_023E",  # Master's
            "B15003_024E",  # Professional
            "B15003_025E",  # Doctorate
            "B15003_017E",  # HS diploma
            "B15003_018E",  # GED
        ]

        edu_data = fetch_census_acs_county_multi(edu_vars, selected_fips)

        if not edu_data.empty:
            for col in edu_vars:
                edu_data[col] = pd.to_numeric(edu_data[col], errors="coerce")

            edu_data["bachelors_plus_pct"] = (
                (edu_data["B15003_022E"] + edu_data["B15003_023E"] + edu_data["B15003_024E"] + edu_data["B15003_025E"])
                / edu_data["B15003_001E"]
                * 100
            )
            edu_data["hs_pct"] = (
                (edu_data["B15003_017E"] + edu_data["B15003_018E"]) / edu_data["B15003_001E"] * 100
            )

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                sorted_edu = edu_data.sort_values("bachelors_plus_pct", ascending=True)
                fig = build_bar_chart(
                    sorted_edu["county_label"].tolist(),
                    sorted_edu["bachelors_plus_pct"].tolist(),
                    "Bachelor's Degree or Higher (%)",
                    color=COLORS["teal"],
                    horizontal=True,
                    y_label="% of 25+",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_e2:
                sorted_edu = edu_data.sort_values("hs_pct", ascending=True)
                fig = build_bar_chart(
                    sorted_edu["county_label"].tolist(),
                    sorted_edu["hs_pct"].tolist(),
                    "HS Diploma / GED Only (%)",
                    color=COLORS["sand"],
                    horizontal=True,
                    y_label="% of 25+",
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: REGIONAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab_regional:
    st.markdown('<div class="section-label">Regional Market Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        "Comparing Philadelphia County with surrounding markets: Montgomery, Bucks, Delaware, Chester, "
        "Berks (Reading), and South Jersey (Camden, Burlington, Gloucester counties)."
        "</div>",
        unsafe_allow_html=True,
    )

    if CENSUS_API_KEY:
        # Comprehensive comparison
        comp_vars = [
            "B01003_001E",  # Total pop
            "B19013_001E",  # Median HH income
            "B25064_001E",  # Median gross rent
            "B25003_001E",  # Total tenure
            "B25003_003E",  # Renters
            "B25077_001E",  # Median home value
            "B01001_011E", "B01001_012E", "B01001_035E", "B01001_036E",  # 25-34
            "B23025_003E",  # In labor force
            "B23025_005E",  # Unemployed
            "B23025_002E",  # In labor force (total)
        ]

        all_fips = COUNTY_FIPS.copy()
        comp_data = fetch_census_acs_county_multi(comp_vars, all_fips)

        if not comp_data.empty:
            for col in comp_vars:
                comp_data[col] = pd.to_numeric(comp_data[col], errors="coerce")

            comp_data["pop_25_34"] = (
                comp_data["B01001_011E"] + comp_data["B01001_012E"]
                + comp_data["B01001_035E"] + comp_data["B01001_036E"]
            )
            comp_data["pop_25_34_pct"] = comp_data["pop_25_34"] / comp_data["B01003_001E"] * 100
            comp_data["renter_pct"] = comp_data["B25003_003E"] / comp_data["B25003_001E"] * 100
            comp_data["rent_to_income_annual"] = (comp_data["B25064_001E"] * 12) / comp_data["B19013_001E"] * 100
            comp_data["local_unemp"] = comp_data["B23025_005E"] / comp_data["B23025_002E"] * 100

            # Summary comparison table
            st.markdown('<div class="section-label">Comparison Scorecard</div>', unsafe_allow_html=True)

            display_comp = comp_data[[
                "county_label", "B01003_001E", "B19013_001E", "B25064_001E",
                "B25077_001E", "pop_25_34_pct", "renter_pct", "rent_to_income_annual", "local_unemp",
            ]].rename(columns={
                "county_label": "County",
                "B01003_001E": "Population",
                "B19013_001E": "Med. HH Income",
                "B25064_001E": "Med. Rent",
                "B25077_001E": "Med. Home Value",
                "pop_25_34_pct": "% Age 25-34",
                "renter_pct": "% Renters",
                "rent_to_income_annual": "Rent/Income %",
                "local_unemp": "Unemp. Rate %",
            }).sort_values("Population", ascending=False)

            st.dataframe(
                display_comp.style.format({
                    "Population": "{:,.0f}",
                    "Med. HH Income": "${:,.0f}",
                    "Med. Rent": "${:,.0f}",
                    "Med. Home Value": "${:,.0f}",
                    "% Age 25-34": "{:.1f}%",
                    "% Renters": "{:.1f}%",
                    "Rent/Income %": "{:.1f}%",
                    "Unemp. Rate %": "{:.1f}%",
                }).background_gradient(cmap="YlOrRd", subset=["% Age 25-34", "% Renters"]),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("")

            # Radar / parallel coordinates
            col_p1, col_p2 = st.columns(2)

            with col_p1:
                sorted_comp = comp_data.sort_values("B25077_001E", ascending=True)
                fig = build_bar_chart(
                    sorted_comp["county_label"].tolist(),
                    sorted_comp["B25077_001E"].tolist(),
                    "Median Home Value by County",
                    color=COLORS["teal"],
                    horizontal=True,
                    y_label="Home Value ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_p2:
                sorted_comp = comp_data.sort_values("local_unemp", ascending=True)
                fig = build_bar_chart(
                    sorted_comp["county_label"].tolist(),
                    sorted_comp["local_unemp"].tolist(),
                    "Local Unemployment Rate (%)",
                    color=COLORS["coral"],
                    horizontal=True,
                    y_label="Unemp %",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Opportunity Score
            st.markdown('<div class="section-label">Blue-Collar Rental Opportunity Score</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">'
                "This composite score weighs: young adult concentration (25%), renter prevalence (25%), "
                "rent affordability (25%), and employment strength (25%). Higher = stronger opportunity "
                "for quality blue-collar rental demand."
                "</div>",
                unsafe_allow_html=True,
            )

            # Normalize each factor 0-1
            factors = {
                "pop_25_34_pct": True,    # Higher = better
                "renter_pct": True,        # Higher = better (more renters)
                "rent_to_income_annual": False,  # Lower = better (more affordable)
                "local_unemp": False,      # Lower = better
            }

            for col, higher_better in factors.items():
                min_v = comp_data[col].min()
                max_v = comp_data[col].max()
                if max_v > min_v:
                    normalized = (comp_data[col] - min_v) / (max_v - min_v)
                    if not higher_better:
                        normalized = 1 - normalized
                    comp_data[f"{col}_score"] = normalized
                else:
                    comp_data[f"{col}_score"] = 0.5

            comp_data["opportunity_score"] = (
                comp_data["pop_25_34_pct_score"] * 25
                + comp_data["renter_pct_score"] * 25
                + comp_data["rent_to_income_annual_score"] * 25
                + comp_data["local_unemp_score"] * 25
            )

            comp_sorted = comp_data.sort_values("opportunity_score", ascending=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=comp_sorted["county_label"],
                    x=comp_sorted["opportunity_score"],
                    orientation="h",
                    marker=dict(
                        color=comp_sorted["opportunity_score"],
                        colorscale=[[0, COLORS["slate"]], [0.5, COLORS["teal"]], [1, COLORS["gold"]]],
                    ),
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>",
                )
            )
            fig.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Blue-Collar Rental Opportunity Score (0–100)", font=dict(size=16)),
                xaxis_title="Opportunity Score",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Score breakdown
            with st.expander("📋 Score Breakdown by County"):
                score_display = comp_data[[
                    "county_label",
                    "pop_25_34_pct_score", "renter_pct_score",
                    "rent_to_income_annual_score", "local_unemp_score",
                    "opportunity_score",
                ]].rename(columns={
                    "county_label": "County",
                    "pop_25_34_pct_score": "Young Adults (25pt)",
                    "renter_pct_score": "Renter Prevalence (25pt)",
                    "rent_to_income_annual_score": "Rent Affordability (25pt)",
                    "local_unemp_score": "Employment Strength (25pt)",
                    "opportunity_score": "Total Score",
                }).sort_values("Total Score", ascending=False)

                for col in score_display.columns[1:]:
                    score_display[col] = score_display[col].apply(
                        lambda x: f"{x:.1f}" if col == "Total Score" else f"{x*25:.1f}"
                    )

                st.dataframe(score_display, use_container_width=True, hide_index=True)
    else:
        st.info("Enter your Census API key in the sidebar to load regional comparison data.")


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #5A6270; font-size: 0.8rem; padding: 1rem 0;">
        <b style="color:#C8A951;">Lapstone Intel</b> · Philadelphia Construction & Real Estate Intelligence<br>
        Data: Federal Reserve (FRED) · U.S. Census Bureau (ACS) · Bureau of Labor Statistics (QCEW)<br>
        Dashboard refreshes every 6 hours · Last loaded: {timestamp}
    </div>
    """.format(timestamp=datetime.now().strftime("%B %d, %Y at %I:%M %p")),
    unsafe_allow_html=True,
)
