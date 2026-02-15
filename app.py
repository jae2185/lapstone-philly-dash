"""
Lapstone LLC — Philadelphia Construction & Real Estate Intelligence Dashboard
With forecasting engine and metric tooltips.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, json, zipfile, io, os
from datetime import datetime

# ── CONFIG ──
def _secret(k):
    try: return st.secrets[k]
    except Exception: return os.environ.get(k, "")

FRED_API_KEY = _secret("FRED_API_KEY")
CENSUS_API_KEY = _secret("CENSUS_API_KEY")

COUNTY_FIPS = {
    "Philadelphia":("42","101"),"Montgomery":("42","091"),"Bucks":("42","017"),
    "Delaware":("42","045"),"Chester":("42","029"),"Berks (Reading)":("42","011"),
    "Camden (NJ)":("34","007"),"Burlington (NJ)":("34","005"),"Gloucester (NJ)":("34","015"),
}
FRED_SERIES = {
    "unemp_philly":"PAPHIL5URN","pop_philly":"PAPHIL5POP",
    "labor_force":"PAPHIL5LFN","med_income":"MHIPA42101A052NCEN",
    "homeown":"HOWNRATEACS042101","unemp_msa":"PHIL942URN",
    "const_emp":"SMU42979611500000001SA","nonfarm":"SMU42379800000000001SA",
    "permits_tot":"PHIL942BPPRIV","permits_1u":"PHIL942BP1FH",
    "gdp":"NGMP37980","cpi_shelter":"CUURA102SAH1","cpi_all":"CUURA102SA0",
    # Interest rates & credit conditions
    "fed_funds":"FEDFUNDS","treasury_10y":"GS10","mortgage_30y":"MORTGAGE30US",
    "treasury_2y":"GS2","spread_10y2y":"T10Y2Y",
}
C = {"gold":"#C8A951","slate":"#4A6274","teal":"#2EC4B6","coral":"#E76F51",
     "lavender":"#9B8EC7","sky":"#48A9A6","sand":"#D4A373","steel":"#7F8C9B",
     "bg":"#0E1117","card":"#1A1D26","text":"#E8E8E8","muted":"#7F8C9B"}
BL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"],family="DM Sans, sans-serif"),
    margin=dict(l=40,r=20,t=50,b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)",zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)",zeroline=False),
    hoverlabel=dict(bgcolor=C["card"],font_size=12,font_family="DM Sans"))

TRACT_VARS = [
    "B01001_011E","B01001_012E","B01001_035E","B01001_036E","B01003_001E",
    "B25064_001E","B19013_001E","B25003_001E","B25003_003E",
    "B25070_007E","B25070_008E","B25070_009E","B25070_010E","B25070_011E","B25070_001E",
    "B23025_002E","B23025_005E",
    # Migration / Geographic Mobility (B07001)
    "B07001_001E",  # Total pop 1yr+
    "B07001_017E",  # Same house (non-movers)
    "B07001_033E",  # Moved within same county
    "B07001_049E",  # Moved from different county, same state
    "B07001_065E",  # Moved from different state
    "B07001_081E",  # Moved from abroad
]

# Migration variables for county-level (used in forecasting)
MIGRATION_COUNTY_VARS = [
    "B07001_001E","B07001_017E","B07001_033E","B07001_049E","B07001_065E","B07001_081E",
]

# ── TOOLTIP DEFINITIONS ──
TT = {
    "unemp": "**Unemployment Rate** — % of the civilian labor force without a job but actively seeking work. Source: BLS via FRED.",
    "const_jobs": "**Construction Jobs** — Total employees in mining, logging, and construction sectors for the Philadelphia metro area. Source: BLS CES via FRED.",
    "permits": "**Building Permits** — Number of new privately-owned housing units authorized by building permits in the past 12 months. A leading indicator of future construction activity. Source: Census via FRED.",
    "gdp": "**Gross Domestic Product** — Total market value of all goods and services produced in the Philadelphia-Camden-Wilmington MSA. Source: BEA via FRED.",
    "homeown": "**Homeownership Rate** — % of occupied housing units that are owner-occupied. Lower rates = more renters = stronger rental demand. Source: Census ACS via FRED.",
    "pct2534": "**% Age 25–34** — Share of the population in this key young professional demographic. Higher = stronger rental demand signal.",
    "renter_pct": "**Renter-Occupied %** — Share of housing units occupied by renters (vs owners). Higher = established rental market.",
    "med_rent": "**Median Gross Rent** — Middle value of monthly rent including utilities. Source: Census ACS 5-Year (B25064).",
    "med_income": "**Median Household Income** — Middle HH income before taxes. Used to gauge affordability when compared to rent. Source: ACS (B19013).",
    "burden": "**Rent-Burdened** — % of renter households paying 30%+ of income on rent. HUD defines this as cost-burdened. Source: ACS (B25070).",
    "r2i": "**Rent-to-Income Ratio** — Annual rent ÷ annual income × 100. Under 30% is generally considered affordable.",
    "demand_score": "**Demand Score** — Custom composite: 50% young adult concentration + 30% renter prevalence + 20% affordability (inverse rent/income). Range 0–100.",
    "opp_score": "**Opportunity Score** — Composite: 25% young adults + 25% renter share + 25% affordability + 25% employment strength. Identifies best counties for blue-collar rental investment.",
    "cpi_shelter": "**Shelter CPI** — Consumer price index for housing costs (rent, owners' equivalent rent). YoY % change shows how fast housing costs are rising.",
    "forecast": "**Forecast Model** — Uses Ridge regression on lagged FRED indicators (unemployment, permits, CPI, GDP) to predict next-year values. Backtested with expanding-window walk-forward validation.",
    "opp_pred": "**Opportunity Prediction** — Ranks census tracts by projected investment potential. Combines current fundamentals (demand, affordability), momentum (ACS year-over-year changes), and macro forecast direction into a single forward-looking score.",
    "fed_funds": "**Federal Funds Rate** — The interest rate at which banks lend reserves to each other overnight. Set by the Federal Reserve. Drives all other borrowing costs. Source: Federal Reserve via FRED.",
    "mortgage_30y": "**30-Year Mortgage Rate** — Average rate on a 30-year fixed-rate mortgage. Directly impacts housing affordability and construction financing costs. Source: Freddie Mac via FRED.",
    "treasury_10y": "**10-Year Treasury Yield** — Benchmark long-term rate. Cap rates, commercial loan pricing, and CMBS spreads all key off this. Source: Federal Reserve via FRED.",
    "yield_curve": "**Yield Curve Spread (10Y–2Y)** — Difference between 10-year and 2-year Treasury yields. Negative = inverted curve, historically a recession leading indicator. Source: FRED.",
    "rate_sensitivity": "**Rate Sensitivity Analysis** — Shows how ±100 basis point shifts in interest rates would affect forecast predictions, based on learned model coefficients.",
    "migration": "**Migration / Geographic Mobility** — Measures population movement in and out of an area over the past year. Inflow rate = % of current residents who moved in from outside the county. Net inflow signals growing demand; net outflow signals declining demand. Source: Census ACS B07001.",
    "inflow_rate": "**Inflow Rate** — % of population 1yr+ who moved into the area from a different county, state, or country in the past year. Higher = more new residents arriving.",
    "turnover_rate": "**Turnover Rate** — % of population 1yr+ who moved at all (including within same county). High turnover can signal either a dynamic market or instability.",
    "out_of_state": "**Out-of-State Inflow** — % of population who moved in from a different state. High values signal the area is attracting talent/residents from other regions.",
}

def tip(key):
    """Return a small help icon with tooltip text."""
    return TT.get(key, "")

# ── DATA FETCHING ──
@st.cache_data(ttl=3600*6, show_spinner=False)
def fetch_fred(sid, start="2010-01-01"):
    if not FRED_API_KEY: return pd.DataFrame(columns=["date","value"])
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
            params={"series_id":sid,"api_key":FRED_API_KEY,"file_type":"json",
                    "observation_start":start,"sort_order":"asc"}, timeout=20)
        if r.status_code!=200: return pd.DataFrame(columns=["date","value"])
        obs = r.json().get("observations",[])
        if not obs: return pd.DataFrame(columns=["date","value"])
        df = pd.DataFrame(obs); df["date"]=pd.to_datetime(df["date"])
        df["value"]=pd.to_numeric(df["value"],errors="coerce")
        return df.dropna(subset=["value"])[["date","value"]].reset_index(drop=True)
    except Exception: return pd.DataFrame(columns=["date","value"])

@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_acs_counties(variables, cfips, year=2023):
    if not CENSUS_API_KEY: return pd.DataFrame()
    frames=[]
    for name,(st_,co) in cfips.items():
        try:
            r=requests.get(f"https://api.census.gov/data/{year}/acs/acs5",
                params={"get":f"NAME,{','.join(variables)}","for":f"county:{co}",
                        "in":f"state:{st_}","key":CENSUS_API_KEY}, timeout=15)
            if r.status_code==200:
                d=r.json()
                if len(d)>=2:
                    row=pd.DataFrame(d[1:],columns=d[0]); row["county_label"]=name; frames.append(row)
        except Exception: pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_acs_tracts(variables, state="42", county="101", year=2023):
    if not CENSUS_API_KEY: return pd.DataFrame()
    try:
        r=requests.get(f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{','.join(variables)}&for=tract:*&in=state:{state}&in=county:{county}&key={CENSUS_API_KEY}", timeout=30)
        if r.status_code!=200: return pd.DataFrame()
        d=r.json(); return pd.DataFrame(d[1:],columns=d[0]) if len(d)>=2 else pd.DataFrame()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_multi_tracts(variables, cdict, year=2023):
    frames=[]
    for name,(st_,co) in cdict.items():
        df=fetch_acs_tracts(variables,st_,co,year)
        if not df.empty: df["county_label"]=name; frames.append(df)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=3600*24, show_spinner="Loading tract boundaries…")
def load_geojson(state_fips="42"):
    try: import geopandas as gpd
    except ImportError: return None
    url=f"https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_{state_fips}_tract_500k.zip"
    try:
        r=requests.get(url,timeout=60)
        if r.status_code!=200: return None
        tmp=f"/tmp/tracts_{state_fips}"; os.makedirs(tmp,exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z: z.extractall(tmp)
        shp=[f for f in os.listdir(tmp) if f.endswith(".shp")][0]
        gdf=gpd.read_file(f"{tmp}/{shp}")
        gdf["GEOID"]=gdf["STATEFP"]+gdf["COUNTYFP"]+gdf["TRACTCE"]
        return json.loads(gdf.to_json())
    except Exception: return None

@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_qcew(area="42101",year="2024",qtr="1"):
    try: return pd.read_csv(f"https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area}.csv",dtype=str,timeout=20)
    except Exception: return pd.DataFrame()

# ── CHART HELPERS ──
def lchart(df,title,yl="",color=C["gold"],trend=False,yp="",ys=""):
    if df.empty:
        fig=go.Figure(); fig.update_layout(**BL,title=dict(text=title,font=dict(size=16)))
        fig.add_annotation(text="No data available",showarrow=False,font=dict(size=14,color=C["muted"])); return fig
    r_,g_,b_=int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["date"],y=df["value"],mode="lines",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({r_},{g_},{b_},0.08)",
        hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{yl}: {yp}%{{y:,.1f}}{ys}<extra></extra>"))
    if trend and len(df)>12:
        s=df.sort_values("date")
        fig.add_trace(go.Scatter(x=s["date"],y=s["value"].rolling(12,min_periods=6).mean(),
            mode="lines",line=dict(color=C["muted"],width=1.5,dash="dot"),name="12-mo avg",hoverinfo="skip"))
    ex={}
    if yp=="$": ex["yaxis_tickprefix"]="$"
    if ys=="%": ex["yaxis_ticksuffix"]="%"
    fig.update_layout(**BL,title=dict(text=title,font=dict(size=16)),**ex); return fig

def bchart(names,vals,title,color=C["gold"],horiz=False,yl="",yp=""):
    fig=go.Figure()
    if horiz: fig.add_trace(go.Bar(y=names,x=vals,orientation="h",marker_color=color,hovertemplate=f"<b>%{{y}}</b><br>{yl}: {yp}%{{x:,.1f}}<extra></extra>"))
    else: fig.add_trace(go.Bar(x=names,y=vals,marker_color=color,hovertemplate=f"<b>%{{x}}</b><br>{yl}: {yp}%{{y:,.1f}}<extra></extra>"))
    fig.update_layout(**BL,title=dict(text=title,font=dict(size=16))); return fig

def compute_tract_metrics(df):
    df = df.copy()
    for col in TRACT_VARS:
        df[col]=pd.to_numeric(df[col],errors="coerce")
        df.loc[df[col]<=-666666666,col]=np.nan
    df["pop2534"]=df[["B01001_011E","B01001_012E","B01001_035E","B01001_036E"]].sum(axis=1)
    df["pct2534"]=df["pop2534"]/df["B01003_001E"].replace(0,np.nan)*100
    df["renter_pct"]=df["B25003_003E"]/df["B25003_001E"].replace(0,np.nan)*100
    df["burden_pct"]=df[["B25070_008E","B25070_009E","B25070_010E","B25070_011E"]].sum(axis=1)/df["B25070_001E"].replace(0,np.nan)*100
    df["r2i"]=(df["B25064_001E"]*12)/df["B19013_001E"].replace(0,np.nan)*100
    df["unemp"]=df["B23025_005E"]/df["B23025_002E"].replace(0,np.nan)*100
    # Migration metrics
    mob_total = df["B07001_001E"].replace(0, np.nan)
    if "B07001_033E" in df.columns:
        df["moved_within"] = df["B07001_033E"] / mob_total * 100  # within county
        df["inflow_rate"] = (df["B07001_049E"].fillna(0) + df["B07001_065E"].fillna(0) + df["B07001_081E"].fillna(0)) / mob_total * 100
        df["out_of_state_in"] = df["B07001_065E"] / mob_total * 100
        df["from_abroad"] = df["B07001_081E"] / mob_total * 100
        df["turnover"] = (mob_total - df["B07001_017E"]) / mob_total * 100  # anyone who moved
    return df

def compute_demand_score(df):
    v=df.dropna(subset=["pct2534","B25064_001E","B19013_001E"]).copy()
    v=v[(v["B01003_001E"]>200)&(v["pct2534"]<100)]
    for s in ["pct2534","renter_pct"]:
        mn,mx=v[s].min(),v[s].max()
        v[f"{s}_n"]=((v[s]-mn)/(mx-mn)) if mx>mn else 0
    v["score"]=(v["pct2534_n"]*0.5+v["renter_pct_n"]*0.3+(1-v["r2i"].clip(0,60)/60)*0.2)*100
    return v

# ── FORECASTING ENGINE ──
@st.cache_data(ttl=3600*24, show_spinner=False)
def fetch_migration_timeseries(state="42", county="101", years=None):
    """Fetch county-level ACS migration data across multiple years for forecasting."""
    if not CENSUS_API_KEY: return pd.DataFrame()
    if years is None:
        years = list(range(2010, 2024))  # ACS 5-year available from ~2009 onward
    rows = []
    for yr in years:
        try:
            r = requests.get(f"https://api.census.gov/data/{yr}/acs/acs5",
                params={"get": f"NAME,{','.join(MIGRATION_COUNTY_VARS)}",
                        "for": f"county:{county}", "in": f"state:{state}",
                        "key": CENSUS_API_KEY}, timeout=15)
            if r.status_code == 200:
                d = r.json()
                if len(d) >= 2:
                    row = dict(zip(d[0], d[1]))
                    for v in MIGRATION_COUNTY_VARS:
                        row[v] = float(row.get(v, 0)) if row.get(v) not in [None, "", "-666666666"] else np.nan
                    total = row.get("B07001_001E", np.nan)
                    if total and total > 0:
                        same_house = row.get("B07001_017E", 0) or 0
                        within_county = row.get("B07001_033E", 0) or 0
                        diff_county = row.get("B07001_049E", 0) or 0
                        diff_state = row.get("B07001_065E", 0) or 0
                        abroad = row.get("B07001_081E", 0) or 0
                        inflow = diff_county + diff_state + abroad
                        rows.append({
                            "year": yr,
                            "inflow_rate": (inflow / total) * 100,
                            "turnover_rate": ((total - same_house) / total) * 100,
                            "out_of_state_rate": (diff_state / total) * 100,
                            "within_county_rate": (within_county / total) * 100,
                        })
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── ZILLOW / HOUSING MARKET INDICES (via FRED + CSV fallback) ──
# ZHVI is available on FRED at state level; FHFA HPI available at metro level
# ZORI is NOT on FRED — try multiple Zillow CSV URL patterns
FRED_ZHVI_PA = "PAUCSFRCONDOSMSAMID"  # ZHVI Pennsylvania (monthly)
FRED_HPI_PHILLY = "ATNHPIUS37964Q"    # FHFA All-Transactions HPI Philadelphia MSAD (quarterly)

# Multiple ZORI CSV URL patterns (Zillow changes these frequently)
ZORI_URLS = [
    "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_sm_month.csv",
    "https://files.zillowstatic.com/research/public_csvs/zori/Metro_ZORI_AllHomesPlusMultifamily_Smoothed.csv",
    "https://files.zillowstatic.com/research/public_v2/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv",
    "https://files.zillowstatic.com/research/public_v2/zori/Metro_zori_sm_month.csv",
]

@st.cache_data(ttl=3600*24, show_spinner=False)
def fetch_zillow_from_fred():
    """Fetch ZHVI (PA state) and HPI (Philly metro) from FRED — reliable source."""
    zhvi = fetch_fred(FRED_ZHVI_PA, "2000-01-01")
    hpi = fetch_fred(FRED_HPI_PHILLY, "2000-01-01")
    return zhvi, hpi

@st.cache_data(ttl=3600*24, show_spinner=False)
def fetch_zori_csv():
    """Try multiple Zillow CSV URL patterns for ZORI metro data."""
    for url in ZORI_URLS:
        try:
            df = pd.read_csv(url, timeout=10)
            philly = df[df["RegionName"].str.contains("Philadelphia", case=False, na=False)]
            if philly.empty: continue
            date_cols = [c for c in philly.columns if c[:4].isdigit()]
            if not date_cols: continue
            row = philly.iloc[0]
            ts = pd.DataFrame({"date": pd.to_datetime(date_cols), "zori": [row[c] for c in date_cols]})
            ts = ts.dropna()
            if not ts.empty:
                return ts
        except Exception:
            continue
    return pd.DataFrame()

# ── IRS SOI COUNTY-TO-COUNTY MIGRATION ──
@st.cache_data(ttl=3600*24, show_spinner=False)
def fetch_irs_soi_migration():
    """Fetch IRS SOI county-to-county migration data for Philadelphia.
    Returns inflow and outflow DataFrames with returns, exemptions, and AGI."""
    base_url = "https://www.irs.gov/pub/irs-soi"
    # IRS publishes county-to-county inflow/outflow CSVs per year pair
    # Format: countyinflow{yy1}{yy2}.csv / countyoutflow{yy1}{yy2}.csv
    # Philadelphia County FIPS: 42101
    years = []
    for y1 in range(2011, 2022):  # 2011-12 through 2021-22
        y2 = y1 + 1
        yy1, yy2 = str(y1)[2:], str(y2)[2:]
        years.append((y1, y2, yy1, yy2))

    inflow_records = []
    outflow_records = []
    for y1, y2, yy1, yy2 in years:
        for direction, records in [("inflow", inflow_records), ("outflow", outflow_records)]:
            try:
                url = f"{base_url}/county{direction}{yy1}{yy2}.csv"
                df = pd.read_csv(url, encoding="latin-1", dtype=str)
                # Normalize column names (they vary across years)
                df.columns = [c.strip().upper() for c in df.columns]
                # Rename common variations
                col_map = {}
                for c in df.columns:
                    if "Y1_STATEFIPS" in c or c == "Y1_STATEFIPS": col_map[c] = "Y1_STATEFIPS"
                    elif "Y2_STATEFIPS" in c or c == "Y2_STATEFIPS": col_map[c] = "Y2_STATEFIPS"
                    elif "Y1_COUNTYFIPS" in c or c == "Y1_COUNTYFIPS": col_map[c] = "Y1_COUNTYFIPS"
                    elif "Y2_COUNTYFIPS" in c or c == "Y2_COUNTYFIPS": col_map[c] = "Y2_COUNTYFIPS"
                    elif c in ("N1", "RETURN"): col_map[c] = "N1"
                    elif c in ("N2", "EXEMPTION"): col_map[c] = "N2"
                    elif c in ("AGI", "AGI_ADJ"): col_map[c] = "AGI"
                df = df.rename(columns=col_map)

                if direction == "inflow":
                    # Filter: destination is Philadelphia (Y2 = 42, 101)
                    mask = (df.get("Y2_STATEFIPS", pd.Series(dtype=str)).str.strip() == "42") & \
                           (df.get("Y2_COUNTYFIPS", pd.Series(dtype=str)).str.strip() == "101")
                else:
                    # Filter: origin is Philadelphia (Y1 = 42, 101)
                    mask = (df.get("Y1_STATEFIPS", pd.Series(dtype=str)).str.strip() == "42") & \
                           (df.get("Y1_COUNTYFIPS", pd.Series(dtype=str)).str.strip() == "101")

                filtered = df[mask].copy()
                if filtered.empty: continue

                # Aggregate: total returns, exemptions, AGI flowing in/out
                for col in ["N1", "N2", "AGI"]:
                    if col in filtered.columns:
                        filtered[col] = pd.to_numeric(filtered[col].str.replace(",", ""), errors="coerce")

                # Exclude "same state non-migrants" and totals rows (county=000 or 999)
                county_col = "Y1_COUNTYFIPS" if direction == "inflow" else "Y2_COUNTYFIPS"
                if county_col in filtered.columns:
                    filtered = filtered[~filtered[county_col].str.strip().isin(["000", "999", "-1"])]

                agg = {}
                for col in ["N1", "N2", "AGI"]:
                    if col in filtered.columns:
                        agg[col] = filtered[col].sum()
                agg["year"] = y2  # Filing year (when people filed, reflecting prior year move)
                agg["period"] = f"{y1}-{y2}"
                records.append(agg)
            except Exception:
                continue

    inflow_df = pd.DataFrame(inflow_records) if inflow_records else pd.DataFrame()
    outflow_df = pd.DataFrame(outflow_records) if outflow_records else pd.DataFrame()

    # Compute net migration
    if not inflow_df.empty and not outflow_df.empty:
        merged = inflow_df.merge(outflow_df, on=["year", "period"], suffixes=("_in", "_out"))
        for col in ["N1", "N2", "AGI"]:
            in_col, out_col = f"{col}_in", f"{col}_out"
            if in_col in merged.columns and out_col in merged.columns:
                merged[f"{col}_net"] = merged[in_col] - merged[out_col]
        return inflow_df, outflow_df, merged

    return inflow_df, outflow_df, pd.DataFrame()

def build_annual_dataset(fd, migration_df=None, zillow_zori_metro=None, zhvi_fred=None):
    """Convert monthly FRED series into annual features for forecasting."""
    targets = {
        "Unemployment (%)": "unemp_philly",
        "Permits (Annual)": "permits_tot",
        "GDP ($M)": "gdp",
        "Median Income ($)": "med_income",
    }
    features_keys = ["unemp_philly","permits_tot","gdp","cpi_shelter","cpi_all","const_emp","homeown","med_income",
                      "fed_funds","treasury_10y","mortgage_30y","spread_10y2y"]
    frames = {}
    for key in features_keys:
        df = fd.get(key, pd.DataFrame())
        if df.empty: continue
        tmp = df.copy(); tmp["year"] = tmp["date"].dt.year
        if key == "permits_tot":
            agg = tmp.groupby("year")["value"].sum().rename(key)
        else:
            agg = tmp.groupby("year")["value"].last().rename(key)
        frames[key] = agg
    # Add migration features
    if migration_df is not None and not migration_df.empty:
        mig = migration_df.set_index("year")
        for col in ["inflow_rate", "turnover_rate", "out_of_state_rate"]:
            if col in mig.columns:
                frames[col] = mig[col]
    # Add Zillow ZORI as monthly-to-annual feature
    if zillow_zori_metro is not None and not zillow_zori_metro.empty:
        zori = zillow_zori_metro.copy()
        zori["year"] = zori["date"].dt.year
        zori_annual = zori.groupby("year")["zori"].last().rename("zori_msa")
        frames["zori_msa"] = zori_annual
    # Add ZHVI (Zillow Home Value Index) from FRED
    if zhvi_fred is not None and not zhvi_fred.empty:
        zhvi = zhvi_fred.copy()
        zhvi["year"] = zhvi["date"].dt.year
        zhvi_annual = zhvi.groupby("year")["value"].last().rename("zhvi_pa")
        frames["zhvi_pa"] = zhvi_annual
    if not frames: return pd.DataFrame(), targets
    annual = pd.concat(frames.values(), axis=1).dropna(how="all")
    # Exclude current incomplete year to avoid partial-year bias
    current_year = datetime.now().year
    annual = annual[annual.index < current_year]
    return annual, targets

def _run_single_forecast(clean, target_col, feature_cols, n_backtest=5, horizons=(1, 2, 5), detrend=False):
    """Core forecast engine. If detrend=True, removes linear trend before modeling."""
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler

    y_raw = clean[target_col].copy()
    trend_coef, trend_intercept = 0.0, 0.0

    if detrend:
        # Fit linear trend on target
        t = np.arange(len(y_raw)).reshape(-1, 1)
        lr = LinearRegression().fit(t, y_raw.values)
        trend_coef = lr.coef_[0]
        trend_intercept = lr.intercept_
        trend_line = lr.predict(t)
        # Detrend: model predicts residuals from trend
        clean = clean.copy()
        clean[target_col] = y_raw.values - trend_line

    lagged = clean.shift(1)
    lagged.columns = [f"{c}_lag1" for c in feature_cols]
    combo = pd.concat([clean[[target_col]], lagged], axis=1).iloc[1:]
    combo = combo.replace([np.inf, -np.inf], np.nan).dropna()
    if len(combo) < 6: return None
    X = combo.drop(columns=[target_col])
    y = combo[target_col]
    feat_names = X.columns.tolist()

    max_h = max(horizons)
    bt_all = []
    min_train = max(5, len(combo) - n_backtest - max_h)
    for split in range(min_train, len(combo) - 1):
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        sc_bt = StandardScaler(); X_tr_s = sc_bt.fit_transform(X_tr)
        m_bt = Ridge(alpha=1.0); m_bt.fit(X_tr_s, y_tr)
        sim_row = clean[feature_cols].iloc[split - 1 + 1:split + 1].copy()
        if sim_row.empty: continue
        sim_row = sim_row.iloc[-1:].copy()
        for h in range(1, max_h + 1):
            future_idx = split + h - 1
            if future_idx >= len(combo): break
            inp = sim_row.copy()
            inp.columns = [f"{c}_lag1" for c in feature_cols]
            inp = inp[feat_names].fillna(0)
            pred_h = m_bt.predict(sc_bt.transform(inp))[0]
            # Re-add trend for comparison against actuals
            if detrend:
                trend_at_target = trend_coef * (split + h) + trend_intercept
                pred_actual = pred_h + trend_at_target
            else:
                pred_actual = pred_h
            actual_h = y_raw.iloc[split + h] if (split + h) < len(y_raw) else None
            if actual_h is not None:
                bt_all.append({
                    "year_origin": y.index[split - 1] if split > 0 else y.index[0],
                    "year_target": y.index[future_idx],
                    "horizon": h, "predicted": pred_actual, "actual": actual_h,
                })
            sim_row = sim_row.copy()
            if target_col in sim_row.columns:
                sim_row[target_col] = pred_h  # feed back detrended prediction

    bt_full = pd.DataFrame(bt_all)
    if bt_full.empty: return None
    bt_full["error"] = bt_full["predicted"] - bt_full["actual"]
    bt_full["abs_pct_error"] = (bt_full["error"].abs() / bt_full["actual"].abs().clip(lower=0.01)) * 100

    horizon_accuracy = {}
    for h in horizons:
        bh = bt_full[bt_full["horizon"] == h]
        if not bh.empty:
            horizon_accuracy[h] = {
                "mape": bh["abs_pct_error"].mean(),
                "rmse": np.sqrt((bh["error"] ** 2).mean()),
                "n": len(bh),
            }

    bt1 = bt_full[bt_full["horizon"] == 1].copy()
    bt1 = bt1.rename(columns={"year_target": "year"})
    mape = bt1["abs_pct_error"].mean() if not bt1.empty else 0
    rmse = np.sqrt((bt1["error"] ** 2).mean()) if not bt1.empty else 0
    std_err = bt1["error"].std() if len(bt1) > 1 else abs(y_raw.iloc[-1]) * 0.05

    # Train final model on all data
    sc = StandardScaler(); X_s = sc.fit_transform(X); m = Ridge(alpha=1.0); m.fit(X_s, y)
    coefs = pd.Series(m.coef_, index=feat_names).abs().sort_values(ascending=False)

    # Multi-horizon forecast
    last_known = clean[feature_cols].iloc[-1:].copy()
    base_year = int(clean.index[-1])
    n_total = len(y_raw)
    forecasts = []
    simulated_row = last_known.copy()
    for h in range(1, max_h + 1):
        inp = simulated_row.copy()
        inp.columns = [f"{c}_lag1" for c in feature_cols]
        inp = inp[feat_names].fillna(0)
        pred = m.predict(sc.transform(inp))[0]
        if detrend:
            pred_actual = pred + trend_coef * (n_total + h) + trend_intercept
        else:
            pred_actual = pred
        h_err = std_err
        if h in horizon_accuracy and horizon_accuracy[h]["n"] > 1:
            bh = bt_full[bt_full["horizon"] == h]
            h_err = bh["error"].std()
        forecasts.append({
            "year": base_year + h, "value": pred_actual,
            "ci_low": pred_actual - 1.96 * h_err, "ci_high": pred_actual + 1.96 * h_err,
            "horizon": h,
        })
        simulated_row = simulated_row.copy()
        if target_col in simulated_row.columns:
            simulated_row[target_col] = pred

    fc_df = pd.DataFrame(forecasts)
    horizon_results = {}
    for h in horizons:
        row = fc_df[fc_df["horizon"] == h]
        if not row.empty:
            r = row.iloc[0]
            horizon_results[h] = {
                "year": int(r["year"]), "value": r["value"],
                "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            }

    # Compute weighted MAPE across all horizons for model selection
    weighted_mape = 0
    total_w = 0
    for h, w in [(1, 3), (2, 2), (5, 1)]:  # weight short-term more heavily
        if h in horizon_accuracy:
            weighted_mape += horizon_accuracy[h]["mape"] * w
            total_w += w
    weighted_mape = weighted_mape / total_w if total_w > 0 else mape

    return {
        "backtest": bt1, "backtest_full": bt_full,
        "horizon_accuracy": horizon_accuracy,
        "mape": mape, "rmse": rmse, "weighted_mape": weighted_mape,
        "forecasts": fc_df, "horizon_results": horizon_results,
        "last_actual_year": base_year, "last_actual_val": y_raw.iloc[-1],
        "importance": coefs, "n_train": len(combo), "std_err": std_err,
        "detrended": detrend,
        "trend_coef": trend_coef, "trend_intercept": trend_intercept,
    }

def run_forecast(annual, target_col, n_backtest=5, horizons=(1, 2, 5)):
    """Run standard and trend-adjusted forecasts, auto-select the better model by backtest."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    if target_col not in annual.columns or len(annual) < 8:
        return None

    # Smart feature selection: only include migration features if they don't cost too many rows
    migration_cols = ["inflow_rate", "turnover_rate", "out_of_state_rate"]
    mig_available = [c for c in migration_cols if c in annual.columns]

    if mig_available:
        without_mig = annual.drop(columns=mig_available, errors="ignore").ffill().bfill().dropna(axis=1)
        with_mig = annual.ffill().bfill().dropna(axis=1)
        n_without = len(without_mig.dropna())
        n_with = len(with_mig.dropna())
        rows_lost = n_without - n_with
        if n_with >= 12 and rows_lost <= n_without * 0.3:
            clean = with_mig
            used_migration = True
        else:
            clean = without_mig
            used_migration = False
    else:
        clean = annual.ffill().bfill().dropna(axis=1)
        used_migration = False

    if target_col not in clean.columns or len(clean) < 8:
        return None

    feature_cols = [c for c in clean.columns]

    # Run both models
    result_standard = _run_single_forecast(clean, target_col, feature_cols, n_backtest, horizons, detrend=False)
    result_trend = _run_single_forecast(clean, target_col, feature_cols, n_backtest, horizons, detrend=True)

    # Select winner by weighted MAPE (favoring short-term accuracy)
    if result_standard is None and result_trend is None:
        return None
    elif result_standard is None:
        best = result_trend
    elif result_trend is None:
        best = result_standard
    else:
        if result_trend["weighted_mape"] < result_standard["weighted_mape"]:
            best = result_trend
        else:
            best = result_standard

    best["used_migration"] = used_migration
    best["model_type"] = "trend-adjusted" if best.get("detrended") else "standard"
    # Include comparison info
    if result_standard and result_trend:
        best["model_comparison"] = {
            "standard_mape": result_standard["weighted_mape"],
            "trend_mape": result_trend["weighted_mape"],
            "selected": best["model_type"],
        }
    return best

# ── OPPORTUNITY AREA PREDICTION ──
@st.cache_data(ttl=3600*12, show_spinner=False)
def fetch_acs_tracts_year(variables, state="42", county="101", year=2023):
    """Fetch tract data for a specific ACS year."""
    return fetch_acs_tracts(variables, state, county, year)

def compute_tract_opportunity(tracts_current, tracts_prior, macro_forecasts, horizon=1):
    """
    Score each tract on forward-looking investment potential.
    Components:
      1. Fundamentals (40%): demand score, renter %, young adults
      2. Momentum (30%): YoY improvement in rent, income, young adult share
         - Decays for longer horizons (momentum is less predictive further out)
      3. Affordability Headroom (20%): low rent-to-income = room for rent growth
      4. Macro Alignment (10% short-term, 15% long-term): macro forecast direction
    """
    curr = tracts_current.copy()
    if curr.empty: return pd.DataFrame()

    # Adjust weights by horizon — momentum matters less at 5yr, macro matters more
    if horizon <= 1:
        w_fund, w_mom, w_head, w_macro = 0.40, 0.30, 0.20, 0.10
    elif horizon <= 2:
        w_fund, w_mom, w_head, w_macro = 0.40, 0.25, 0.22, 0.13
    else:
        w_fund, w_mom, w_head, w_macro = 0.42, 0.18, 0.25, 0.15

    # -- Fundamentals (reuse demand score) --
    scored = compute_demand_score(curr)
    if scored.empty: return pd.DataFrame()
    fund_min, fund_max = scored["score"].min(), scored["score"].max()
    scored["fund_n"] = (scored["score"] - fund_min) / (fund_max - fund_min + 0.01)

    # -- Momentum: compare to prior year --
    if not tracts_prior.empty:
        prior = compute_tract_metrics(tracts_prior.copy())
        prior["GEOID_p"] = prior["state"] + prior["county"] + prior["tract"]
        prior_cols = {"GEOID_p": "GEOID", "pct2534": "pct2534_prev", "B25064_001E": "rent_prev",
                      "B19013_001E": "income_prev", "renter_pct": "renter_prev"}
        pm = prior[list(prior_cols.keys())].rename(columns=prior_cols)
        scored = scored.merge(pm, on="GEOID", how="left")
        scored["rent_growth"] = ((scored["B25064_001E"] - scored["rent_prev"]) / scored["rent_prev"].clip(lower=1)) * 100
        scored["income_growth"] = ((scored["B19013_001E"] - scored["income_prev"]) / scored["income_prev"].clip(lower=1)) * 100
        scored["youth_growth"] = scored["pct2534"] - scored["pct2534_prev"]
        for mc in ["rent_growth", "income_growth", "youth_growth"]:
            scored[mc] = scored[mc].clip(-50, 50)
            mn, mx = scored[mc].min(), scored[mc].max()
            scored[f"{mc}_n"] = (scored[mc] - mn) / (mx - mn + 0.01)
        scored["momentum"] = (scored["rent_growth_n"] * 0.4 + scored["income_growth_n"] * 0.35 + scored["youth_growth_n"] * 0.25)
    else:
        scored["momentum"] = 0.5
        scored["rent_growth"] = np.nan
        scored["income_growth"] = np.nan
        scored["youth_growth"] = np.nan

    # -- Affordability Headroom --
    scored["headroom"] = 1 - (scored["r2i"].clip(0, 60) / 60)
    hr_min, hr_max = scored["headroom"].min(), scored["headroom"].max()
    scored["headroom_n"] = (scored["headroom"] - hr_min) / (hr_max - hr_min + 0.01)

    # -- Macro Alignment (horizon-specific) --
    macro_boost = 0.5
    if macro_forecasts:
        signals = []
        if "Permits (Annual)" in macro_forecasts:
            pf = macro_forecasts["Permits (Annual)"]
            if pf and pf.get("forecast_val", 0) > pf.get("last_actual_val", 0):
                signals.append(0.6)
            else:
                signals.append(0.4)
        if "GDP ($M)" in macro_forecasts:
            gf = macro_forecasts["GDP ($M)"]
            if gf and gf.get("forecast_val", 0) > gf.get("last_actual_val", 0):
                signals.append(0.7)
            else:
                signals.append(0.3)
        if "Unemployment (%)" in macro_forecasts:
            uf = macro_forecasts["Unemployment (%)"]
            if uf and uf.get("forecast_val", 0) < uf.get("last_actual_val", 0):
                signals.append(0.7)
            else:
                signals.append(0.3)
        if signals:
            macro_boost = np.mean(signals)
    scored["macro_n"] = macro_boost

    # -- Migration Signal (if available) --
    has_migration = "inflow_rate" in scored.columns and scored["inflow_rate"].notna().sum() > 10
    if has_migration:
        inf_min, inf_max = scored["inflow_rate"].min(), scored["inflow_rate"].max()
        scored["inflow_n"] = (scored["inflow_rate"] - inf_min) / (inf_max - inf_min + 0.01)
        # Adjust weights to include migration
        w_mig = 0.08
        w_fund_adj = w_fund - 0.02
        w_mom_adj = w_mom - 0.03
        w_head_adj = w_head - 0.01
        w_macro_adj = w_macro - 0.02
        scored["opp_pred"] = (
            scored["fund_n"] * w_fund_adj +
            scored["momentum"] * w_mom_adj +
            scored["headroom_n"] * w_head_adj +
            scored["macro_n"] * w_macro_adj +
            scored["inflow_n"] * w_mig
        ) * 100
    else:
        scored["opp_pred"] = (
            scored["fund_n"] * w_fund +
            scored["momentum"] * w_mom +
            scored["headroom_n"] * w_head +
            scored["macro_n"] * w_macro
        ) * 100

    return scored

# ── PAGE SETUP ──
st.set_page_config(page_title="Lapstone Intel — Philly Dashboard", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');
.stApp{font-family:'DM Sans',sans-serif}
.dashboard-header{background:linear-gradient(135deg,#1A1D26,#0E1117);border:1px solid rgba(200,169,81,.2);border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem}
.dashboard-header h1{color:#C8A951;font-size:2rem;font-weight:700;margin:0;letter-spacing:-.5px}
.dashboard-header p{color:#7F8C9B;font-size:.95rem;margin:.25rem 0 0}
[data-testid="stMetric"]{background:#1A1D26;border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:1rem 1.25rem}
[data-testid="stMetricLabel"]{color:#7F8C9B!important;font-size:.8rem!important;text-transform:uppercase;letter-spacing:.5px}
[data-testid="stMetricValue"]{color:#E8E8E8!important;font-family:'Space Mono',monospace!important;font-size:1.6rem!important}
[data-testid="stMetricDelta"]{font-family:'Space Mono',monospace!important}
.stTabs [data-baseweb="tab-list"]{gap:.5rem}
.stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0;padding:.6rem 1.2rem;color:#7F8C9B}
.stTabs [aria-selected="true"]{color:#C8A951!important;border-bottom:2px solid #C8A951}
section[data-testid="stSidebar"]{background:#12151C;border-right:1px solid rgba(255,255,255,.06)}
.section-label{color:#C8A951;font-size:.75rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin:2rem 0 .5rem;padding-bottom:.3rem;border-bottom:1px solid rgba(200,169,81,.15)}
.info-box{background:rgba(200,169,81,.06);border-left:3px solid #C8A951;border-radius:0 8px 8px 0;padding:.75rem 1rem;font-size:.85rem;color:#A8ADB5;margin:.5rem 0}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
div.block-container{padding-top:1.5rem}
.stPlotlyChart{border:1px solid rgba(255,255,255,.04);border-radius:10px;overflow:hidden}
</style>""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🏗️ Lapstone Intel")
    st.markdown('<p style="color:#7F8C9B;font-size:.85rem;">Philadelphia Construction & Real Estate Intelligence</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)
    if not FRED_API_KEY:
        fi=st.text_input("FRED API Key",type="password",help="Free at https://fred.stlouisfed.org/docs/api/api_key.html")
        if fi: FRED_API_KEY=fi; os.environ["FRED_API_KEY"]=fi; st.rerun()
    else: st.success("✓ FRED API connected")
    if not CENSUS_API_KEY:
        ci=st.text_input("Census API Key",type="password",help="Free at https://api.census.gov/data/key_signup.html")
        if ci: CENSUS_API_KEY=ci; os.environ["CENSUS_API_KEY"]=ci; st.rerun()
    else: st.success("✓ Census API connected")
    st.caption("+ Zillow ZORI/ZHVI · IRS SOI Migration")
    st.divider()
    st.markdown('<div class="section-label">Time Range</div>', unsafe_allow_html=True)
    start_year=st.slider("Start year",2010,2024,2015)
    start_date=f"{start_year}-01-01"
    st.divider()
    st.markdown('<div class="section-label">Regional Comparison</div>', unsafe_allow_html=True)
    sel_counties=st.multiselect("Compare counties",list(COUNTY_FIPS.keys()),default=["Philadelphia","Montgomery","Bucks","Berks (Reading)","Delaware"])
    st.divider()
    st.markdown('<div style="color:#5A6270;font-size:.75rem;margin-top:1rem"><b>Data Sources</b><br>• FRED • Census ACS • BLS QCEW<br><br>Built for <a href="https://www.lapstonellc.com" target="_blank" style="color:#C8A951;">Lapstone LLC</a></div>', unsafe_allow_html=True)

st.markdown('<div class="dashboard-header"><h1>Philadelphia Metro Intelligence</h1><p>Construction economy, rental demand, and demographic analytics for the greater Philadelphia region</p></div>', unsafe_allow_html=True)

# ── PREFETCH ──
fd={}
if FRED_API_KEY:
    with st.spinner("Loading FRED data…"):
        for k,sid in FRED_SERIES.items(): fd[k]=fetch_fred(sid,start_date)
sel_fips={k:v for k,v in COUNTY_FIPS.items() if k in sel_counties}

t_ov,t_con,t_rent,t_maps,t_demo,t_reg,t_fc=st.tabs(["📊 Overview","🏗️ Construction","🏠 Rental Demand","🗺️ Tract Maps","👥 Demographics","📍 Regional","🔮 Forecast"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW (with tooltips)
# ══════════════════════════════════════════════════════════════════════════════
with t_ov:
    if not FRED_API_KEY: st.info("👈 Enter your **FRED API key** in the sidebar.")
    st.markdown('<div class="section-label">Key Indicators</div>',unsafe_allow_html=True)
    def slat(k):
        df=fd.get(k,pd.DataFrame())
        if df.empty: return None,None
        la=df.iloc[-1]["value"]; pr=df.iloc[-13]["value"] if len(df)>13 else (df.iloc[-2]["value"] if len(df)>1 else None)
        return la,pr
    c1,c2,c3,c4,c5=st.columns(5)
    with c1:
        v,p=slat("unemp_philly"); d=f"{v-p:+.1f}pp YoY" if v is not None and p is not None else None
        st.metric("Unemployment Rate",f"{v:.1f}%" if v else "—",d,delta_color="inverse",help=tip("unemp"))
    with c2:
        v,p=slat("const_emp"); d=f"{((v/p)-1)*100:+.1f}% YoY" if v and p else None
        st.metric("Construction Jobs",f"{v:,.1f}K" if v else "—",d,help=tip("const_jobs"))
    with c3:
        dp=fd.get("permits_tot",pd.DataFrame())
        if not dp.empty and len(dp)>=12:
            r12=dp.tail(12)["value"].sum(); p12=dp.iloc[-24:-12]["value"].sum() if len(dp)>=24 else None
            d=f"{((r12/p12)-1)*100:+.1f}% YoY" if p12 and p12>0 else None
            st.metric("Permits (12-mo)",f"{r12:,.0f}",d,help=tip("permits"))
        else: st.metric("Permits","—",help=tip("permits"))
    with c4:
        v,p=slat("gdp"); d=f"{((v/p)-1)*100:+.1f}% YoY" if v and p else None
        def fnum(n):
            if n is None: return "—"
            if n>=1e3: return f"${n/1e3:,.0f}B"
            return f"${n:,.0f}M"
        st.metric("GDP (Philly MSA)",fnum(v),d,help=tip("gdp"))
    with c5:
        v,p=slat("homeown"); d=f"{v-p:+.1f}pp" if v is not None and p is not None else None
        st.metric("Homeownership",f"{v:.1f}%" if v else "—",d,delta_color="off",help=tip("homeown"))
    st.markdown("")
    cl,cr=st.columns(2)
    with cl: st.plotly_chart(lchart(fd.get("unemp_philly",pd.DataFrame()),"Unemployment — Philadelphia County","Rate",C["coral"],True,ys="%"),use_container_width=True)
    with cr: st.plotly_chart(lchart(fd.get("const_emp",pd.DataFrame()),"Construction Employment — Philadelphia (K)","Jobs",C["teal"],True),use_container_width=True)
    cl2,cr2=st.columns(2)
    with cl2: st.plotly_chart(lchart(fd.get("permits_tot",pd.DataFrame()),"Building Permits — Philly MSA (Monthly)","Permits",C["gold"],True),use_container_width=True)
    with cr2: st.plotly_chart(lchart(fd.get("gdp",pd.DataFrame()),"GDP — Philly MSA ($M)","GDP",C["lavender"],yp="$"),use_container_width=True)
    st.markdown('<div class="section-label">Inflation & Income</div>',unsafe_allow_html=True)
    cc1,cc2=st.columns(2)
    with cc1:
        sh=fd.get("cpi_shelter",pd.DataFrame()); al=fd.get("cpi_all",pd.DataFrame())
        if not sh.empty and not al.empty:
            s=sh.sort_values("date").copy(); a=al.sort_values("date").copy()
            s["yoy"]=s["value"].pct_change(12)*100; a["yoy"]=a["value"].pct_change(12)*100
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=s["date"],y=s["yoy"],mode="lines",name="Shelter CPI YoY%",line=dict(color=C["coral"],width=2.5)))
            fig.add_trace(go.Scatter(x=a["date"],y=a["yoy"],mode="lines",name="All Items CPI YoY%",line=dict(color=C["steel"],width=2)))
            fig.update_layout(**BL,title=dict(text="Philly MSA — CPI Year-over-Year (%)",font=dict(size=16)),yaxis_ticksuffix="%")
            st.plotly_chart(fig,use_container_width=True)
    with cc2: st.plotly_chart(lchart(fd.get("med_income",pd.DataFrame()),"Median HH Income — Philly County","Income",C["sky"],yp="$"),use_container_width=True)
    # ── Rates & Credit Conditions ──
    st.markdown('<div class="section-label">Rates & Credit Conditions</div>',unsafe_allow_html=True)
    rc1,rc2,rc3,rc4=st.columns(4)
    with rc1:
        v,p=slat("fed_funds"); d=f"{v-p:+.2f}pp YoY" if v is not None and p is not None else None
        st.metric("Fed Funds Rate",f"{v:.2f}%" if v else "—",d,delta_color="inverse",help=tip("fed_funds"))
    with rc2:
        v,p=slat("mortgage_30y"); d=f"{v-p:+.2f}pp YoY" if v is not None and p is not None else None
        st.metric("30-Yr Mortgage",f"{v:.2f}%" if v else "—",d,delta_color="inverse",help=tip("mortgage_30y"))
    with rc3:
        v,p=slat("treasury_10y"); d=f"{v-p:+.2f}pp YoY" if v is not None and p is not None else None
        st.metric("10-Yr Treasury",f"{v:.2f}%" if v else "—",d,delta_color="inverse",help=tip("treasury_10y"))
    with rc4:
        v,p=slat("spread_10y2y"); d=f"{v-p:+.2f}pp" if v is not None and p is not None else None
        inv_warn = " ⚠️" if v is not None and v < 0 else ""
        st.metric("Yield Curve (10Y–2Y)",f"{v:+.2f}%{inv_warn}" if v else "—",d,delta_color="normal",help=tip("yield_curve"))
    rr1,rr2=st.columns(2)
    with rr1:
        ff=fd.get("fed_funds",pd.DataFrame()); mg=fd.get("mortgage_30y",pd.DataFrame())
        if not ff.empty and not mg.empty:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ff["date"],y=ff["value"],mode="lines",name="Fed Funds",line=dict(color=C["coral"],width=2.5)))
            fig.add_trace(go.Scatter(x=mg["date"],y=mg["value"],mode="lines",name="30-Yr Mortgage",line=dict(color=C["gold"],width=2.5)))
            t10=fd.get("treasury_10y",pd.DataFrame())
            if not t10.empty:
                fig.add_trace(go.Scatter(x=t10["date"],y=t10["value"],mode="lines",name="10-Yr Treasury",line=dict(color=C["teal"],width=2)))
            fig.update_layout(**BL,title=dict(text="Interest Rates",font=dict(size=16)),yaxis_ticksuffix="%")
            st.plotly_chart(fig,use_container_width=True)
    with rr2:
        sp=fd.get("spread_10y2y",pd.DataFrame())
        if not sp.empty:
            s=sp.sort_values("date").copy()
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=s["date"],y=s["value"],mode="lines",line=dict(color=C["lavender"],width=2.5),
                fill="tozeroy",fillcolor="rgba(155,142,199,0.08)",
                hovertemplate="<b>%{x|%b %Y}</b><br>Spread: %{y:.2f}%<extra></extra>"))
            fig.add_hline(y=0,line=dict(color=C["coral"],width=1.5,dash="dash"))
            fig.add_annotation(x=s["date"].iloc[len(s)//2],y=-0.3,text="← Inverted (recession signal)",
                font=dict(size=10,color=C["coral"]),showarrow=False)
            fig.update_layout(**BL,title=dict(text="Yield Curve Spread (10Y–2Y)",font=dict(size=16)),yaxis_ticksuffix="%")
            st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
with t_con:
    st.markdown('<div class="section-label">Construction Deep Dive</div>',unsafe_allow_html=True)
    co1,co2=st.columns(2)
    with co1:
        tot=fd.get("permits_tot",pd.DataFrame()); u1=fd.get("permits_1u",pd.DataFrame())
        if not tot.empty and not u1.empty:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=tot["date"],y=tot["value"],mode="lines",name="Total",line=dict(color=C["gold"],width=2.5)))
            fig.add_trace(go.Scatter(x=u1["date"],y=u1["value"],mode="lines",name="Single-Family",line=dict(color=C["teal"],width=2)))
            m=tot.merge(u1,on="date",suffixes=("_t","_1")); m["multi"]=m["value_t"]-m["value_1"]
            fig.add_trace(go.Scatter(x=m["date"],y=m["multi"],mode="lines",name="Multi-Family (2+)",line=dict(color=C["coral"],width=2)))
            fig.update_layout(**BL,title=dict(text="Permits by Structure Type — Philly MSA",font=dict(size=16)))
            st.plotly_chart(fig,use_container_width=True)
    with co2:
        ce=fd.get("const_emp",pd.DataFrame()); nf=fd.get("nonfarm",pd.DataFrame())
        if not ce.empty and not nf.empty:
            m=ce.merge(nf,on="date",suffixes=("_c","_t")); m["pct"]=(m["value_c"]/m["value_t"])*100
            fig=go.Figure(go.Scatter(x=m["date"],y=m["pct"],mode="lines",line=dict(color=C["gold"],width=2.5),
                fill="tozeroy",fillcolor="rgba(200,169,81,0.08)",hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.2f}%<extra></extra>"))
            fig.update_layout(**BL,title=dict(text="Construction % of Employment",font=dict(size=16)),yaxis_ticksuffix="%")
            st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section-label">BLS QCEW — Construction Industry</div>',unsafe_allow_html=True)
    qy=st.selectbox("Year",["2024","2023","2022","2021","2020"],index=0)
    qcew=fetch_qcew("42101",qy,"1")
    if not qcew.empty and "industry_code" in qcew.columns:
        codes={"1012":"Construction (Total)","1013":"Construction of Buildings","1014":"Heavy & Civil Engineering","1015":"Specialty Trade Contractors"}
        rows=qcew[(qcew["industry_code"].isin(codes))&(qcew["own_code"]=="5")].copy()
        if not rows.empty:
            for c_ in ["annual_avg_emplvl","avg_annual_pay","annual_avg_estabs_count"]:
                if c_ in rows.columns: rows[c_]=pd.to_numeric(rows[c_],errors="coerce")
            rows["label"]=rows["industry_code"].map(codes)
            tot_r=rows[rows["industry_code"]=="1012"]
            if not tot_r.empty:
                mc1,mc2,mc3=st.columns(3)
                with mc1: st.metric("Employment",f"{tot_r.iloc[0].get('annual_avg_emplvl',0):,.0f}")
                with mc2: st.metric("Avg Annual Pay",f"${tot_r.iloc[0].get('avg_annual_pay',0):,.0f}")
                with mc3: st.metric("Establishments",f"{tot_r.iloc[0].get('annual_avg_estabs_count',0):,.0f}")
            sub=rows[rows["industry_code"]!="1012"]
            if not sub.empty:
                sc1,sc2=st.columns(2)
                with sc1: st.plotly_chart(bchart(sub["label"].tolist(),sub["annual_avg_emplvl"].tolist(),f"Employment — {qy}",C["teal"],yl="Employees"),use_container_width=True)
                with sc2:
                    if "avg_annual_pay" in sub.columns: st.plotly_chart(bchart(sub["label"].tolist(),sub["avg_annual_pay"].tolist(),f"Avg Pay — {qy}",C["gold"],yl="Pay",yp="$"),use_container_width=True)
    st.markdown('<div class="section-label">Annual Permit Trend</div>',unsafe_allow_html=True)
    pm=fd.get("permits_tot",pd.DataFrame())
    if not pm.empty:
        a=pm.copy(); a["yr"]=a["date"].dt.year; asum=a.groupby("yr")["value"].sum().reset_index()
        asum=asum[asum["yr"]>=start_year]
        fig=go.Figure(go.Bar(x=asum["yr"],y=asum["value"],
            marker_color=[C["gold"] if y==asum["yr"].max() else C["slate"] for y in asum["yr"]],
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"))
        fig.update_layout(**BL,title=dict(text="Annual Permits — Philly MSA",font=dict(size=16)))
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: RENTAL DEMAND (with tooltips)
# ══════════════════════════════════════════════════════════════════════════════
with t_rent:
    st.markdown('<div class="section-label">Rental Market — Young Professional Demand</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Target Avatar:</b> Young professionals (25–34), employed, responsible renters. This section identifies where they cluster and tracks affordability.</div>',unsafe_allow_html=True)
    if not CENSUS_API_KEY: st.info("👈 Enter your Census API key in the sidebar.")
    else:
        cdata=fetch_acs_counties(TRACT_VARS,sel_fips)
        if not cdata.empty:
            cdata=compute_tract_metrics(cdata)
            pr=cdata[cdata["county_label"]=="Philadelphia"]
            if not pr.empty:
                pr=pr.iloc[0]; m1,m2,m3,m4=st.columns(4)
                with m1: st.metric("Young Adults (25–34)",f"{pr['pop2534']:,.0f}",help=tip("pct2534"))
                with m2: st.metric("% Population 25–34",f"{pr['pct2534']:.1f}%",help=tip("pct2534"))
                with m3: st.metric("Median Rent",f"${pr['B25064_001E']:,.0f}",help=tip("med_rent"))
                with m4: st.metric("% Renters",f"{pr['renter_pct']:.1f}%",help=tip("renter_pct"))
            st.markdown("")
            r1,r2=st.columns(2)
            with r1:
                s=cdata.sort_values("pct2534",ascending=True)
                st.plotly_chart(bchart(s["county_label"].tolist(),s["pct2534"].tolist(),"% Age 25–34",C["teal"],True,"% pop"),use_container_width=True)
            with r2:
                s=cdata.sort_values("renter_pct",ascending=True)
                st.plotly_chart(bchart(s["county_label"].tolist(),s["renter_pct"].tolist(),"Renter-Occupied %",C["coral"],True,"% renter"),use_container_width=True)
            r3,r4=st.columns(2)
            with r3:
                s=cdata.sort_values("B25064_001E",ascending=True)
                st.plotly_chart(bchart(s["county_label"].tolist(),s["B25064_001E"].tolist(),"Median Rent by County",C["gold"],True,"Rent","$"),use_container_width=True)
            with r4:
                s=cdata.sort_values("burden_pct",ascending=True)
                st.plotly_chart(bchart(s["county_label"].tolist(),s["burden_pct"].tolist(),"Rent-Burdened (30%+ Income)",C["lavender"],True,"Burdened %"),use_container_width=True)
            st.markdown('<div class="section-label">Affordability Matrix</div>',unsafe_allow_html=True)
            fig=go.Figure(go.Scatter(x=cdata["B19013_001E"],y=cdata["B25064_001E"],mode="markers+text",
                text=cdata["county_label"],textposition="top center",textfont=dict(size=11,color=C["text"]),
                marker=dict(size=cdata["pct2534"]*3,color=cdata["renter_pct"],
                    colorscale=[[0,C["slate"]],[1,C["gold"]]],colorbar=dict(title="Renter%"),
                    line=dict(width=1,color="rgba(255,255,255,0.2)")),
                hovertemplate="<b>%{text}</b><br>Income: $%{x:,.0f}<br>Rent: $%{y:,.0f}<extra></extra>"))
            fig.update_layout(**BL,title=dict(text="Rent vs Income (Bubble=Young Adult %)",font=dict(size=16)),
                xaxis_title="Median HH Income",yaxis_title="Median Rent",xaxis_tickprefix="$",yaxis_tickprefix="$")
            st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="section-label">Tract-Level Hotspots (Philadelphia)</div>',unsafe_allow_html=True)
        tdf=fetch_acs_tracts(TRACT_VARS,"42","101")
        if not tdf.empty:
            tdf=compute_tract_metrics(tdf); v=compute_demand_score(tdf)
            top=v.nlargest(25,"score").copy()
            top["lbl"]=top["NAME"].str.replace(r"Census Tract (\d+\.?\d*),.*",r"Tract \1",regex=True)
            tc1,tc2=st.columns(2)
            with tc1: st.plotly_chart(bchart(top["lbl"].tolist()[:15],top["score"].tolist()[:15],"Top 15 — Demand Score",C["gold"],yl="Score"),use_container_width=True)
            with tc2:
                fig=go.Figure(go.Scatter(x=v["B25064_001E"],y=v["pct2534"],mode="markers",
                    marker=dict(size=5,color=v["score"],opacity=0.7,
                        colorscale=[[0,C["slate"]],[0.5,C["teal"]],[1,C["gold"]]],colorbar=dict(title="Score")),
                    customdata=np.column_stack((v["NAME"].values,v["score"].values)),
                    hovertemplate="<b>%{customdata[0]}</b><br>Rent: $%{x:,.0f}<br>%25-34: %{y:.1f}%<br>Score: %{customdata[1]:.0f}<extra></extra>"))
                fig.update_layout(**BL,title=dict(text="All Tracts — Rent vs Young Adult %",font=dict(size=16)),
                    xaxis_title="Median Rent ($)",yaxis_title="% Age 25–34",xaxis_tickprefix="$")
                st.plotly_chart(fig,use_container_width=True)
            with st.expander("📋 Top 25 Tracts Data"):
                d=top[["lbl","score","pop2534","pct2534","renter_pct","B25064_001E","B19013_001E","r2i"]].rename(columns={
                    "lbl":"Tract","score":"Score","pop2534":"Pop 25-34","pct2534":"% 25-34",
                    "renter_pct":"% Renters","B25064_001E":"Med Rent","B19013_001E":"Med Income","r2i":"Rent/Inc %"})
                st.dataframe(d.style.format({"Score":"{:.0f}","Pop 25-34":"{:,.0f}","% 25-34":"{:.1f}%",
                    "% Renters":"{:.1f}%","Med Rent":"${:,.0f}","Med Income":"${:,.0f}","Rent/Inc %":"{:.1f}%"}),
                    use_container_width=True,hide_index=True)

        # Migration / Mobility Section
        st.markdown('<div class="section-label">Migration & Geographic Mobility</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{tip("migration")}</div>',unsafe_allow_html=True)
        if not tdf.empty and "inflow_rate" in tdf.columns:
            tdf_mig = tdf.dropna(subset=["inflow_rate"]).copy()
            if not tdf_mig.empty:
                mm1,mm2,mm3,mm4=st.columns(4)
                with mm1: st.metric("Avg Inflow Rate",f"{tdf_mig['inflow_rate'].mean():.1f}%",help=tip("inflow_rate"))
                with mm2: st.metric("Avg Turnover",f"{tdf_mig['turnover'].mean():.1f}%",help=tip("turnover_rate"))
                with mm3: st.metric("Avg Out-of-State In",f"{tdf_mig['out_of_state_in'].mean():.1f}%",help=tip("out_of_state"))
                with mm4: st.metric("Avg From Abroad",f"{tdf_mig['from_abroad'].mean():.1f}%")
                mr1,mr2=st.columns(2)
                with mr1:
                    top_inflow = tdf_mig.nlargest(15,"inflow_rate").copy()
                    top_inflow["lbl"]=top_inflow["NAME"].str.replace(r"Census Tract (\d+\.?\d*),.*",r"Tract \1",regex=True)
                    st.plotly_chart(bchart(top_inflow["lbl"].tolist(),top_inflow["inflow_rate"].tolist(),
                        "Top 15 — Inflow Rate (%)",C["teal"],yl="Inflow %"),use_container_width=True)
                with mr2:
                    fig=go.Figure(go.Scatter(x=tdf_mig["inflow_rate"],y=tdf_mig["B25064_001E"],mode="markers",
                        marker=dict(size=5,color=tdf_mig.get("score",tdf_mig["inflow_rate"]) if "score" in tdf_mig.columns else tdf_mig["inflow_rate"],
                            opacity=0.7,colorscale=[[0,C["slate"]],[0.5,C["teal"]],[1,C["gold"]]],
                            colorbar=dict(title="Score") if "score" in tdf_mig.columns else None),
                        customdata=np.column_stack((tdf_mig["NAME"].values,tdf_mig["inflow_rate"].values)),
                        hovertemplate="<b>%{customdata[0]}</b><br>Inflow: %{customdata[1]:.1f}%<br>Rent: $%{y:,.0f}<extra></extra>"))
                    fig.update_layout(**BL,title=dict(text="Inflow Rate vs Median Rent",font=dict(size=16)),
                        xaxis_title="Inflow Rate (%)",yaxis_title="Median Rent ($)",yaxis_tickprefix="$")
                    st.plotly_chart(fig,use_container_width=True)

        # ── ZILLOW REAL-TIME RENT & HOME VALUES ──
        st.markdown('<div class="section-label">Housing Market Indices (Zillow ZHVI + FHFA HPI + ZORI)</div>',unsafe_allow_html=True)
        st.markdown('<div class="info-box">Monthly home values (ZHVI via FRED), quarterly house price index (FHFA via FRED), and observed rent index (ZORI via Zillow CSV). Fills the gap between annual ACS releases with real-time market signals.</div>',unsafe_allow_html=True)

        zhvi_pa, hpi_philly = fetch_zillow_from_fred()
        zori_metro = fetch_zori_csv()

        has_data = False
        zc1, zc2 = st.columns(2)
        with zc1:
            if not zhvi_pa.empty:
                has_data = True
                latest_val = zhvi_pa["value"].iloc[-1]
                yoy_val = ((zhvi_pa["value"].iloc[-1] / zhvi_pa["value"].iloc[-13]) - 1) * 100 if len(zhvi_pa) > 13 else 0
                st.metric("PA Home Value (ZHVI)", f"${latest_val:,.0f}", f"{yoy_val:+.1f}% YoY")
                fig_zhvi = go.Figure(go.Scatter(x=zhvi_pa["date"], y=zhvi_pa["value"], mode="lines",
                    line=dict(color=C["gold"], width=2)))
                fig_zhvi.update_layout(**BL, title=dict(text="Pennsylvania — ZHVI (Zillow Home Value Index)", font=dict(size=14)),
                    yaxis_tickprefix="$", height=300)
                st.plotly_chart(fig_zhvi, use_container_width=True)
            elif not hpi_philly.empty:
                has_data = True
                latest_hpi = hpi_philly["value"].iloc[-1]
                yoy_hpi = ((hpi_philly["value"].iloc[-1] / hpi_philly["value"].iloc[-5]) - 1) * 100 if len(hpi_philly) > 5 else 0
                st.metric("Philly HPI (FHFA)", f"{latest_hpi:.1f}", f"{yoy_hpi:+.1f}% YoY")
                fig_hpi = go.Figure(go.Scatter(x=hpi_philly["date"], y=hpi_philly["value"], mode="lines",
                    line=dict(color=C["gold"], width=2)))
                fig_hpi.update_layout(**BL, title=dict(text="Philadelphia MSAD — FHFA House Price Index", font=dict(size=14)),
                    height=300)
                st.plotly_chart(fig_hpi, use_container_width=True)
        with zc2:
            if not zori_metro.empty:
                has_data = True
                latest_rent = zori_metro["zori"].iloc[-1]
                yoy_rent = ((zori_metro["zori"].iloc[-1] / zori_metro["zori"].iloc[-13]) - 1) * 100 if len(zori_metro) > 13 else 0
                st.metric("MSA Median Rent (ZORI)", f"${latest_rent:,.0f}/mo", f"{yoy_rent:+.1f}% YoY")
                fig_zori = go.Figure(go.Scatter(x=zori_metro["date"], y=zori_metro["zori"], mode="lines",
                    line=dict(color=C["teal"], width=2)))
                fig_zori.update_layout(**BL, title=dict(text="Philadelphia MSA — ZORI (Zillow Observed Rent Index)", font=dict(size=14)),
                    yaxis_tickprefix="$", height=300)
                st.plotly_chart(fig_zori, use_container_width=True)
            elif not hpi_philly.empty and zhvi_pa.empty:
                pass  # Already shown HPI on left
            else:
                st.caption("ZORI unavailable — Zillow CSV endpoints may have changed.")

        if not has_data:
            st.caption("Housing index data unavailable — check FRED API key and connectivity.")

        # ── IRS SOI INCOME-STRATIFIED MIGRATION ──
        st.markdown('<div class="section-label">IRS SOI — Income-Stratified Migration</div>',unsafe_allow_html=True)
        st.markdown('<div class="info-box">County-to-county migration flows from IRS tax returns. Shows not just <i>how many</i> people move in/out, but their <b>adjusted gross income (AGI)</b>. A county gaining high-AGI households has different investment implications than one gaining low-AGI households. Source: IRS Statistics of Income, county-to-county migration files.</div>',unsafe_allow_html=True)

        inflow_soi, outflow_soi, net_soi = fetch_irs_soi_migration()

        if not net_soi.empty:
            sc1, sc2, sc3 = st.columns(3)
            latest_net = net_soi.iloc[-1]
            with sc1:
                n1_net = latest_net.get("N1_net", 0)
                st.metric("Net Households (Returns)", f"{n1_net:+,.0f}",
                    help="Positive = more households moved TO Philadelphia than left. Based on tax return address changes.")
            with sc2:
                n2_net = latest_net.get("N2_net", 0)
                st.metric("Net Individuals (Exemptions)", f"{n2_net:+,.0f}")
            with sc3:
                agi_net = latest_net.get("AGI_net", 0)
                # AGI is in thousands of dollars
                st.metric("Net AGI", f"${agi_net/1000:+,.0f}M" if abs(agi_net) >= 1000 else f"${agi_net:+,.0f}K",
                    help="Net adjusted gross income flowing in vs. out. Positive = Philadelphia gaining taxable income.")

            # Time series chart
            fig_soi = go.Figure()
            if "N1_net" in net_soi.columns:
                fig_soi.add_trace(go.Bar(x=net_soi["period"], y=net_soi["N1_net"],
                    name="Net Households", marker_color=[C["teal"] if v >= 0 else C["coral"] for v in net_soi["N1_net"]]))
            fig_soi.update_layout(**BL, title=dict(text="Philadelphia Net Migration (Households) — IRS SOI", font=dict(size=14)),
                yaxis_title="Net Households", height=350)
            fig_soi.add_hline(y=0, line_dash="dash", line_color=C["slate"], opacity=0.5)
            st.plotly_chart(fig_soi, use_container_width=True)

            # AGI net flow chart
            if "AGI_net" in net_soi.columns:
                fig_agi = go.Figure()
                fig_agi.add_trace(go.Bar(x=net_soi["period"], y=net_soi["AGI_net"] / 1000,
                    name="Net AGI ($M)", marker_color=[C["gold"] if v >= 0 else C["coral"] for v in net_soi["AGI_net"]]))
                fig_agi.update_layout(**BL, title=dict(text="Philadelphia Net AGI Flow ($M) — IRS SOI", font=dict(size=14)),
                    yaxis_title="Net AGI ($M)", yaxis_tickprefix="$", height=350)
                fig_agi.add_hline(y=0, line_dash="dash", line_color=C["slate"], opacity=0.5)
                st.plotly_chart(fig_agi, use_container_width=True)

            # Inflow vs outflow detail
            with st.expander("📊 Inflow vs Outflow Detail"):
                if not inflow_soi.empty and not outflow_soi.empty:
                    fig_io = go.Figure()
                    fig_io.add_trace(go.Scatter(x=inflow_soi["period"], y=inflow_soi.get("N1", []),
                        mode="lines+markers", name="Inflow (Households)", line=dict(color=C["teal"], width=2)))
                    fig_io.add_trace(go.Scatter(x=outflow_soi["period"], y=outflow_soi.get("N1", []),
                        mode="lines+markers", name="Outflow (Households)", line=dict(color=C["coral"], width=2)))
                    fig_io.update_layout(**BL, title=dict(text="Household Inflow vs Outflow", font=dict(size=14)),
                        yaxis_title="Households (Tax Returns)", height=350)
                    st.plotly_chart(fig_io, use_container_width=True)
        else:
            st.caption("IRS SOI data unavailable — may be a connectivity issue. Data loads from irs.gov.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: TRACT MAPS
# ══════════════════════════════════════════════════════════════════════════════
with t_maps:
    st.markdown('<div class="section-label">Interactive Census Tract Maps</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box">Choropleth maps at <b>census-tract level</b>. Hover any tract for details.</div>',unsafe_allow_html=True)
    if not CENSUS_API_KEY: st.info("👈 Enter Census API key.")
    else:
        pa_counties={k:v for k,v in COUNTY_FIPS.items() if v[0]=="42"}
        map_sel=st.multiselect("Counties to map (PA)",list(pa_counties.keys()),default=["Philadelphia"],key="mc")
        map_fips={k:v for k,v in pa_counties.items() if k in map_sel}
        map_metric=st.selectbox("Metric",[
            "Young Adults (% Age 25–34)","Median Gross Rent ($)","Renter-Occupied (%)",
            "Median Household Income ($)","Rent Burden (30%+ Income)",
            "Rent-to-Income Ratio (%)","Demand Score",
            "Inflow Rate (%)","Turnover Rate (%)","Out-of-State Inflow (%)"])
        if map_fips:
            geo=load_geojson("42")
            atract=fetch_multi_tracts(TRACT_VARS,map_fips)
            if geo and not atract.empty:
                atract=compute_tract_metrics(atract)
                atract["GEOID"]=atract["state"]+atract["county"]+atract["tract"]
                scored=compute_demand_score(atract)
                mmap={"Young Adults (% Age 25–34)":("pct2534","% 25-34","","%"),
                    "Median Gross Rent ($)":("B25064_001E","Median Rent","$",""),
                    "Renter-Occupied (%)":("renter_pct","% Renters","","%"),
                    "Median Household Income ($)":("B19013_001E","Med Income","$",""),
                    "Rent Burden (30%+ Income)":("burden_pct","% Burdened","","%"),
                    "Rent-to-Income Ratio (%)":("r2i","Rent/Income","","%"),
                    "Demand Score":("score","Score","",""),
                    "Inflow Rate (%)":("inflow_rate","Inflow %","","%"),
                    "Turnover Rate (%)":("turnover","Turnover %","","%"),
                    "Out-of-State Inflow (%)":("out_of_state_in","Out-of-State %","","%")}
                zcol,zlbl,zpre,zsuf=mmap[map_metric]
                pdf=scored if zcol=="score" else atract
                pdf=pdf.dropna(subset=[zcol])
                cfset=set(atract["state"]+atract["county"])
                fgeo={"type":"FeatureCollection","features":[
                    f for f in geo["features"] if f["properties"]["STATEFP"]+f["properties"]["COUNTYFP"] in cfset]}
                zoom=10.5 if len(map_sel)==1 else (9.5 if len(map_sel)<=3 else 8.5)
                if zcol in ["burden_pct","r2i"]: cs=[[0,C["teal"]],[0.5,C["sand"]],[1,C["coral"]]]
                else: cs=[[0,C["slate"]],[0.5,C["teal"]],[1,C["gold"]]]
                fig=go.Figure(go.Choroplethmapbox(
                    geojson=fgeo,locations=pdf["GEOID"],z=pdf[zcol],
                    featureidkey="properties.GEOID",text=pdf["NAME"],
                    colorscale=cs,marker_opacity=0.75,marker_line_width=0.5,
                    marker_line_color="rgba(255,255,255,0.15)",
                    colorbar=dict(title=dict(text=zlbl,font=dict(size=11)),
                        tickprefix=zpre,ticksuffix=zsuf,tickfont=dict(size=10),len=0.6),
                    hovertemplate=f"<b>%{{text}}</b><br>{zlbl}: {zpre}%{{z:,.1f}}{zsuf}<extra></extra>"))
                fig.update_layout(
                    mapbox=dict(style="carto-darkmatter",center=dict(lat=39.99,lon=-75.16),zoom=zoom),
                    paper_bgcolor="rgba(0,0,0,0)",font=dict(color=C["text"],family="DM Sans"),
                    margin=dict(l=0,r=0,t=50,b=0),
                    title=dict(text=f"{map_metric} — Census Tract Level",font=dict(size=16,color=C["text"])),height=620)
                st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True})
                st.markdown(f"**{len(pdf):,} tracts** across {', '.join(map_sel)}")
                s1,s2,s3,s4=st.columns(4)
                with s1: st.metric("Min",f"{zpre}{pdf[zcol].min():,.1f}{zsuf}")
                with s2: st.metric("Median",f"{zpre}{pdf[zcol].median():,.1f}{zsuf}")
                with s3: st.metric("Mean",f"{zpre}{pdf[zcol].mean():,.1f}{zsuf}")
                with s4: st.metric("Max",f"{zpre}{pdf[zcol].max():,.1f}{zsuf}")
            elif geo is None: st.warning("Could not load tract boundaries. Make sure `geopandas` is installed.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with t_demo:
    st.markdown('<div class="section-label">Demographic Trends</div>',unsafe_allow_html=True)
    d1,d2=st.columns(2)
    with d1: st.plotly_chart(lchart(fd.get("pop_philly",pd.DataFrame()),"Population — Philadelphia County (K)","Pop",C["teal"]),use_container_width=True)
    with d2: st.plotly_chart(lchart(fd.get("labor_force",pd.DataFrame()),"Labor Force — Philadelphia County","LF",C["lavender"]),use_container_width=True)
    if CENSUS_API_KEY:
        st.markdown('<div class="section-label">Age Distribution</div>',unsafe_allow_html=True)
        age_v=["B01001_003E","B01001_004E","B01001_005E","B01001_006E","B01001_007E","B01001_008E","B01001_009E","B01001_010E",
            "B01001_011E","B01001_012E","B01001_013E","B01001_014E","B01001_015E","B01001_016E","B01001_017E","B01001_018E",
            "B01001_019E","B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
            "B01001_027E","B01001_028E","B01001_029E","B01001_030E","B01001_031E","B01001_032E","B01001_033E","B01001_034E",
            "B01001_035E","B01001_036E","B01001_037E","B01001_038E","B01001_039E","B01001_040E","B01001_041E","B01001_042E",
            "B01001_043E","B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E","B01003_001E"]
        adf=fetch_acs_counties(age_v,sel_fips)
        if not adf.empty:
            for c_ in age_v: adf[c_]=pd.to_numeric(adf[c_],errors="coerce")
            adf["u18"]=adf[["B01001_003E","B01001_004E","B01001_005E","B01001_006E","B01001_027E","B01001_028E","B01001_029E","B01001_030E"]].sum(1)
            adf["a1824"]=adf[["B01001_007E","B01001_008E","B01001_009E","B01001_010E","B01001_031E","B01001_032E","B01001_033E","B01001_034E"]].sum(1)
            adf["a2534"]=adf[["B01001_011E","B01001_012E","B01001_035E","B01001_036E"]].sum(1)
            adf["a3544"]=adf[["B01001_013E","B01001_014E","B01001_037E","B01001_038E"]].sum(1)
            adf["a4554"]=adf[["B01001_015E","B01001_016E","B01001_039E","B01001_040E"]].sum(1)
            adf["a5564"]=adf[["B01001_017E","B01001_018E","B01001_019E","B01001_041E","B01001_042E","B01001_043E"]].sum(1)
            adf["a65"]=adf[["B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E","B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E"]].sum(1)
            tot=adf["B01003_001E"]
            grps=["u18","a1824","a2534","a3544","a4554","a5564","a65"]
            lbls=["Under 18","18–24","25–34","35–44","45–54","55–64","65+"]
            clrs=[C["steel"],C["sky"],C["gold"],C["teal"],C["lavender"],C["coral"],C["sand"]]
            fig=go.Figure()
            for g,l,cl in zip(grps,lbls,clrs):
                fig.add_trace(go.Bar(x=adf["county_label"],y=(adf[g]/tot*100),name=l,marker_color=cl,
                    hovertemplate=f"<b>%{{x}}</b><br>{l}: %{{y:.1f}}%<extra></extra>"))
            fig.update_layout(**BL,barmode="stack",title=dict(text="Age Distribution (%)",font=dict(size=16)),
                yaxis_ticksuffix="%",legend=dict(orientation="h",y=-0.15,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
            st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="section-label">Education (25+)</div>',unsafe_allow_html=True)
        ev=["B15003_001E","B15003_022E","B15003_023E","B15003_024E","B15003_025E","B15003_017E","B15003_018E"]
        edf=fetch_acs_counties(ev,sel_fips)
        if not edf.empty:
            for c_ in ev: edf[c_]=pd.to_numeric(edf[c_],errors="coerce")
            edf["bach"]=(edf[["B15003_022E","B15003_023E","B15003_024E","B15003_025E"]].sum(1)/edf["B15003_001E"]*100)
            edf["hs"]=((edf["B15003_017E"]+edf["B15003_018E"])/edf["B15003_001E"]*100)
            e1,e2=st.columns(2)
            with e1:
                s=edf.sort_values("bach",ascending=True)
                st.plotly_chart(bchart(s["county_label"].tolist(),s["bach"].tolist(),"Bachelor's+ (%)",C["teal"],True,"% 25+"),use_container_width=True)
            with e2:
                s=edf.sort_values("hs",ascending=True)
                st.plotly_chart(bchart(s["county_label"].tolist(),s["hs"].tolist(),"HS/GED Only (%)",C["sand"],True,"% 25+"),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: REGIONAL (with tooltips, no background_gradient)
# ══════════════════════════════════════════════════════════════════════════════
with t_reg:
    st.markdown('<div class="section-label">Regional Comparison</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box">Full cross-county scorecard with <b>Blue-Collar Rental Opportunity Score</b>.</div>',unsafe_allow_html=True)
    if CENSUS_API_KEY:
        comp=fetch_acs_counties(TRACT_VARS,COUNTY_FIPS)
        if not comp.empty:
            comp=compute_tract_metrics(comp)
            st.markdown('<div class="section-label">Scorecard</div>',unsafe_allow_html=True)
            disp=comp[["county_label","B01003_001E","B19013_001E","B25064_001E","pct2534","renter_pct","r2i","unemp"]].rename(columns={
                "county_label":"County","B01003_001E":"Population","B19013_001E":"Med Income",
                "B25064_001E":"Med Rent","pct2534":"% 25-34","renter_pct":"% Renters","r2i":"Rent/Inc %","unemp":"Unemp %"
            }).sort_values("Population",ascending=False)
            st.dataframe(disp.style.format({"Population":"{:,.0f}","Med Income":"${:,.0f}","Med Rent":"${:,.0f}",
                "% 25-34":"{:.1f}%","% Renters":"{:.1f}%","Rent/Inc %":"{:.1f}%","Unemp %":"{:.1f}%"
            }),use_container_width=True,hide_index=True)
            st.markdown('<div class="section-label">Blue-Collar Rental Opportunity Score</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">{tip("opp_score")}</div>',unsafe_allow_html=True)
            factors={"pct2534":True,"renter_pct":True,"r2i":False,"unemp":False}
            for col,higher in factors.items():
                mn,mx=comp[col].min(),comp[col].max()
                n=((comp[col]-mn)/(mx-mn)) if mx>mn else 0.5
                comp[f"{col}_s"]=n if higher else (1-n)
            comp["opp"]=(comp["pct2534_s"]+comp["renter_pct_s"]+comp["r2i_s"]+comp["unemp_s"])*25
            cs=comp.sort_values("opp",ascending=True)
            fig=go.Figure(go.Bar(y=cs["county_label"],x=cs["opp"],orientation="h",
                marker=dict(color=cs["opp"],colorscale=[[0,C["slate"]],[0.5,C["teal"]],[1,C["gold"]]]),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>"))
            fig.update_layout(**BL,title=dict(text="Opportunity Score (0–100)",font=dict(size=16)),xaxis_title="Score")
            st.plotly_chart(fig,use_container_width=True)
            with st.expander("📋 Score Breakdown"):
                sb=comp[["county_label","pct2534_s","renter_pct_s","r2i_s","unemp_s","opp"]].rename(columns={
                    "county_label":"County","pct2534_s":"Young Adults","renter_pct_s":"Renters",
                    "r2i_s":"Affordability","unemp_s":"Employment","opp":"Total"}).sort_values("Total",ascending=False)
                for c_ in ["Young Adults","Renters","Affordability","Employment"]: sb[c_]=(sb[c_]*25).round(1)
                sb["Total"]=sb["Total"].round(1)
                st.dataframe(sb,use_container_width=True,hide_index=True)
    else: st.info("Enter Census API key in sidebar.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: FORECAST (NEW)
# ══════════════════════════════════════════════════════════════════════════════
with t_fc:
    st.markdown('<div class="section-label">Forecasting Engine</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">{tip("forecast")}</div>',unsafe_allow_html=True)

    if not FRED_API_KEY:
        st.info("👈 Enter FRED API key to enable forecasting.")
    else:
        # Need longer history for forecasting
        fd_full = {}
        with st.spinner("Loading full history for forecasting…"):
            for k, sid in FRED_SERIES.items():
                fd_full[k] = fetch_fred(sid, "2001-01-01")

        # Fetch county-level migration time series for forecast features
        migration_ts = pd.DataFrame()
        if CENSUS_API_KEY:
            with st.spinner("Loading migration history…"):
                migration_ts = fetch_migration_timeseries("42", "101", list(range(2010, 2024)))

        # Fetch ZHVI from FRED for forecast features
        zhvi_fred, _ = fetch_zillow_from_fred()
        zori_fc = fetch_zori_csv()
        # Use ZHVI if available, ZORI as backup
        zillow_for_fc = zhvi_fred if not zhvi_fred.empty else None
        annual, targets = build_annual_dataset(fd_full, migration_ts,
            zillow_zori_metro=zori_fc if not zori_fc.empty else None,
            zhvi_fred=zhvi_fred if not zhvi_fred.empty else None)

        if annual.empty:
            st.warning("Not enough data to build forecasts.")
        else:
            mig_status = f" · Migration data: {len(migration_ts)} years ({int(migration_ts['year'].min())}–{int(migration_ts['year'].max())})" if not migration_ts.empty else " · No migration data"
            st.markdown(f"**{len(annual)} years** of annual data assembled ({int(annual.index.min())}–{int(annual.index.max())}){mig_status}")

            for target_label, target_key in targets.items():
                if target_key not in annual.columns:
                    continue

                st.markdown(f'<div class="section-label">{target_label}</div>', unsafe_allow_html=True)
                result = run_forecast(annual, target_key)

                if result is None:
                    st.markdown("_Insufficient data for this forecast._")
                    continue

                # Format helper
                is_dollar = "$" in target_label
                is_pct = "%" in target_label
                def fv(v):
                    if is_dollar and abs(v) >= 1e6: return f"${v/1e3:,.0f}B" if "GDP" in target_label else f"${v/1e6:,.0f}M"
                    if is_dollar and abs(v) >= 1e3: return f"${v/1e3:,.0f}K"
                    if is_dollar: return f"${v:,.0f}"
                    if is_pct: return f"{v:.1f}%"
                    return f"{v:,.0f}"

                hr = result["horizon_results"]

                # Metrics: Actual + 1yr, 2yr, 5yr forecasts
                fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                with fc1:
                    st.metric(f"{result['last_actual_year']} (Actual)", fv(result["last_actual_val"]))
                for col, h, label in [(fc2, 1, "1-Year"), (fc3, 2, "2-Year"), (fc4, 5, "5-Year")]:
                    with col:
                        if h in hr:
                            chg_pct = ((hr[h]["value"] - result["last_actual_val"]) / abs(result["last_actual_val"])) * 100 if result["last_actual_val"] != 0 else 0
                            dc = "normal" if not is_pct else ("inverse" if "Unemployment" in target_label else "normal")
                            st.metric(f"{hr[h]['year']} ({label})", fv(hr[h]["value"]),
                                f"{chg_pct:+.1f}%", delta_color=dc,
                                help=f"95% CI: {fv(hr[h]['ci_low'])} – {fv(hr[h]['ci_high'])}")
                with fc5:
                    mig_tag = " 🌍" if result.get("used_migration") else ""
                    model_tag = " 📈" if result.get("model_type") == "trend-adjusted" else ""
                    st.metric(f"Backtest MAPE{mig_tag}{model_tag}", f"{result['mape']:.1f}%",
                        help=f"Mean Absolute Percentage Error on walk-forward backtest. Lower = more accurate."
                             f"{' Migration features included.' if result.get('used_migration') else ' Migration features excluded.'}"
                             f" Model: {result.get('model_type', 'standard')}."
                             f"{' 📈=trend-adjusted, 🌍=migration' if model_tag else ''}")
                    # Show model comparison if available
                    mc = result.get("model_comparison")
                    if mc:
                        winner = mc["selected"]
                        loser_mape = mc["trend_mape"] if winner == "standard" else mc["standard_mape"]
                        st.caption(f"✓ {winner.title()} selected (wMAPE {result['weighted_mape']:.1f}% vs {loser_mape:.1f}%)")

                # Charts
                bt = result["backtest"]
                fc_df = result["forecasts"]

                # Build full actual history from annual dataset
                full_actual = annual[[target_key]].dropna().copy()
                full_actual = full_actual.reset_index()
                full_actual.columns = ["year", "actual"]

                ch1, ch2 = st.columns(2)
                with ch1:
                    fig = go.Figure()
                    # Full actual history line
                    fig.add_trace(go.Scatter(x=full_actual["year"], y=full_actual["actual"],
                        mode="lines+markers", name="Actual",
                        line=dict(color=C["teal"], width=2.5), marker=dict(size=5),
                        hovertemplate="<b>%{x}</b><br>Actual: " + ("$" if is_dollar else "") + "%{y:,.1f}" + ("%" if is_pct else "") + "<extra></extra>"))
                    # Backtest predictions overlay
                    if not bt.empty:
                        fig.add_trace(go.Scatter(x=bt["year"], y=bt["predicted"], mode="markers",
                            name="Backtest Pred", marker=dict(color=C["gold"], size=7, symbol="diamond", opacity=0.8),
                            hovertemplate="<b>%{x}</b><br>Backtest: " + ("$" if is_dollar else "") + "%{y:,.1f}" + ("%" if is_pct else "") + "<extra></extra>"))
                    # Bridge: connect last actual to first forecast
                    last_yr = result["last_actual_year"]
                    last_val = result["last_actual_val"]
                    bridge_x = [last_yr, fc_df["year"].iloc[0]]
                    bridge_y = [last_val, fc_df["value"].iloc[0]]
                    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines",
                        line=dict(color=C["coral"], width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"))
                    # Forecast CI band
                    ci_x = [last_yr] + fc_df["year"].tolist()
                    ci_hi = [last_val] + fc_df["ci_high"].tolist()
                    ci_lo = [last_val] + fc_df["ci_low"].tolist()
                    fig.add_trace(go.Scatter(x=ci_x, y=ci_hi,
                        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=ci_x, y=ci_lo,
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(231,111,81,0.10)", showlegend=False, hoverinfo="skip"))
                    # Forecast line
                    fig.add_trace(go.Scatter(x=fc_df["year"], y=fc_df["value"], mode="lines+markers",
                        name="Forecast", line=dict(color=C["coral"], width=2.5),
                        marker=dict(size=8, symbol="star"),
                        hovertemplate="<b>%{x}</b><br>Forecast: " + ("$" if is_dollar else "") + "%{y:,.1f}" + ("%" if is_pct else "") + "<extra></extra>"))
                    # Mark specific horizons
                    for h in [1, 2, 5]:
                        if h in hr:
                            fig.add_annotation(x=hr[h]["year"], y=hr[h]["value"],
                                text=f"{h}yr", showarrow=True, arrowhead=2, arrowcolor=C["gold"],
                                font=dict(size=10, color=C["gold"]), ax=0, ay=-25)
                    fmt = {}
                    if is_dollar: fmt["yaxis_tickprefix"] = "$"
                    if is_pct: fmt["yaxis_ticksuffix"] = "%"
                    if "GDP" in target_label:
                        # GDP values are in $M, format axis as billions
                        fig.update_yaxes(tickformat=",", tickprefix="$", ticksuffix="")
                        # Custom tick text showing B
                        fig.update_layout(yaxis=dict(
                            tickvals=[v for v in range(200000, 800001, 100000)],
                            ticktext=[f"${v//1000}B" for v in range(200000, 800001, 100000)],
                        ))
                        fmt.pop("yaxis_tickprefix", None)
                    fig.update_layout(**BL, title=dict(text=f"{target_label} — History + Forecast", font=dict(size=14)), **fmt)
                    st.plotly_chart(fig, use_container_width=True)

                with ch2:
                    imp = result["importance"].head(6)
                    clean_names = [n.replace("_lag1","").replace("_"," ").title() for n in imp.index]
                    fig = go.Figure(go.Bar(y=clean_names[::-1], x=imp.values[::-1], orientation="h",
                        marker_color=C["teal"],
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"))
                    fig.update_layout(**BL, title=dict(text="Feature Importance (Abs Coefficients)", font=dict(size=14)))
                    st.plotly_chart(fig, use_container_width=True)

                # Multi-horizon backtest accuracy
                ha = result.get("horizon_accuracy", {})
                if ha:
                    bt_cols = st.columns(len(ha))
                    for i, (h, acc) in enumerate(sorted(ha.items())):
                        with bt_cols[i]:
                            label = f"{h}-Year" if h > 1 else "1-Year"
                            color = "normal" if acc["mape"] < 15 else ("off" if acc["mape"] < 30 else "inverse")
                            st.metric(f"{label} MAPE", f"{acc['mape']:.1f}%",
                                help=f"Backtest MAPE for {h}-year ahead predictions. RMSE: {acc['rmse']:,.1f}. Based on {acc['n']} test windows.")

                # Multi-horizon backtest chart
                bt_full = result.get("backtest_full", pd.DataFrame())
                if not bt_full.empty and len(ha) > 1:
                    with st.expander("📊 Multi-Horizon Backtest Detail"):
                        bch1, bch2 = st.columns(2)
                        with bch1:
                            # MAPE by horizon bar chart
                            h_labels = [f"{h}-Year" for h in sorted(ha.keys())]
                            h_mapes = [ha[h]["mape"] for h in sorted(ha.keys())]
                            h_colors = [C["teal"] if m < 10 else (C["gold"] if m < 25 else C["coral"]) for m in h_mapes]
                            fig = go.Figure(go.Bar(x=h_labels, y=h_mapes, marker_color=h_colors,
                                hovertemplate="<b>%{x}</b><br>MAPE: %{y:.1f}%<extra></extra>"))
                            fig.update_layout(**BL, title=dict(text="Backtest MAPE by Horizon", font=dict(size=14)),
                                yaxis_ticksuffix="%", yaxis_title="MAPE (%)")
                            st.plotly_chart(fig, use_container_width=True)

                        with bch2:
                            # Actual vs predicted scatter for each horizon
                            fig = go.Figure()
                            h_colors_map = {1: C["teal"], 2: C["gold"], 5: C["coral"]}
                            for h in sorted(ha.keys()):
                                bh = bt_full[bt_full["horizon"] == h]
                                fig.add_trace(go.Scatter(x=bh["actual"], y=bh["predicted"],
                                    mode="markers", name=f"{h}-Year",
                                    marker=dict(color=h_colors_map.get(h, C["steel"]), size=8, opacity=0.7),
                                    hovertemplate=f"<b>{h}-yr</b><br>Actual: %{{x:,.1f}}<br>Predicted: %{{y:,.1f}}<extra></extra>"))
                            # Perfect prediction line
                            all_vals = pd.concat([bt_full["actual"], bt_full["predicted"]])
                            mn, mx = all_vals.min(), all_vals.max()
                            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                line=dict(color=C["muted"], width=1, dash="dot"), name="Perfect", showlegend=True))
                            fmt = {}
                            if is_dollar: fmt.update({"xaxis_tickprefix": "$", "yaxis_tickprefix": "$"})
                            if is_pct: fmt.update({"xaxis_ticksuffix": "%", "yaxis_ticksuffix": "%"})
                            fig.update_layout(**BL, title=dict(text="Predicted vs Actual (All Horizons)", font=dict(size=14)),
                                xaxis_title="Actual", yaxis_title="Predicted", **fmt)
                            st.plotly_chart(fig, use_container_width=True)

            # Methodology note
            with st.expander("📖 Forecast Methodology"):
                st.markdown("""
**Model:** Ridge Regression with L2 regularization (α=1.0)

**Features:** 1-year lagged values of all available FRED indicators (unemployment, permits, GDP, CPI shelter, CPI all items, construction employment, homeownership, median income, fed funds rate, 10-year Treasury, 30-year mortgage, yield curve spread) plus Census ACS county-level migration features (inflow rate, turnover rate, out-of-state inflow rate)

**Validation:** Expanding-window walk-forward backtest — the model is retrained at each step using only data available up to that point, then predicts the next year. This prevents data leakage and simulates real-world forecasting conditions.

**Confidence Interval:** 95% CI derived from backtest residuals, widening with √horizon (uncertainty compounds over time)

**Multi-Year Forecast:** Iterative approach — each year's prediction feeds back as input for the next year. The 1-year forecast is most reliable; 5-year forecasts carry substantially more uncertainty (reflected in wider CI bands).

**MAPE:** Mean Absolute Percentage Error across all backtest windows. Under 5% is excellent; under 10% is good.

**Limitations:** Annual frequency limits sample size (~20 observations). Model assumes linear relationships and stable regime. Structural breaks (pandemics, policy shifts) may not be captured. GDP data lags ~1 year from BEA. Panel regression (9-county with fixed effects) and Bayesian structural time series (BSTS) were tested as alternative models but did not outperform on backtesting: the cross-county panel introduced too much heterogeneity, and BSTS without exogenous features could not beat Ridge with 12+ macro predictors. These may become viable as more years of data accumulate.
""")

        # ── RATE SENSITIVITY ANALYSIS ──
        st.markdown('<div class="section-label">📈 Interest Rate Sensitivity</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{tip("rate_sensitivity")}</div>', unsafe_allow_html=True)

        # Re-run forecasts and perturb rate features
        rate_features = ["fed_funds_lag1", "treasury_10y_lag1", "mortgage_30y_lag1"]
        shock_bps = st.select_slider("Rate shock (basis points)", options=[-200, -100, -50, 0, 50, 100, 200], value=0, key="rate_shock")

        if shock_bps != 0:
            st.markdown(f"**Scenario:** All interest rates {'↑' if shock_bps > 0 else '↓'} {abs(shock_bps)}bp from current levels")

        sens_rows = []
        for target_label, target_key in targets.items():
            if target_key not in annual.columns: continue
            result = run_forecast(annual, target_key)
            if result is None: continue
            hr = result.get("horizon_results", {})
            if 1 not in hr: continue
            base_val = hr[1]["value"]

            if shock_bps != 0 and result.get("importance") is not None:
                # Compute shocked forecast by perturbing rate features
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                clean = annual.ffill().bfill().dropna(axis=1)
                if target_key not in clean.columns: continue
                feature_cols = list(clean.columns)
                lagged = clean.shift(1)
                lagged.columns = [f"{c}_lag1" for c in feature_cols]
                combo = pd.concat([clean[[target_key]], lagged], axis=1).iloc[1:]
                combo = combo.replace([np.inf, -np.inf], np.nan).dropna()
                if len(combo) < 6: continue
                X = combo.drop(columns=[target_key])
                y = combo[target_key]
                feat_names = X.columns.tolist()
                sc = StandardScaler(); X_s = sc.fit_transform(X)
                m = Ridge(alpha=1.0); m.fit(X_s, y)
                # Build prediction row with shocked rates
                last_vals = clean[feature_cols].iloc[-1:].copy()
                last_vals.columns = [f"{c}_lag1" for c in feature_cols]
                last_vals = last_vals[feat_names].fillna(0)
                shocked = last_vals.copy()
                for rf in rate_features:
                    if rf in shocked.columns:
                        shocked[rf] = shocked[rf] + (shock_bps / 100.0)
                shocked_val = m.predict(sc.transform(shocked))[0]
                delta = shocked_val - base_val
                delta_pct = (delta / abs(base_val)) * 100 if base_val != 0 else 0
                sens_rows.append({
                    "Metric": target_label, "Base Forecast": base_val,
                    "Shocked Forecast": shocked_val, "Δ": delta, "Δ%": delta_pct,
                })

        if sens_rows and shock_bps != 0:
            sens_df = pd.DataFrame(sens_rows)
            sc1, sc2 = st.columns(2)
            with sc1:
                # Table
                is_d = lambda l: "$" in l
                is_p = lambda l: "%" in l
                def fvs(v, lbl):
                    if "$" in lbl and abs(v) >= 1e3: return f"${v/1e3:,.0f}K"
                    if "$" in lbl: return f"${v:,.0f}"
                    if "%" in lbl: return f"{v:.2f}%"
                    return f"{v:,.0f}"
                display_rows = []
                for _, r in sens_df.iterrows():
                    display_rows.append({
                        "Metric": r["Metric"],
                        "Base": fvs(r["Base Forecast"], r["Metric"]),
                        f"{'↑' if shock_bps > 0 else '↓'}{abs(shock_bps)}bp": fvs(r["Shocked Forecast"], r["Metric"]),
                        "Impact": f"{r['Δ%']:+.2f}%",
                    })
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
            with sc2:
                # Impact bar chart
                fig = go.Figure(go.Bar(
                    y=sens_df["Metric"].tolist()[::-1],
                    x=sens_df["Δ%"].tolist()[::-1],
                    orientation="h",
                    marker_color=[C["teal"] if d > 0 else C["coral"] for d in sens_df["Δ%"].tolist()[::-1]],
                    hovertemplate="<b>%{y}</b><br>Impact: %{x:+.2f}%<extra></extra>"))
                fig.add_vline(x=0, line=dict(color=C["muted"], width=1))
                fig.update_layout(**BL, title=dict(text=f"Rate {'Hike' if shock_bps > 0 else 'Cut'} Impact on 1-Year Forecasts", font=dict(size=14)),
                    xaxis_ticksuffix="%", xaxis_title="Change from Base (%)")
                st.plotly_chart(fig, use_container_width=True)
        elif shock_bps == 0:
            st.markdown("_Move the slider above to simulate a rate shock scenario._")

        # ── PREDICTION TRACKER ──
        st.markdown('<div class="section-label">📋 Prediction Tracker</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">**Prediction Log** — Captures today\'s forecasts for future validation. As actuals become available, accuracy can be measured. Export this table periodically to build a track record.</div>', unsafe_allow_html=True)

        tracker_rows = []
        for target_label, target_key in targets.items():
            if target_key not in annual.columns: continue
            result = run_forecast(annual, target_key)
            if result is None: continue
            hr = result.get("horizon_results", {})
            for h in [1, 2, 5]:
                if h not in hr: continue
                tracker_rows.append({
                    "Metric": target_label,
                    "Horizon": f"{h}-Year",
                    "Target Year": hr[h]["year"],
                    "Forecast": hr[h]["value"],
                    "95% CI Low": hr[h]["ci_low"],
                    "95% CI High": hr[h]["ci_high"],
                    "Base Year Actual": result["last_actual_val"],
                    "Backtest MAPE": result["mape"],
                    "Generated": datetime.now().strftime("%Y-%m-%d"),
                })
        if tracker_rows:
            trk = pd.DataFrame(tracker_rows)
            # Format for display
            def fmt_trk(row):
                lbl = row["Metric"]
                def f(v):
                    if "$" in lbl and abs(v) >= 1e3: return f"${v/1e3:,.1f}K"
                    if "$" in lbl: return f"${v:,.0f}"
                    if "%" in lbl: return f"{v:.2f}%"
                    return f"{v:,.0f}"
                return pd.Series({
                    "Metric": lbl, "Horizon": row["Horizon"], "Target Year": int(row["Target Year"]),
                    "Forecast": f(row["Forecast"]),
                    "95% CI": f"{f(row['95% CI Low'])} – {f(row['95% CI High'])}",
                    "Last Actual": f(row["Base Year Actual"]),
                    "MAPE": f"{row['Backtest MAPE']:.1f}%",
                    "Date": row["Generated"],
                })
            trk_display = trk.apply(fmt_trk, axis=1)
            st.dataframe(trk_display, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = trk.to_csv(index=False)
            st.download_button("📥 Export Predictions (CSV)", csv_data,
                f"lapstone_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", key="pred_dl")

        # ── OPPORTUNITY AREA PREDICTION ──
        st.markdown('<div class="section-label">🎯 Opportunity Area Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{tip("opp_pred")}</div>', unsafe_allow_html=True)

        if CENSUS_API_KEY:
            # Collect macro forecast results for all horizons
            macro_by_horizon = {}
            for h in [1, 2, 5]:
                macro_h = {}
                for tl, tk in targets.items():
                    if tk in annual.columns:
                        r = run_forecast(annual, tk)
                        if r and r.get("horizon_results") and h in r["horizon_results"]:
                            macro_h[tl] = {
                                "forecast_val": r["horizon_results"][h]["value"],
                                "last_actual_val": r["last_actual_val"],
                            }
                macro_by_horizon[h] = macro_h

            opp_counties = st.multiselect("Counties to score", [k for k,v in COUNTY_FIPS.items() if v[0]=="42"],
                default=["Philadelphia"], key="opp_mc")
            opp_fips = {k: v for k, v in COUNTY_FIPS.items() if k in opp_counties and v[0] == "42"}

            if opp_fips:
                with st.spinner("Loading tract data for opportunity scoring…"):
                    tracts_curr = fetch_multi_tracts(TRACT_VARS, opp_fips, year=2023)
                    tracts_prev = fetch_multi_tracts(TRACT_VARS, opp_fips, year=2022)

                if not tracts_curr.empty:
                    tracts_curr = compute_tract_metrics(tracts_curr)
                    tracts_curr["GEOID"] = tracts_curr["state"] + tracts_curr["county"] + tracts_curr["tract"]

                    # Compute opportunity scores for all horizons
                    opp_horizons = {}
                    for h in [1, 2, 5]:
                        opp_h = compute_tract_opportunity(tracts_curr, tracts_prev, macro_by_horizon.get(h, {}), horizon=h)
                        if not opp_h.empty:
                            opp_h = opp_h.rename(columns={"opp_pred": f"opp_{h}yr"})
                            opp_horizons[h] = opp_h

                    if opp_horizons:
                        # Merge all horizons into one dataframe
                        opp = opp_horizons[1].copy() if 1 in opp_horizons else list(opp_horizons.values())[0].copy()
                        for h in [2, 5]:
                            if h in opp_horizons and f"opp_{h}yr" in opp_horizons[h].columns:
                                opp[f"opp_{h}yr"] = opp_horizons[h][f"opp_{h}yr"].values
                        # Default display column
                        if "opp_1yr" not in opp.columns and "opp_2yr" in opp.columns:
                            opp["opp_1yr"] = opp["opp_2yr"]

                        # Horizon selector
                        opp_horizon_sel = st.radio("Forecast horizon", ["1-Year", "2-Year", "5-Year"],
                            horizontal=True, key="opp_hz")
                        hz_map = {"1-Year": 1, "2-Year": 2, "5-Year": 5}
                        sel_h = hz_map[opp_horizon_sel]
                        score_col = f"opp_{sel_h}yr"
                        if score_col not in opp.columns:
                            score_col = "opp_1yr"

                        top_n = st.slider("Top tracts to display", 10, 50, 25, key="opp_n")
                        top = opp.nlargest(top_n, score_col).copy()
                        top["lbl"] = top["NAME"].str.replace(r"Census Tract (\d+\.?\d*),.*", r"Tract \1", regex=True)

                        # Summary metrics
                        om1, om2, om3, om4, om5 = st.columns(5)
                        with om1: st.metric(f"Top Score ({opp_horizon_sel})", f"{top[score_col].max():.1f}/100")
                        with om2: st.metric("Median (Top)", f"{top[score_col].median():.1f}/100")
                        with om3: st.metric(f"Tracts Scored", f"{len(opp):,}")
                        with om4:
                            macro_vals = macro_by_horizon.get(sel_h, {})
                            macro_dir = "📈 Favorable" if macro_vals and np.mean([r.get("forecast_val",0) > r.get("last_actual_val",0) for r in macro_vals.values() if r]) > 0.5 else "📉 Cautious"
                            st.metric("Macro Outlook", macro_dir)
                        with om5:
                            # Show how rankings shift across horizons
                            if "opp_1yr" in opp.columns and "opp_5yr" in opp.columns:
                                top1 = set(opp.nlargest(10, "opp_1yr")["GEOID"])
                                top5 = set(opp.nlargest(10, "opp_5yr")["GEOID"])
                                overlap = len(top1 & top5)
                                st.metric("1yr↔5yr Overlap", f"{overlap}/10",
                                    help="How many of the top 10 tracts at the 1-year horizon remain in the top 10 at 5 years. Low overlap = rankings shift significantly over time.")

                        # Horizon comparison for top tracts
                        if all(f"opp_{h}yr" in opp.columns for h in [1, 2, 5]):
                            with st.expander("📊 Horizon Comparison — How Scores Shift Over Time"):
                                top_compare = opp.nlargest(15, score_col).copy()
                                top_compare["lbl"] = top_compare["NAME"].str.replace(r"Census Tract (\d+\.?\d*),.*", r"Tract \1", regex=True)
                                fig = go.Figure()
                                for h, clr in [(1, C["teal"]), (2, C["gold"]), (5, C["coral"])]:
                                    fig.add_trace(go.Bar(
                                        y=top_compare["lbl"].tolist()[::-1],
                                        x=top_compare[f"opp_{h}yr"].tolist()[::-1],
                                        name=f"{h}-Year", orientation="h",
                                        marker_color=clr, opacity=0.8))
                                fig.update_layout(**BL, barmode="group",
                                    title=dict(text="Top 15 Tracts — Score by Horizon", font=dict(size=14)),
                                    xaxis_title="Opportunity Score", height=max(400, 15 * 28),
                                    legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
                                st.plotly_chart(fig, use_container_width=True)

                        # Map
                        geo = load_geojson("42")
                        if geo:
                            cfset = set(tracts_curr["state"] + tracts_curr["county"])
                            fgeo = {"type": "FeatureCollection", "features": [
                                f for f in geo["features"] if f["properties"]["STATEFP"]+f["properties"]["COUNTYFP"] in cfset]}
                            zoom = 10.5 if len(opp_counties) == 1 else (9.5 if len(opp_counties) <= 3 else 8.5)
                            fig = go.Figure(go.Choroplethmapbox(
                                geojson=fgeo, locations=opp["GEOID"], z=opp[score_col],
                                featureidkey="properties.GEOID", text=opp["NAME"],
                                colorscale=[[0, C["slate"]], [0.4, C["teal"]], [0.7, C["gold"]], [1, C["coral"]]],
                                marker_opacity=0.8, marker_line_width=0.5,
                                marker_line_color="rgba(255,255,255,0.15)",
                                colorbar=dict(title=dict(text="Opp Score", font=dict(size=11)),
                                    tickfont=dict(size=10), len=0.6),
                                hovertemplate="<b>%{text}</b><br>Opportunity: %{z:.1f}/100<extra></extra>"))
                            fig.update_layout(
                                mapbox=dict(style="carto-darkmatter", center=dict(lat=39.99, lon=-75.16), zoom=zoom),
                                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], family="DM Sans"),
                                margin=dict(l=0, r=0, t=50, b=0),
                                title=dict(text=f"Opportunity Area Prediction — {opp_horizon_sel}", font=dict(size=16, color=C["text"])), height=620)
                            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

                        # Bar chart: top tracts
                        fig = go.Figure(go.Bar(
                            y=top["lbl"].tolist()[::-1], x=top[score_col].tolist()[::-1],
                            orientation="h",
                            marker=dict(color=top[score_col].tolist()[::-1],
                                colorscale=[[0, C["teal"]], [1, C["gold"]]]),
                            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>"))
                        fig.update_layout(**BL, title=dict(text=f"Top {top_n} Opportunity Tracts ({opp_horizon_sel})", font=dict(size=16)),
                            xaxis_title="Opportunity Score", height=max(400, top_n * 22))
                        st.plotly_chart(fig, use_container_width=True)

                        # Detailed table
                        with st.expander(f"📋 Top {top_n} Tracts — Full Breakdown ({opp_horizon_sel})"):
                            show_cols = ["lbl", score_col, "score", "B25064_001E", "B19013_001E",
                                         "r2i", "pct2534", "renter_pct"]
                            rename = {"lbl": "Tract", score_col: f"Opp Score ({opp_horizon_sel})", "score": "Demand Score",
                                      "B25064_001E": "Rent", "B19013_001E": "Income",
                                      "r2i": "Rent/Inc %", "pct2534": "% 25-34", "renter_pct": "% Renters"}
                            # Add all horizon scores to table
                            for h in [1, 2, 5]:
                                hcol = f"opp_{h}yr"
                                if hcol in top.columns and hcol != score_col:
                                    show_cols.append(hcol)
                                    rename[hcol] = f"{h}yr Score"
                            extra_cols = {}
                            if "rent_growth" in top.columns:
                                show_cols.extend(["rent_growth", "income_growth", "youth_growth"])
                                extra_cols = {"rent_growth": "Rent Δ%", "income_growth": "Income Δ%", "youth_growth": "Youth Δpp"}
                            if "inflow_rate" in top.columns:
                                show_cols.extend(["inflow_rate", "turnover"])
                                extra_cols.update({"inflow_rate": "Inflow %", "turnover": "Turnover %"})
                            rename.update(extra_cols)
                            tbl = top[[c for c in show_cols if c in top.columns]].rename(columns=rename)
                            fmt = {f"Opp Score ({opp_horizon_sel})": "{:.1f}", "Demand Score": "{:.0f}", "Rent": "${:,.0f}",
                                   "Income": "${:,.0f}", "Rent/Inc %": "{:.1f}%", "% 25-34": "{:.1f}%",
                                   "% Renters": "{:.1f}%"}
                            for h in [1, 2, 5]:
                                if f"{h}yr Score" in tbl.columns:
                                    fmt[f"{h}yr Score"] = "{:.1f}"
                            if "Rent Δ%" in tbl.columns:
                                fmt.update({"Rent Δ%": "{:+.1f}%", "Income Δ%": "{:+.1f}%", "Youth Δpp": "{:+.1f}pp"})
                            if "Inflow %" in tbl.columns:
                                fmt.update({"Inflow %": "{:.1f}%", "Turnover %": "{:.1f}%"})
                            st.dataframe(tbl.style.format({k: v for k, v in fmt.items() if k in tbl.columns}),
                                use_container_width=True, hide_index=True)

                        # Score decomposition
                        with st.expander("📖 Score Methodology"):
                            st.markdown("""
**Opportunity Area Prediction Score** (0–100) combines four signals:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **Fundamentals** | 40% | Current demand score (young adults, renters, affordability) |
| **Momentum** | 30% | Year-over-year improvement in rent, income, and young adult share (ACS 2022→2023) |
| **Affordability Headroom** | 20% | Low rent-to-income ratio = room for rent increases without burdening tenants |
| **Macro Alignment** | 10% | Whether FRED macro forecasts (GDP↑, unemployment↓, permits↑) are favorable |
| **Migration Inflow** | 8%* | Higher inflow rate = more people moving in from outside the county (demand signal) |

*When migration data is available, weights are redistributed: Fundamentals -2%, Momentum -3%, Headroom -1%, Macro -2%. When unavailable, the original 4-factor weighting is used.

**Interpretation:**
- **75+**: High-opportunity — strong fundamentals AND improving trajectory
- **60–75**: Moderate — solid base, watch for momentum shifts
- **Below 60**: Lower priority — may have affordability issues or declining demographics

**Data Sources:** Census ACS 5-Year (2022, 2023), FRED macro indicators, Ridge regression forecasts
""")
        else:
            st.info("👈 Enter Census API key to enable opportunity prediction.")

# ── FOOTER ──
st.markdown("---")
st.markdown(f'<div style="text-align:center;color:#5A6270;font-size:.8rem;padding:1rem 0"><b style="color:#C8A951">Lapstone Intel</b> · Philadelphia Construction & Real Estate Intelligence<br>Data: FRED · Census ACS · BLS QCEW · Loaded: {datetime.now().strftime("%B %d, %Y %I:%M %p")}</div>',unsafe_allow_html=True)
