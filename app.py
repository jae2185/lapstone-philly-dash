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
    "labor_force":"LAUCN421010000000006","med_income":"MHIPA42101A052NCEN",
    "homeown":"HOWNRATEACS042101","unemp_msa":"PHIL942URN",
    "const_emp":"SMU42979611500000001SA","nonfarm":"SMU42379800000000001SA",
    "permits_tot":"PHIL942BPPRIV","permits_1u":"PHIL942BP1FH",
    "gdp":"NGMP37980","cpi_shelter":"CUURA102SAH1","cpi_all":"CUURA102SA0",
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
    for col in TRACT_VARS:
        df[col]=pd.to_numeric(df[col],errors="coerce")
        df.loc[df[col]<=-666666666,col]=np.nan
    df["pop2534"]=df[["B01001_011E","B01001_012E","B01001_035E","B01001_036E"]].sum(axis=1)
    df["pct2534"]=df["pop2534"]/df["B01003_001E"]*100
    df["renter_pct"]=df["B25003_003E"]/df["B25003_001E"]*100
    df["burden_pct"]=df[["B25070_008E","B25070_009E","B25070_010E","B25070_011E"]].sum(axis=1)/df["B25070_001E"]*100
    df["r2i"]=(df["B25064_001E"]*12)/df["B19013_001E"]*100
    df["unemp"]=df["B23025_005E"]/df["B23025_002E"]*100
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
def build_annual_dataset(fd):
    """Convert monthly FRED series into annual features for forecasting."""
    targets = {
        "Unemployment (%)": "unemp_philly",
        "Permits (Annual)": "permits_tot",
        "GDP ($M)": "gdp",
        "Median Income ($)": "med_income",
    }
    features_keys = ["unemp_philly","permits_tot","gdp","cpi_shelter","cpi_all","const_emp","homeown","med_income"]
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
    if not frames: return pd.DataFrame(), targets
    annual = pd.concat(frames.values(), axis=1).dropna(how="all")
    return annual, targets

def run_forecast(annual, target_col, n_backtest=5):
    """Walk-forward expanding-window Ridge regression forecast."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    if target_col not in annual.columns or len(annual) < 8:
        return None
    df = annual.dropna(subset=[target_col]).copy()
    # Create lag features (1-year lag of all available columns)
    feature_cols = [c for c in df.columns]
    lagged = df[feature_cols].shift(1)
    lagged.columns = [f"{c}_lag1" for c in feature_cols]
    combo = pd.concat([df[[target_col]], lagged], axis=1).dropna()
    # Drop any feature columns that still have NaN
    combo = combo.dropna(axis=1)
    if len(combo) < 6: return None
    X = combo.drop(columns=[target_col])
    y = combo[target_col]
    feat_names = X.columns.tolist()
    # Walk-forward backtest
    bt_results = []
    min_train = max(5, len(combo) - n_backtest - 1)
    for split in range(min_train, len(combo) - 1):
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_te, y_te = X.iloc[split:split+1], y.iloc[split:split+1]
        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        m = Ridge(alpha=1.0); m.fit(X_tr_s, y_tr)
        pred = m.predict(X_te_s)[0]
        bt_results.append({"year": y_te.index[0], "actual": y_te.values[0], "predicted": pred})
    bt = pd.DataFrame(bt_results)
    if bt.empty: return None
    bt["error"] = bt["predicted"] - bt["actual"]
    bt["abs_pct_error"] = (bt["error"].abs() / bt["actual"].abs()) * 100
    mape = bt["abs_pct_error"].mean()
    rmse = np.sqrt((bt["error"]**2).mean())
    # Final forecast: train on ALL data, predict next year
    sc = StandardScaler(); X_s = sc.fit_transform(X); m = Ridge(alpha=1.0); m.fit(X_s, y)
    last_row = df[feature_cols].iloc[-1:].copy()
    last_row.columns = [f"{c}_lag1" for c in feature_cols]
    # Only use columns that survived NaN filtering
    last_row = last_row[[c for c in feat_names if c in last_row.columns]]
    last_row = last_row.fillna(last_row.mean())
    forecast_val = m.predict(sc.transform(last_row))[0]
    forecast_year = int(df.index[-1]) + 1
    # Confidence interval from backtest residuals
    std_err = bt["error"].std()
    ci_low = forecast_val - 1.96 * std_err
    ci_high = forecast_val + 1.96 * std_err
    # Feature importance
    coefs = pd.Series(m.coef_, index=feat_names).abs().sort_values(ascending=False)
    return {
        "backtest": bt, "mape": mape, "rmse": rmse,
        "forecast_year": forecast_year, "forecast_val": forecast_val,
        "ci_low": ci_low, "ci_high": ci_high,
        "last_actual_year": int(df.index[-1]), "last_actual_val": y.iloc[-1],
        "importance": coefs, "n_train": len(combo),
    }

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
            "Rent-to-Income Ratio (%)","Demand Score"])
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
                    "Demand Score":("score","Score","","")}
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

        annual, targets = build_annual_dataset(fd_full)

        if annual.empty:
            st.warning("Not enough data to build forecasts.")
        else:
            st.markdown(f"**{len(annual)} years** of annual data assembled ({int(annual.index.min())}–{int(annual.index.max())})")

            for target_label, target_key in targets.items():
                if target_key not in annual.columns:
                    continue

                st.markdown(f'<div class="section-label">{target_label}</div>', unsafe_allow_html=True)
                result = run_forecast(annual, target_key)

                if result is None:
                    st.markdown("_Insufficient data for this forecast._")
                    continue

                # Metrics row
                fc1, fc2, fc3, fc4 = st.columns(4)
                is_dollar = "$" in target_label
                is_pct = "%" in target_label
                pref = "$" if is_dollar else ""
                suf = "%" if is_pct else ""

                def fv(v):
                    if is_dollar and abs(v) >= 1e6: return f"${v/1e6:,.0f}M"
                    if is_dollar and abs(v) >= 1e3: return f"${v/1e3:,.0f}K"
                    if is_dollar: return f"${v:,.0f}"
                    if is_pct: return f"{v:.1f}%"
                    return f"{v:,.0f}"

                chg = result["forecast_val"] - result["last_actual_val"]
                chg_pct = (chg / abs(result["last_actual_val"])) * 100 if result["last_actual_val"] != 0 else 0

                with fc1: st.metric(f"{result['last_actual_year']} (Actual)", fv(result["last_actual_val"]))
                with fc2: st.metric(f"{result['forecast_year']} (Forecast)", fv(result["forecast_val"]),
                    f"{chg_pct:+.1f}%", delta_color="normal" if not is_pct else ("inverse" if "Unemployment" in target_label else "normal"))
                with fc3: st.metric("95% CI", f"{fv(result['ci_low'])} – {fv(result['ci_high'])}")
                with fc4: st.metric("Backtest MAPE", f"{result['mape']:.1f}%", help="Mean Absolute Percentage Error on walk-forward backtest. Lower = more accurate.")

                # Charts: backtest + forecast
                bt = result["backtest"]
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=bt["year"], y=bt["actual"], mode="lines+markers",
                        name="Actual", line=dict(color=C["teal"], width=2.5), marker=dict(size=6)))
                    fig.add_trace(go.Scatter(x=bt["year"], y=bt["predicted"], mode="lines+markers",
                        name="Predicted", line=dict(color=C["gold"], width=2, dash="dash"), marker=dict(size=6, symbol="diamond")))
                    # Add forecast point
                    fig.add_trace(go.Scatter(x=[result["forecast_year"]], y=[result["forecast_val"]],
                        mode="markers", name=f"{result['forecast_year']} Forecast",
                        marker=dict(size=12, color=C["coral"], symbol="star")))
                    # CI band
                    fig.add_shape(type="rect", x0=result["forecast_year"]-0.3, x1=result["forecast_year"]+0.3,
                        y0=result["ci_low"], y1=result["ci_high"],
                        fillcolor="rgba(231,111,81,0.15)", line=dict(width=0))
                    fmt = {}
                    if is_dollar: fmt["yaxis_tickprefix"] = "$"
                    if is_pct: fmt["yaxis_ticksuffix"] = "%"
                    fig.update_layout(**BL, title=dict(text=f"Backtest + Forecast: {target_label}", font=dict(size=14)), **fmt)
                    st.plotly_chart(fig, use_container_width=True)

                with ch2:
                    # Feature importance
                    imp = result["importance"].head(6)
                    clean_names = [n.replace("_lag1","").replace("_"," ").title() for n in imp.index]
                    fig = go.Figure(go.Bar(y=clean_names[::-1], x=imp.values[::-1], orientation="h",
                        marker_color=C["teal"],
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"))
                    fig.update_layout(**BL, title=dict(text="Feature Importance (Abs Coefficients)", font=dict(size=14)))
                    st.plotly_chart(fig, use_container_width=True)

            # Methodology note
            with st.expander("📖 Forecast Methodology"):
                st.markdown("""
**Model:** Ridge Regression with L2 regularization (α=1.0)

**Features:** 1-year lagged values of all available FRED indicators (unemployment, permits, GDP, CPI shelter, CPI all items, construction employment, homeownership, median income)

**Validation:** Expanding-window walk-forward backtest — the model is retrained at each step using only data available up to that point, then predicts the next year. This prevents data leakage and simulates real-world forecasting conditions.

**Confidence Interval:** 95% CI derived from the standard deviation of backtest residuals (±1.96σ)

**MAPE:** Mean Absolute Percentage Error across all backtest windows. Under 5% is excellent; under 10% is good.

**Limitations:** Annual frequency limits sample size. Model assumes linear relationships and stable regime. Structural breaks (pandemics, policy shifts) may not be captured. GDP data lags ~1 year from BEA.
""")

# ── FOOTER ──
st.markdown("---")
st.markdown(f'<div style="text-align:center;color:#5A6270;font-size:.8rem;padding:1rem 0"><b style="color:#C8A951">Lapstone Intel</b> · Philadelphia Construction & Real Estate Intelligence<br>Data: FRED · Census ACS · BLS QCEW · Loaded: {datetime.now().strftime("%B %d, %Y %I:%M %p")}</div>',unsafe_allow_html=True)
