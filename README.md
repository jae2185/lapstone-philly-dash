# 🏗️ Lapstone Intel — Philadelphia Construction & Real Estate Dashboard

A comprehensive Streamlit dashboard built for [Lapstone LLC](https://www.lapstonellc.com/) that tracks the Philadelphia metro area's construction economy, rental demand dynamics, and demographic trends.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 What It Does

This dashboard pulls **live data** from three major public sources to provide an all-in-one intelligence tool for Philadelphia-area construction and real estate investment decisions.

### Dashboard Tabs

| Tab | What You'll Find |
|-----|-----------------|
| **📊 Overview** | Unemployment, construction employment, building permits, GDP, CPI shelter vs. all items, median income |
| **🏗️ Construction** | Permit breakdowns (single-family vs. multi-family), construction share of employment, QCEW industry detail, annual permit trends |
| **🏠 Rental Demand** | Young professional (25-34) concentration, rent burden analysis, tract-level demand scoring, affordability matrix |
| **👥 Demographics** | Population trends, age distribution, education attainment across counties |
| **🗺️ Regional** | Cross-county comparison scorecard, home values, unemployment, **Blue-Collar Rental Opportunity Score** |

### Blue-Collar Rental Opportunity Score

A composite metric (0–100) that weights:
- **Young Adult Concentration** (25%) — % of population age 25–34
- **Renter Prevalence** (25%) — % of renter-occupied housing
- **Rent Affordability** (25%) — Rent-to-income ratio (lower = better)
- **Employment Strength** (25%) — Local unemployment (lower = better)

Covers: Philadelphia, Montgomery, Bucks, Delaware, Chester, Berks (Reading), Camden NJ, Burlington NJ, Gloucester NJ.

## 🗂️ Data Sources

| Source | Data | Refresh |
|--------|------|---------|
| [FRED (Federal Reserve)](https://fred.stlouisfed.org/) | Unemployment, employment, building permits, GDP, CPI, income, homeownership, population | Monthly/Quarterly |
| [U.S. Census Bureau (ACS 5-Year)](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Age demographics, rent, income, education, tenure — county & tract level | Annual |
| [BLS QCEW](https://www.bls.gov/cew/) | Construction industry employment, wages, establishment counts by sub-industry | Quarterly |

All data is cached (6–12 hours) for fast loading.

## 🚀 Quick Start

### 1. Get Free API Keys

- **FRED**: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **Census**: [https://api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/philly-construction-dashboard.git
cd philly-construction-dashboard
pip install -r requirements.txt
```

### 3. Run Locally

```bash
# Option A: Enter keys in the sidebar UI
streamlit run app.py

# Option B: Set environment variables
export FRED_API_KEY="your_fred_key"
export CENSUS_API_KEY="your_census_key"
streamlit run app.py
```

### 4. Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Add secrets in **Settings → Secrets**:

```toml
FRED_API_KEY = "your_fred_key"
CENSUS_API_KEY = "your_census_key"
```

## 📁 Project Structure

```
philly-construction-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme configuration (dark + gold accent)
└── README.md
```

## 🔧 Configuration

### Streamlit Secrets (for Cloud deployment)

Create `.streamlit/secrets.toml` (gitignored):

```toml
FRED_API_KEY = "your_key"
CENSUS_API_KEY = "your_key"
```

### Customizing Counties

Edit the `COUNTY_FIPS` dictionary in `app.py` to add/remove comparison regions.

## 📝 Notes

- **Census tract-level data** uses ACS 5-Year estimates (most recent: 2019-2023), which are the most granular available for small geographies
- **FRED data** updates monthly/quarterly depending on the series
- **BLS QCEW** data is published ~6 months after each quarter
- The "Young Professional Demand Score" is a custom composite — adjust weights in the code to match your investment thesis
- All charts are interactive (Plotly) — hover, zoom, pan, and export as PNG

## License

MIT — Built for Lapstone LLC.
