
import os
import time
import json
import math
import requests
import pandas as pd
import streamlit as st
from io import StringIO
from typing import List, Dict, Tuple, Optional

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from streamlit_folium import st_folium
import folium
import pgeocode
import matplotlib.pyplot as plt
import random

# >>> Streamlit requires this to be the FIRST Streamlit call <<<
st.set_page_config(page_title="Education Desert Dashboard (Secure-Ready)", layout="wide")

# ------------------------------
# Auth Gate (optional)
# ------------------------------
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # Prefer Streamlit Secrets, fallback to environment
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

APP_PASSWORD = get_secret("APP_PASSWORD")
if APP_PASSWORD:
    with st.sidebar:
        pw = st.text_input("Enter app password", type="password")
    if pw != APP_PASSWORD:
        st.stop()

st.title("Education Desert & Market Analysis — Secure-Ready")
st.caption("This build avoids hardcoding keys and offers privacy controls for addresses.")

# ------------------------------
# Helper: Safe requests with basic retry
# ------------------------------
def http_get(url: str, params: Dict[str, str], retries: int = 3, timeout: int = 30) -> requests.Response:
    last_exc = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception as e:
            last_exc = e
            time.sleep(0.7)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed GET for {url} with {params}")

# ------------------------------
# Sidebar Inputs
# ------------------------------
with st.sidebar:
    st.header("Settings & Privacy")
    api_key = st.text_input(
        "Census API key",
        value=get_secret("CENSUS_API_KEY", ""),
        type="password",
        help="Stored in Streamlit Secrets or environment; not committed to GitHub."
    )
    if not api_key:
        st.warning("No API key provided. Add CENSUS_API_KEY in Streamlit Secrets or enter it here.")
    st.markdown("**Geography:** ZIP Code Tabulation Areas (ZCTA)")
    default_zips = ["19104", "19139", "19143", "19142", "19153", "19151", "19131"]
    zcta_list = st.text_input(
        "ZCTAs to Analyze (comma-separated)",
        value=",".join(default_zips),
        help="Enter nearby ZCTAs (ZIP Code Tabulation Areas) separated by commas."
    )
    zcta_list = [z.strip() for z in zcta_list.split(",") if z.strip()]
    st.divider()
    st.subheader("School Site Addresses")
    st.caption("Upload a CSV with 'name,address' OR type them below. These are not stored server-side.")
    uploaded = st.file_uploader("Upload CSV of sites (name,address)", type=["csv"])
    addr1 = st.text_input("Address 1 (optional)")
    name1 = st.text_input("Name 1", value="School A" if addr1 else "")
    addr2 = st.text_input("Address 2 (optional)")
    name2 = st.text_input("Name 2", value="School B" if addr2 else "")
    st.divider()
    st.subheader("Privacy Controls")
    jitter = st.checkbox("Jitter coordinates (±150m)", value=False, help="Obscures exact locations on the public map.")
    hide_exact = st.checkbox("Show only neighborhood markers", value=False, help="Aggregates to ZIP centroids instead of exact address points.")
    st.divider()
    st.markdown("**Scoring Weights** (adjust to taste)")
    w_no_hs = st.slider("% Adults w/o HS Diploma", 0.0, 1.0, 0.45, 0.05)
    w_kids = st.slider("% Under 18", 0.0, 1.0, 0.30, 0.05)
    w_income = st.slider("Inverse Median Income", 0.0, 1.0, 0.25, 0.05)
    normalize_choice = st.selectbox("Normalization", ["z-score", "min-max"])

# Build sites_df
sites = []
if uploaded is not None:
    try:
        up = pd.read_csv(uploaded)
        if all(c in up.columns for c in ["name", "address"]):
            sites.extend(up.to_dict(orient="records"))
        else:
            st.warning("Uploaded CSV must include 'name' and 'address' columns.")
    except Exception as e:
        st.warning(f"Could not read uploaded CSV: {e}")

if addr1:
    sites.append({"name": name1 or "Site 1", "address": addr1})
if addr2:
    sites.append({"name": name2 or "Site 2", "address": addr2})

sites_df = pd.DataFrame(sites) if sites else pd.DataFrame(columns=["name","address"])

# ------------------------------
# Geocode addresses (Nominatim)
# ------------------------------
geolocator = Nominatim(user_agent="cca_edu_desert_app_secure")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

def geocode_address(addr: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        loc = geocode(addr)
        if loc is None:
            return None, None
        return loc.latitude, loc.longitude
    except Exception:
        return None, None

if not sites_df.empty:
    sites_df["lat"], sites_df["lon"] = zip(*sites_df["address"].map(geocode_address))
    # Jitter or hide
    if hide_exact:
        # Replace with ZIP centroid
        nomi = pgeocode.Nominatim("us")
        def to_zip_centroid(addr, lat, lon):
            try:
                rev = geolocator.reverse((lat, lon), exactly_one=True, language="en")
                postcode = (rev.raw.get("address", {}) or {}).get("postcode")
            except Exception:
                postcode = None
            pc = nomi.query_postal_code(postcode)
            if pc is None or pd.isna(pc.latitude) or pd.isna(pc.longitude):
                return lat, lon
            return float(pc.latitude), float(pc.longitude)
        sites_df["lat"], sites_df["lon"] = zip(*[to_zip_centroid(a, la, lo) for a, la, lo in zip(sites_df["address"], sites_df["lat"], sites_df["lon"])])
    elif jitter:
        def jitter_point(la, lo, meters=150):
            if pd.isna(la) or pd.isna(lo):
                return la, lo
            dlat = (random.uniform(-meters, meters) / 111_111.0)
            dlon = (random.uniform(-meters, meters) / (111_111.0 * max(math.cos(math.radians(la)), 1e-6)))
            return la + dlat, lo + dlon
        sites_df["lat"], sites_df["lon"] = zip(*[jitter_point(la, lo) for la, lo in zip(sites_df["lat"], sites_df["lon"])])

# ------------------------------
# Fetch ACS data for ZCTAs
# ------------------------------
def fetch_acs_zcta(zcta: str, key: str) -> Optional[Dict[str, float]]:
    base = "https://api.census.gov/data/2022/acs/acs5"
    edu_vars = [f"B15003_{i:03d}E" for i in range(1, 26)]
    age_vars = [f"B01001_{i:03d}E" for i in range(1, 50)]
    income_var = ["B19013_001E"]
    vars_all = ",".join(edu_vars + age_vars + income_var)
    params = {"get": vars_all, "for": f"zip code tabulation area:{zcta}", "key": key}
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        cols, vals = data[0], data[1]
        rec = dict(zip(cols, vals))
        less_than_hs = sum(float(rec.get(f"B15003_{i:03d}E", 0) or 0) for i in range(2, 17))
        total_25p = float(rec.get("B15003_001E", 0) or 0)
        pct_no_hs = (less_than_hs / total_25p * 100.0) if total_25p > 0 else None
        under18_indices_m = list(range(3, 10))
        under18_indices_f = list(range(27, 34))
        total_pop = float(rec.get("B01001_001E", 0) or 0)
        under18 = sum(float(rec.get(f"B01001_{i:03d}E", 0) or 0) for i in under18_indices_m + under18_indices_f)
        pct_under18 = (under18 / total_pop * 100.0) if total_pop > 0 else None
        med_income = float(rec.get("B19013_001E", 0) or 0) if rec.get("B19013_001E") not in (None, "") else None
        return {"zcta": zcta, "pct_no_hs": pct_no_hs, "pct_under18": pct_under18, "median_income": med_income, "total_pop": total_pop}
    except Exception:
        return None

def fetch_many(zctas: List[str], key: str) -> pd.DataFrame:
    rows = []
    for z in zctas:
        rec = fetch_acs_zcta(z, key)
        if rec:
            rows.append(rec)
    return pd.DataFrame(rows)

if not api_key:
    st.error("Provide a Census API key (sidebar) or set CENSUS_API_KEY in Streamlit Secrets.")
    st.stop()

with st.spinner("Fetching ACS data..."):
    df = fetch_many(zcta_list, api_key)

if df.empty:
    st.error("No data returned. Check your API key and ZCTAs.")
    st.stop()

# ------------------------------
# Scoring
# ------------------------------
def minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if math.isclose(hi, lo):
        return pd.Series([0.0]*len(series), index=series.index)
    return (series - lo) / (hi - lo)

def zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std(ddof=0)
    if math.isclose(sigma, 0.0):
        return pd.Series([0.0]*len(series), index=series.index)
    return (series - mu) / (sigma)

work = df.copy()
if normalize_choice == "min-max":
    work["no_hs_n"] = minmax(work["pct_no_hs"])
    work["kids_n"] = minmax(work["pct_under18"])
    work["income_n"] = 1 - minmax(work["median_income"])
else:
    work["no_hs_n"] = zscore(work["pct_no_hs"])
    work["kids_n"] = zscore(work["pct_under18"])
    work["income_n"] = -zscore(work["median_income"])

work["desert_score"] = (w_no_hs * work["no_hs_n"]) + (w_kids * work["kids_n"]) + (w_income * work["income_n"])
work = work.sort_values("desert_score", ascending=False)

st.subheader("Education Desert Ranking (by ZCTA)")
st.dataframe(work[["zcta", "pct_no_hs", "pct_under18", "median_income", "total_pop", "desert_score"]].round(2), use_container_width=True)

# ------------------------------
# Simple charts
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    top = work.head(10).sort_values("desert_score")
    ax1.barh(top["zcta"].astype(str), top["desert_score"])
    ax1.set_title("Top 10 ZCTAs by Desert Score")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(work["median_income"], work["pct_no_hs"])
    ax2.set_xlabel("Median Household Income")
    ax2.set_ylabel("% Adults w/o HS")
    ax2.set_title("Income vs. Educational Attainment")
    st.pyplot(fig2)

# ------------------------------
# Map
# ------------------------------
st.subheader("Interactive Map")
st.caption("Markers show site locations (jittered or aggregated if selected). ZCTA circles reflect 'desert_score'.")

nomi = pgeocode.Nominatim("us")

def zcta_centroid(zipcode: str) -> Tuple[Optional[float], Optional[float]]:
    info = nomi.query_postal_code(zipcode)
    if info is None or pd.isna(info.latitude) or pd.isna(info.longitude):
        return None, None
    return float(info.latitude), float(info.longitude)

work["lat"], work["lon"] = zip(*work["zcta"].map(zcta_centroid))

# Center map: average of available points
def mean_or_default(series, default):
    s = series.dropna()
    return s.mean() if not s.empty else default

mean_lat = mean_or_default(work["lat"], 39.95)
mean_lon = mean_or_default(work["lon"], -75.2)

m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="cartodbpositron")

# Add site markers
if not sites_df.empty:
    for _, r in sites_df.iterrows():
        if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
            folium.Marker(
                location=[r["lat"], r["lon"]],
                popup=f"<b>{r['name']}</b>",
                icon=folium.Icon(color="blue", icon="graduation-cap", prefix="fa")
            ).add_to(m)

# Add ZCTA circles
def minmax_for_radius(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if math.isclose(hi, lo):
        return pd.Series([0.5]*len(series), index=series.index)
    return (series - lo) / (hi - lo)

radius_mm = minmax_for_radius(work["desert_score"])
for idx, r in work.iterrows():
    if pd.notna(r["lat"]) and pd.notna(r["lon"]):
        color = "#d73027" if r["desert_score"] > work["desert_score"].median() else "#1a9850"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + 10 * float(radius_mm.loc[idx]),
            fill=True,
            fill_opacity=0.7,
            color=color,
            popup=(
                f"<b>ZCTA {r['zcta']}</b><br/>"
                f"Desert Score: {r['desert_score']:.2f}<br/>"
                f"% No HS: {r['pct_no_hs']:.1f}%<br/>"
                f"% Under 18: {r['pct_under18']:.1f}%<br/>"
                f"Median Income: ${r['median_income']:.0f}"
            )
        ).add_to(m)

st_folium(m, width=None, height=600)

with st.expander("Security & Privacy Tips"):
    st.markdown(
        """
- **API Keys**: Do **not** hardcode keys in the repo. Use Streamlit **Secrets** (Cloud > App > Settings > *Secrets*) or env vars.
- **Addresses**: If sharing publicly, enable *Jitter* or *Neighborhood-only* to avoid exposing exact locations.
- **Auth**: Set an `APP_PASSWORD` in secrets to gate the app for your group.
- **GitHub**: Use a **private** repo if you need to keep code and discussions internal.
- **Colab**: Tunnels are temporary and public; avoid pasting secrets directly in cells.
        """
    )
