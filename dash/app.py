import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="HDB Resale Intelligence", layout="wide")
st.title("HDB Resale Intelligence")

# --- Load fully processed dataset ---
ROOT = Path(__file__).resolve().parents[1]
features_path = ROOT / "data" / "processed" / "features_real.csv"

@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize headers
    df.columns = [c.strip() for c in df.columns]

    # numeric coercion for safety
    num_cols = [
        "resale_price", "floor_area_sqm", "price_per_sqm",
        "remaining_lease_years", "storey_mid",
        "lat", "lon", "dist_to_mrt_m", "dist_to_school_m",
        "lease_commence_year"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # month -> datetime + year
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["year"] = df["month"].dt.year

    # unify lease start column name to "lease_commence_date" to match your header spec
    if "lease_commence_date" not in df.columns and "lease_commence_year" in df.columns:
        df = df.rename(columns={"lease_commence_year": "lease_commence_date"})

    # tidy text cols
    for c in ["town", "flat_type", "street_name", "block", "flat_model", "storey_range", "remaining_lease"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # create address (optional, not displayed unless you add it)
    if {"block","street_name"}.issubset(df.columns):
        df["address"] = (df["block"].fillna("").astype(str).str.strip() + " " +
                         df["street_name"].fillna("").astype(str).str.strip()).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

if not features_path.exists():
    st.error("`features_real.csv` not found. Run the real pipeline first:\n\n`python -m src.flows.etl_flow_real`")
    st.stop()

df = load_df(features_path)
df_all = df.copy()

# ================= Sidebar filters =================
st.sidebar.header("Filters")

# Year
if "year" in df.columns and not df_all["year"].dropna().empty:
    years = sorted(df_all["year"].dropna().unique().tolist())
    sel_years = st.sidebar.multiselect("Year", years, default=years)
    if sel_years:
        df = df[df["year"].isin(sel_years)]

# Towns (24)
if "town" in df.columns:
    towns = sorted(df_all["town"].dropna().unique().tolist())
    sel_towns = st.sidebar.multiselect("Town (24)", towns, default=towns)
    if sel_towns:
        df = df[df["town"].isin(sel_towns)]

# HDB type
if "flat_type" in df.columns:
    types_ = sorted(df_all["flat_type"].dropna().unique().tolist())
    sel_types = st.sidebar.multiselect("HDB type", types_, default=types_)
    if sel_types:
        df = df[df["flat_type"].isin(sel_types)]

# Location text search
loc_q = st.sidebar.text_input("Street/Block contains", value="", placeholder="e.g. ANG MO KIO, AVE 10, 406")
if loc_q:
    q = loc_q.strip().lower()
    mask = pd.Series(False, index=df.index)
    if "street_name" in df.columns:
        mask |= df["street_name"].str.lower().str.contains(q, na=False)
    if "block" in df.columns:
        mask |= df["block"].str.lower().str.contains(q, na=False)
    if "address" in df.columns:
        mask |= df["address"].str.lower().str.contains(q, na=False)
    df = df[mask]

# Floor area
if "floor_area_sqm" in df.columns and not df_all["floor_area_sqm"].dropna().empty:
    mn, mx = float(df_all["floor_area_sqm"].min()), float(df_all["floor_area_sqm"].max())
    a_min, a_max = st.sidebar.slider("Floor area (sqm)", min_value=mn, max_value=mx, value=(mn, mx))
    df = df[(df["floor_area_sqm"] >= a_min) & (df["floor_area_sqm"] <= a_max)]

# ================= KPIs =================
kpi_cols = st.columns(4)
fmt = lambda x, d=0: ("–" if pd.isna(x) else f"{x:,.{d}f}")

with kpi_cols[0]:
    st.metric("Rows (filtered)", f"{len(df):,}")
    st.caption(f"Total rows: {len(df_all):,}")
with kpi_cols[1]:
    st.metric("Avg resale price", fmt(df["resale_price"].mean() if "resale_price" in df else float("nan"), 0))
with kpi_cols[2]:
    st.metric("Avg price / sqm", fmt(df["price_per_sqm"].mean() if "price_per_sqm" in df else float("nan"), 0))
with kpi_cols[3]:
    st.metric("Avg remaining lease (yrs)", fmt(df["remaining_lease_years"].mean() if "remaining_lease_years" in df else float("nan"), 1))

st.divider()

# ================= Table with EXACT headers + distances =================
st.subheader("HDB Resale Transactions")

# Exact header order per your CSV + appended distances
desired_order = [
    "month", "town", "flat_type", "block", "street_name",
    "storey_range", "floor_area_sqm", "flat_model",
    "lease_commence_date", "remaining_lease", "resale_price",
    "dist_to_mrt_m", "dist_to_school_m"
]

# prepare display frame
tbl = df.copy()

# if month exists, show as YYYY-MM for clarity but keep column name "month"
if "month" in tbl.columns:
    # avoid messing with original: display-only string
    mstr = tbl["month"].dt.strftime("%Y-%m")
    tbl = tbl.copy()
    tbl["month"] = mstr

# ensure both distance columns exist for consistent headers (fill if missing)
for c in ["dist_to_mrt_m", "dist_to_school_m"]:
    if c not in tbl.columns:
        tbl[c] = pd.NA

# Subset to desired columns that exist
present = [c for c in desired_order if c in tbl.columns]
df_display = tbl[present]

# Sort by date desc if available
if "month" in df.columns and df["month"].notna().any():
    # Use the original datetime column in df (not the string in df_display)
    order = df["month"].sort_values(ascending=False).index
    df_display = df_display.loc[order.intersection(df_display.index)]

st.dataframe(df_display.head(200), use_container_width=True)
st.caption("Showing first 200 rows. Use filters above to refine; download below for full filtered result.")

# Download filtered (full, untruncated)
csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered CSV",
    data=csv_bytes,
    file_name="hdb_resale_filtered.csv",
    mime="text/csv",
)

st.divider()

# ================= Optional visuals (unchanged) =================
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Price per sqm by Town")
    if {"town", "price_per_sqm"}.issubset(df.columns) and not df.empty:
        st.bar_chart(df.groupby("town")["price_per_sqm"].mean().sort_values(ascending=False))
with c2:
    st.subheader("Price per sqm vs Floor area")
    if {"price_per_sqm", "floor_area_sqm"}.issubset(df.columns) and not df.empty:
        st.scatter_chart(df[["floor_area_sqm", "price_per_sqm"]].dropna().rename(columns={
            "floor_area_sqm": "x", "price_per_sqm": "y"
        }))

if {"lat", "lon"}.issubset(df.columns) and not df[["lat", "lon"]].dropna().empty:
    st.subheader("Map of transactions")
    plot_df = df[["lat", "lon"]].dropna().copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)
        st.caption("Showing a random sample of 5,000 points for performance.")
    st.map(plot_df, latitude="lat", longitude="lon", zoom=11)
