"""
Retail Intelligence — Online Retail RFM & Market-Basket dashboard (Streamlit)

Run:   streamlit run app.py
Data:  put `online_retail_raw.csv` (or `Online Retail.xlsx`) next to this file,
       or upload it from the first screen.

The cleaning / RFM / Apriori logic is a port of the RFM notebook, so the numbers match it.
"""
from __future__ import annotations

import io
import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Retail Intelligence", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="expanded")

# =====================================================================================
# 1. CONSTANTS
# =====================================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = ["online_retail_raw.csv", "Online Retail.xlsx"]

PALETTE = ["#2E86DE", "#E67E22", "#1B7F4C", "#8E44AD", "#F5B041", "#C0392B", "#16A085", "#7F8C8D"]
BLUE, ORANGE, GREEN, PURPLE = "#2E86DE", "#E67E22", "#1B7F4C", "#8E44AD"

STOCKCODE_CATEGORIES = {
    "POST": "Shipping", "DOT": "Shipping", "C2": "Shipping",
    "D": "Discount", "M": "Manual Adjustment", "S": "Samples",
    "BANK CHARGES": "Financial Charges", "AMAZONFEE": "Fees", "CRUK": "Commission",
    "gift_0001_20": "Gift Voucher", "gift_0001_10": "Gift Voucher", "gift_0001_30": "Gift Voucher",
}

SEGMENT_GRID = {   # rows: R score 5..1, columns: FM score 1..5
    5: ["New Customers", "Potential Loyalists", "Potential Loyalists", "Champions", "Champions"],
    4: ["Promising", "Potential Loyalists", "Potential Loyalists", "Loyal Customers", "Loyal Customers"],
    3: ["About to Sleep", "About to Sleep", "Need Attention", "Loyal Customers", "Loyal Customers"],
    2: ["Hibernating", "Hibernating", "At Risk", "At Risk", "Can't Lose Them"],
    1: ["Lost", "Hibernating", "At Risk", "At Risk", "Can't Lose Them"],
}
SEGMENT_MAP = {
    "Champions": "Champions",
    "Loyal Customers": "Loyal", "Potential Loyalists": "Loyal", "New Customers": "Loyal", "Promising": "Loyal",
    "Need Attention": "At Risk", "About to Sleep": "At Risk", "At Risk": "At Risk", "Can't Lose Them": "At Risk",
    "Hibernating": "Lost", "Lost": "Lost",
}
SEGMENT_ORDER = ["Champions", "Loyal", "At Risk", "Lost"]
SEGMENT_COLORS = {"Champions": "#1B7F4C", "Loyal": "#2E86DE", "At Risk": "#E67E22", "Lost": "#7F8C8D"}
DETAILED_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Promising",
                  "Need Attention", "About to Sleep", "At Risk", "Can't Lose Them", "Hibernating", "Lost"]
DETAILED_COLORS = {
    "Champions": "#1B7F4C", "Loyal Customers": "#4CAF7A", "Potential Loyalists": "#9BCB6B",
    "New Customers": "#2E86DE", "Promising": "#74B9FF", "Need Attention": "#F5B041",
    "About to Sleep": "#F8C471", "At Risk": "#E67E22", "Can't Lose Them": "#C0392B",
    "Hibernating": "#95A5A6", "Lost": "#7F8C8D",
}
PLAYBOOK = {
    "Champions": "Reward them: early access, loyalty perks, ask for reviews and referrals.",
    "Loyal": "Upsell and cross-sell with bundles; keep them engaged so they move up to Champions.",
    "At Risk": "Win-back campaigns with a personal touch. **Can't Lose Them** is the top priority (big spenders drifting away).",
    "Lost": "Low-cost reactivation only (one email / a discount). Do not spend heavily here.",
}
PRODUCT_SEGMENT_ORDER = ["Star Products", "Core Sellers", "New Arrivals", "Niche High-Value",
                         "Volume Drivers", "Occasional Sellers", "Fading Products", "Dead Stock"]
PRODUCT_SEGMENT_COLORS = {
    "Star Products": "#1B7F4C", "Core Sellers": "#4CAF7A", "New Arrivals": "#2E86DE",
    "Niche High-Value": "#8E44AD", "Volume Drivers": "#F5B041", "Occasional Sellers": "#95A5A6",
    "Fading Products": "#E67E22", "Dead Stock": "#7F8C8D",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COUNTRY_FIX = {"EIRE": "Ireland", "RSA": "South Africa", "USA": "United States", "Czech Republic": "Czechia"}


# =====================================================================================
# 2. COMPAT HELPERS (work on old and new Streamlit versions)
# =====================================================================================
def is_dark() -> bool:
    if "dark" in st.session_state:
        return bool(st.session_state["dark"])
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def plot(fig: go.Figure, key: str | None = None, height: int | None = None) -> None:
    """Render a Plotly figure with the template that matches the current mode."""
    dark = is_dark()
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(title_text=""),
    )
    if fig.layout.title.text:
        fig.update_layout(title=dict(x=0.0, xanchor="left", font=dict(size=16)))
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def download_csv(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)


def kpi(col, label: str, value: str, help: str | None = None) -> None:
    with col:
        with st.container(border=True):
            st.metric(label, value, help=help)


def fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    ax = abs(x)
    if ax >= 1e6:
        return f"£{x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"£{x / 1e3:,.1f}K"
    return f"£{x:,.0f}"


def short(text: str, n: int = 28) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


# =====================================================================================
# 3. DATA PIPELINE (port of the notebook)
# =====================================================================================
def read_raw(buf, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        raw = pd.read_csv(buf, dtype={"InvoiceNo": str, "StockCode": str}, encoding_errors="replace")
    else:
        raw = pd.read_excel(buf, dtype={"InvoiceNo": str, "StockCode": str})
    missing = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
               "UnitPrice", "CustomerID", "Country"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in the file: {sorted(missing)}")
    return raw


def quintile_score(values: pd.Series, reverse: bool = False) -> pd.Series:
    labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
    return pd.qcut(values.rank(method="first"), q=5, labels=labels).astype(int)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sections 2-6 of the notebook: classify lines, fix descriptions, recover prices, flag duplicates."""
    df = raw.copy()
    stats: dict = {"raw_rows": len(df)}
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # --- classify lines (products vs shipping / fees / adjustments ...) ---
    df["ItemType"] = df["StockCode"].map(STOCKCODE_CATEGORIES).fillna("Product")
    is_cancel = df["InvoiceNo"].str.startswith("C")
    inv_adj = (df["Quantity"] < 0) & (df["UnitPrice"] == 0) & df["CustomerID"].isna() & ~is_cancel
    df.loc[inv_adj, "ItemType"] = "Inventory Adjustment"
    bad_debt = ((df["StockCode"] == "B")
                & df["Description"].astype("object").str.contains("Adjust bad debt", case=False, na=False)
                & (df["UnitPrice"] < 0))
    df.loc[bad_debt, "ItemType"] = "Financial Adjustment"
    stats["itemtype_counts"] = df["ItemType"].value_counts().to_dict()

    p = df[df["ItemType"] == "Product"].copy()
    stats["product_rows"] = len(p)

    # --- one description per StockCode (the most frequent one) ---
    desc = (p.dropna(subset=["Description"]).assign(Description=lambda d: d["Description"].str.strip())
            .groupby(["StockCode", "Description"]).size().reset_index(name="n")
            .sort_values(["StockCode", "n"], ascending=[True, False])
            .drop_duplicates("StockCode").set_index("StockCode")["Description"])
    p["Description"] = p["StockCode"].map(desc)
    p["Description_Missing"] = p["Description"].isna()
    p["Description"] = p["Description"].fillna("Unknown Product")

    # --- recover zero unit prices: (1) description with ONE known price, (2) nearest price of same StockCode ---
    hist = p.loc[p["UnitPrice"] > 0, ["StockCode", "InvoiceDate", "UnitPrice"]].copy()
    zero_mask = p["UnitPrice"] == 0
    stats["zero_price_rows"] = int(zero_mask.sum())

    key = p["Description"].str.strip().str.upper()
    valid_idx = p.index[p["UnitPrice"] > 0]
    dp = p.loc[valid_idx, "UnitPrice"].groupby(key[valid_idx]).agg(["nunique", "median"])
    unique_map = dp.loc[dp["nunique"] == 1, "median"]
    sugg = key[zero_mask].map(unique_map)
    fillable = sugg.notna()
    p.loc[sugg[fillable].index, "UnitPrice"] = sugg[fillable]
    stats["recovered_by_description"] = int(fillable.sum())

    rz = p[p["UnitPrice"] == 0].copy()
    rz["_target_index"] = rz.index
    rz = rz.reset_index(drop=True).dropna(subset=["InvoiceDate", "StockCode"]).sort_values("InvoiceDate")
    hist = (hist.dropna(subset=["InvoiceDate", "StockCode"]).rename(
        columns={"InvoiceDate": "Price_Date", "UnitPrice": "Historical_Price"}).sort_values("Price_Date"))
    recovered_nearest = 0
    if len(rz) and len(hist):
        prev = pd.merge_asof(rz, hist, left_on="InvoiceDate", right_on="Price_Date", by="StockCode",
                             direction="backward", allow_exact_matches=False)
        nxt = pd.merge_asof(rz, hist, left_on="InvoiceDate", right_on="Price_Date", by="StockCode",
                            direction="forward", allow_exact_matches=False)
        prev_diff = (prev["InvoiceDate"] - prev["Price_Date"]).abs()
        next_diff = (nxt["Price_Date"] - nxt["InvoiceDate"]).abs()
        use_prev = (prev_diff.notna() & (next_diff.isna() | (prev_diff <= next_diff))).to_numpy()
        best = np.where(use_prev, prev["Historical_Price"].to_numpy(dtype=float),
                        nxt["Historical_Price"].to_numpy(dtype=float))
        matched = ~pd.isna(best)
        targets = rz["_target_index"].to_numpy()
        p.loc[targets[matched], "UnitPrice"] = best[matched]
        recovered_nearest = int(matched.sum())
    stats["recovered_by_nearest"] = recovered_nearest

    before = len(p)
    p = p[p["UnitPrice"] != 0].copy()
    stats["dropped_zero_price"] = before - len(p)

    # --- guests, duplicates (flagged, not deleted), revenue, date features ---
    p["Customer_Type"] = np.where(p["CustomerID"].isna(), "Guest", "Registered Customer")
    base_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    p["Is_Duplicate_Line"] = p.duplicated(subset=base_cols, keep="first")
    stats["duplicate_lines"] = int(p["Is_Duplicate_Line"].sum())
    p["TotalLineRevenue"] = p["Quantity"] * p["UnitPrice"]
    dt = p["InvoiceDate"].dt
    p["Month_Start"] = dt.to_period("M").dt.to_timestamp()
    p["Month_Name"] = dt.month_name()
    p["Day_Name"] = dt.day_name()
    p["Hour"] = dt.hour
    p["Is_UK"] = p["Country"].eq("United Kingdom")
    stats["clean_rows"] = len(p)
    stats["guest_share"] = float((p["Customer_Type"] == "Guest").mean())
    return p, stats


def build_rfm(sales: pd.DataFrame, label_by_code: pd.Series):
    """Sections 8-9: customer RFM + segments, product RFM + segments."""
    ct = sales[sales["CustomerID"].notna()]
    snapshot = ct["InvoiceDate"].max() + pd.Timedelta(days=1)

    cust = (ct.groupby("CustomerID")
            .agg(Last_Purchase=("InvoiceDate", "max"), First_Purchase=("InvoiceDate", "min"),
                 Frequency=("InvoiceNo", "nunique"), Monetary=("TotalLineRevenue", "sum"),
                 Units=("Quantity", "sum")).reset_index())
    cust["CustomerID"] = cust["CustomerID"].astype("int64")
    cust["Recency"] = (snapshot - cust["Last_Purchase"]).dt.days
    country = (ct.groupby(["CustomerID", "Country"]).size().reset_index(name="n")
               .sort_values("n", ascending=False).drop_duplicates("CustomerID"))
    country["CustomerID"] = country["CustomerID"].astype("int64")
    cust = cust.merge(country[["CustomerID", "Country"]], on="CustomerID", how="left")

    cust["R_Score"] = quintile_score(cust["Recency"], reverse=True)
    cust["F_Score"] = quintile_score(cust["Frequency"])
    cust["M_Score"] = quintile_score(cust["Monetary"])
    cust["RFM_Score"] = cust["R_Score"].astype(str) + cust["F_Score"].astype(str) + cust["M_Score"].astype(str)
    cust["FM_Score"] = np.ceil((cust["F_Score"] + cust["M_Score"]) / 2).astype(int)
    cust["Segment_Detailed"] = [SEGMENT_GRID[int(r)][int(fm) - 1] for r, fm in zip(cust["R_Score"], cust["FM_Score"])]
    cust["Segment"] = cust["Segment_Detailed"].map(SEGMENT_MAP)
    cust["Avg_Order_Value"] = cust["Monetary"] / cust["Frequency"]

    desc = sales.groupby("StockCode")["Description"].first()
    prod = (sales.groupby("StockCode")
            .agg(Last_Sale=("InvoiceDate", "max"), First_Sale=("InvoiceDate", "min"),
                 Frequency=("InvoiceNo", "nunique"), Monetary=("TotalLineRevenue", "sum"),
                 Units_Sold=("Quantity", "sum"), Unique_Customers=("CustomerID", "nunique"))
            .reset_index())
    prod["Description"] = prod["StockCode"].map(desc)
    prod["Recency"] = (snapshot - prod["Last_Sale"]).dt.days
    prod["Active_Days"] = (prod["Last_Sale"] - prod["First_Sale"]).dt.days
    prod["Avg_Price"] = prod["Monetary"] / prod["Units_Sold"]
    prod["Product_Label"] = prod["StockCode"].map(label_by_code)
    prod["R_Score"] = quintile_score(prod["Recency"], reverse=True)
    prod["F_Score"] = quintile_score(prod["Frequency"])
    prod["M_Score"] = quintile_score(prod["Monetary"])
    r, f, m = prod["R_Score"], prod["F_Score"], prod["M_Score"]
    new_cutoff = snapshot - pd.Timedelta(days=60)
    conditions = [
        (r >= 4) & (f >= 4) & (m >= 4), prod["First_Sale"] >= new_cutoff,
        (r >= 3) & (f >= 3) & (m >= 3), (r <= 2) & (f >= 3) & (m >= 3),
        (m >= 4) & (f <= 3), (f >= 4) & (m <= 3), (r <= 2) & (f <= 2) & (m <= 2),
    ]
    choices = ["Star Products", "New Arrivals", "Core Sellers", "Fading Products",
               "Niche High-Value", "Volume Drivers", "Dead Stock"]
    prod["Segment"] = np.select(conditions, choices, default="Occasional Sellers")
    return cust, prod, snapshot


@st.cache_resource(show_spinner="Cleaning the data and building RFM… (first run only, ~30-60 s)")
def build_all(source_key: str, _payload) -> SimpleNamespace:
    buf, name = _payload
    raw = read_raw(buf, name)
    clean, stats = clean_data(raw)

    is_cancel = clean["InvoiceNo"].str.startswith("C")
    has_date = clean["InvoiceDate"].notna()
    returns = clean[(clean["Quantity"] < 0) & is_cancel & has_date].copy()
    returns["Return_Units"] = -returns["Quantity"]
    returns["Return_Value"] = -returns["TotalLineRevenue"]
    sales = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0) & (clean["TotalLineRevenue"] > 0)
                  & ~is_cancel & has_date].copy()

    desc_all = clean.groupby("StockCode")["Description"].first()
    label_by_code = (clean["StockCode"].drop_duplicates().to_frame()
                     .assign(Label=lambda d: d["StockCode"] + " - " + d["StockCode"].map(desc_all).astype(str).str.slice(0, 35))
                     .set_index("StockCode")["Label"])

    cust, prod, snapshot = build_rfm(sales, label_by_code)

    # customer segment x product segment (revenue)
    seg_c = cust.set_index("CustomerID")["Segment"]
    seg_p = prod.set_index("StockCode")["Segment"]
    x = sales[sales["CustomerID"].notna()][["CustomerID", "StockCode", "TotalLineRevenue"]].copy()
    x["Customer_Segment"] = x["CustomerID"].astype("int64").map(seg_c)
    x["Product_Segment"] = x["StockCode"].map(seg_p)
    cross = x.groupby(["Customer_Segment", "Product_Segment"])["TotalLineRevenue"].sum().reset_index()

    country_order = (sales.groupby("Country")["TotalLineRevenue"].sum().sort_values(ascending=False).index.tolist())
    stats["customer_rows"] = int(sales["CustomerID"].notna().sum())
    stats["sales_rows"] = len(sales)

    return SimpleNamespace(
        clean=clean, sales=sales, returns=returns, cust=cust, prod=prod, snapshot=snapshot,
        label=label_by_code, prod_segment=seg_p, cross=cross, stats=stats, countries=country_order,
        date_min=sales["InvoiceDate"].min().date(), date_max=sales["InvoiceDate"].max().date(),
    )


def find_source():
    for name in DATA_CANDIDATES:
        for base in (APP_DIR, Path.cwd()):
            path = base / name
            if path.exists():
                return path
    return None


def get_data() -> SimpleNamespace:
    src = find_source()
    if src is not None:
        stat = src.stat()
        return build_all(f"{src.name}-{stat.st_size}-{int(stat.st_mtime)}", (str(src), src.name))
    if "_upload" not in st.session_state:
        st.markdown("<div class='hero'><h1>🛍️ Retail Intelligence</h1>"
                    "<p>Upload the data file to start.</p></div>", unsafe_allow_html=True)
        st.info("`online_retail_raw.csv` / `Online Retail.xlsx` was not found next to `app.py`. "
                "Upload it here (or put it in the app folder and refresh).")
        up = st.file_uploader("Data file (.csv or .xlsx)", type=["csv", "xlsx"])
        if up is None:
            st.stop()
        st.session_state["_upload"] = (up.name, up.getvalue())
        st.rerun()
    name, content = st.session_state["_upload"]
    return build_all(f"upload-{name}-{len(content)}", (io.BytesIO(content), name))


# =====================================================================================
# 4. GLOBAL FILTERS
# =====================================================================================
def default_filters(D) -> dict:
    return dict(dates=(D.date_min, D.date_max), region="All countries", countries=[],
                ctype=["Registered Customer", "Guest"])


def sidebar_filters(D) -> dict:
    saved = st.session_state.setdefault("_filters", default_filters(D))
    st.sidebar.markdown("### 🎛️ Global filters")
    st.sidebar.caption("Apply to Overview, Sales, Returns and Market Basket.")
    d = st.sidebar.date_input("Period", value=saved["dates"], min_value=D.date_min, max_value=D.date_max)
    dates = tuple(d) if isinstance(d, (tuple, list)) and len(d) == 2 else saved["dates"]
    regions = ["All countries", "UK only", "International only"]
    region = st.sidebar.radio("Region", regions, index=regions.index(saved["region"]), horizontal=True)
    countries = st.sidebar.multiselect("Specific countries (empty = all)", D.countries,
                                       default=[c for c in saved["countries"] if c in D.countries])
    ctype = st.sidebar.multiselect("Customer type", ["Registered Customer", "Guest"], default=saved["ctype"])
    if not ctype:
        ctype = ["Registered Customer", "Guest"]
    F = dict(dates=dates, region=region, countries=countries, ctype=ctype)
    st.session_state["_filters"] = F
    if st.sidebar.button("↺ Reset filters"):
        st.session_state["_filters"] = default_filters(D)
        st.rerun()
    return F


def current_filters(D) -> dict:
    return st.session_state.get("_filters") or default_filters(D)


def apply_filters(df: pd.DataFrame, F: dict) -> pd.DataFrame:
    start, end = pd.Timestamp(F["dates"][0]), pd.Timestamp(F["dates"][1]) + pd.Timedelta(days=1)
    mask = (df["InvoiceDate"] >= start) & (df["InvoiceDate"] < end) & df["Customer_Type"].isin(F["ctype"])
    if F["region"] == "UK only":
        mask &= df["Is_UK"]
    elif F["region"] == "International only":
        mask &= ~df["Is_UK"]
    if F["countries"]:
        mask &= df["Country"].isin(F["countries"])
    return df[mask]


def filter_summary(F: dict) -> str:
    parts = [f"{F['dates'][0]:%d %b %Y} → {F['dates'][1]:%d %b %Y}", F["region"]]
    if F["countries"]:
        parts.append(", ".join(F["countries"][:3]) + ("…" if len(F["countries"]) > 3 else ""))
    if len(F["ctype"]) == 1:
        parts.append(F["ctype"][0])
    return " · ".join(parts)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def empty_guard(df: pd.DataFrame, msg: str = "No data for the current filters. Widen the period or clear the country filter.") -> None:
    if df.empty:
        st.warning(msg)
        st.stop()


# =====================================================================================
# 5. PAGES
# =====================================================================================
def page_overview():
    D = get_data(); F = current_filters(D)
    hero("🏠 Overview", "Revenue, customers and geography at a glance.")
    st.caption("Filters: " + filter_summary(F))
    s = apply_filters(D.sales, F); r = apply_filters(D.returns, F)
    empty_guard(s)

    revenue = s["TotalLineRevenue"].sum()
    inv = s.groupby("InvoiceNo")["TotalLineRevenue"].sum()
    ret_rate = r["Return_Value"].sum() / revenue * 100 if revenue else 0
    c = st.columns(5)
    kpi(c[0], "Revenue", fmt_money(revenue), "Sum of Quantity × UnitPrice of valid sales lines.")
    kpi(c[1], "Invoices", f"{inv.size:,}")
    kpi(c[2], "Registered customers", f"{s['CustomerID'].nunique():,}")
    kpi(c[3], "Avg order value", f"£{inv.mean():,.0f}", f"Median order = £{inv.median():,.0f} (the mean is pulled up by wholesale invoices).")
    kpi(c[4], "Return rate (value)", f"{ret_rate:.1f}%", "Returned value ÷ sales value.")

    tab_trend, tab_geo, tab_cust = st.tabs(["📈 Trends", "🌍 Geography", "👥 Customers"])

    with tab_trend:
        a, b, c2 = st.columns([2, 1, 1])
        metric = a.segmented_control("Metric", ["Revenue", "Invoices", "Units", "Customers"], default="Revenue", key="ov_metric") \
            if hasattr(st, "segmented_control") else a.radio("Metric", ["Revenue", "Invoices", "Units", "Customers"], horizontal=True, key="ov_metric")
        metric = metric or "Revenue"
        kind = b.radio("Chart", ["Area", "Bars", "Line"], horizontal=True, key="ov_kind")
        grain = c2.radio("Grain", ["Month", "Week", "Day"], horizontal=True, key="ov_grain")
        freq = {"Month": "MS", "Week": "W-MON", "Day": "D"}[grain]
        t = s.set_index("InvoiceDate").groupby(pd.Grouper(freq=freq)).agg(
            Revenue=("TotalLineRevenue", "sum"), Invoices=("InvoiceNo", "nunique"),
            Units=("Quantity", "sum"), Customers=("CustomerID", "nunique")).reset_index()
        t = t.rename(columns={"InvoiceDate": "Period"})
        show_ret = st.checkbox("Overlay returned value", value=False, key="ov_ret") if metric == "Revenue" else False
        if kind == "Area":
            fig = px.area(t, x="Period", y=metric, title=f"{metric} per {grain.lower()}")
        elif kind == "Bars":
            fig = px.bar(t, x="Period", y=metric, title=f"{metric} per {grain.lower()}")
        else:
            fig = px.line(t, x="Period", y=metric, markers=True, title=f"{metric} per {grain.lower()}")
        fig.update_traces(marker_color=BLUE) if kind == "Bars" else fig.update_traces(line_color=BLUE)
        if show_ret and not r.empty:
            tr = r.set_index("InvoiceDate").groupby(pd.Grouper(freq=freq))["Return_Value"].sum().reset_index()
            fig.add_trace(go.Scatter(x=tr["InvoiceDate"], y=tr["Return_Value"], name="Returns", mode="lines",
                                     line=dict(color=ORANGE, dash="dot")))
        plot(fig, key="ov_trend", height=420)
        last = s["InvoiceDate"].max()
        if grain == "Month" and last.day < 28:
            st.caption(f"ℹ️ The last month is partial (data ends on {last:%d %b %Y}).")

        m = s.groupby("Month_Start")["TotalLineRevenue"].sum()
        hr = s.groupby("Hour")["InvoiceNo"].nunique()
        top20 = D.cust.sort_values("Monetary", ascending=False).head(int(len(D.cust) * 0.2))["Monetary"].sum() / D.cust["Monetary"].sum()
        st.info(f"💡 **Best month:** {m.idxmax():%B %Y} ({fmt_money(m.max())}) · **Busiest hour:** {hr.idxmax()}:00 · "
                f"**Top 20 % of customers** generate {top20:.0%} of registered revenue.")

    with tab_geo:
        a, b, c2 = st.columns(3)
        gm = a.radio("Metric", ["Revenue", "Invoices", "Customers"], horizontal=True, key="geo_metric")
        hide_uk = b.checkbox("Hide United Kingdom", value=True, key="geo_uk")
        view = c2.radio("View", ["Map", "Bars"], horizontal=True, key="geo_view")
        g = s.groupby("Country").agg(Revenue=("TotalLineRevenue", "sum"), Invoices=("InvoiceNo", "nunique"),
                                     Customers=("CustomerID", "nunique")).reset_index()
        if hide_uk:
            g = g[g["Country"] != "United Kingdom"]
        if g.empty:
            st.info("Nothing to show without the UK for these filters.")
        elif view == "Map":
            g["Mapped"] = g["Country"].replace(COUNTRY_FIX)
            fig = px.choropleth(g, locations="Mapped", locationmode="country names", color=gm,
                                hover_name="Country", hover_data=["Revenue", "Invoices", "Customers"],
                                color_continuous_scale="Blues", title=f"{gm} by country")
            fig.update_geos(showframe=False, showcoastlines=False, bgcolor="rgba(0,0,0,0)",
                            projection_type="natural earth")
            plot(fig, key="geo_map", height=480)
            st.caption("Names that are not countries (Channel Islands, European Community, Unspecified) do not appear on the map. Use the Bars view for all.")
        else:
            n = st.slider("Countries to show", 5, min(40, max(5, len(g))), min(15, len(g)), key="geo_n")
            top = g.nlargest(n, gm).sort_values(gm)
            fig = px.bar(top, x=gm, y="Country", orientation="h", color=gm, color_continuous_scale="Blues",
                         title=f"Top {n} countries by {gm.lower()}")
            fig.update_coloraxes(showscale=False)
            plot(fig, key="geo_bar", height=max(380, 24 * n + 120))
        uk_share = s.loc[s["Is_UK"], "TotalLineRevenue"].sum() / revenue
        st.info(f"💡 The UK is **{uk_share:.0%}** of revenue in the current view; {s['Country'].nunique()} countries are present.")

    with tab_cust:
        cu = D.cust
        a, b = st.columns(2)
        seg = cu.groupby("Segment").agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum")).reindex(SEGMENT_ORDER).reset_index()
        with a:
            fig = px.pie(seg, names="Segment", values="Customers", hole=0.55, color="Segment",
                         color_discrete_map=SEGMENT_COLORS, title="Customers by segment")
            plot(fig, key="ov_pie_c", height=360)
        with b:
            fig = px.pie(seg, names="Segment", values="Revenue", hole=0.55, color="Segment",
                         color_discrete_map=SEGMENT_COLORS, title="Revenue by segment")
            plot(fig, key="ov_pie_r", height=360)
        st.caption("Segments come from the Customer RFM page and are computed on all registered purchases (not affected by the filters).")


def page_sales():
    D = get_data(); F = current_filters(D)
    hero("📊 Sales Explorer", "When, what and how much customers buy.")
    st.caption("Filters: " + filter_summary(F))
    s = apply_filters(D.sales, F)
    empty_guard(s)

    t1, t2, t3, t4 = st.tabs(["🕒 Time patterns", "🏷️ Products", "🧺 Basket & AOV", "📈 Pareto"])

    with t1:
        a, b = st.columns([1, 2])
        metric = a.radio("Metric", ["Revenue", "Invoices", "Units"], horizontal=True, key="s_t_metric")
        view = b.radio("View", ["Day × Hour heatmap", "By hour", "By day of week", "By month"], horizontal=True, key="s_t_view")
        agg = {"Revenue": ("TotalLineRevenue", "sum"), "Invoices": ("InvoiceNo", "nunique"), "Units": ("Quantity", "sum")}[metric]
        if view == "Day × Hour heatmap":
            g = s.groupby(["Day_Name", "Hour"]).agg(V=agg).reset_index()
            piv = g.pivot(index="Day_Name", columns="Hour", values="V").reindex(
                [d for d in DAY_ORDER if d in set(g["Day_Name"])]).fillna(0)
            fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues",
                            labels=dict(x="Hour", y="", color=metric), title=f"{metric}: day × hour")
            plot(fig, key="s_heat", height=380)
        elif view == "By hour":
            g = s.groupby("Hour").agg(V=agg).reset_index()
            fig = px.bar(g, x="Hour", y="V", labels={"V": metric}, title=f"{metric} by hour", color_discrete_sequence=[BLUE])
            plot(fig, key="s_hour", height=380)
        elif view == "By day of week":
            g = s.groupby("Day_Name").agg(V=agg).reindex([d for d in DAY_ORDER if d in set(s["Day_Name"])]).reset_index()
            fig = px.bar(g, x="Day_Name", y="V", labels={"V": metric, "Day_Name": ""}, title=f"{metric} by day of week",
                         color_discrete_sequence=[BLUE])
            plot(fig, key="s_dow", height=380)
        else:
            g = s.groupby("Month_Start").agg(V=agg).reset_index()
            fig = px.bar(g, x="Month_Start", y="V", labels={"V": metric, "Month_Start": ""}, title=f"{metric} by month",
                         color_discrete_sequence=[BLUE])
            plot(fig, key="s_month", height=380)
        days = set(s["Day_Name"].unique())
        missing = [d for d in DAY_ORDER if d not in days]
        if missing:
            st.caption(f"ℹ️ No sales are recorded on: {', '.join(missing)}.")

    with t2:
        a, b, c = st.columns(3)
        metric = a.radio("Rank by", ["Revenue", "Units", "Invoices"], horizontal=True, key="s_p_metric")
        n = b.slider("Top N", 5, 50, 15, key="s_p_n")
        q = c.text_input("Search product", key="s_p_q", placeholder="e.g. HEART, LANTERN")
        p = s.groupby("StockCode").agg(Revenue=("TotalLineRevenue", "sum"), Units=("Quantity", "sum"),
                                       Invoices=("InvoiceNo", "nunique")).reset_index()
        p["Product"] = p["StockCode"].map(D.label)
        p["Segment"] = p["StockCode"].map(D.prod_segment)
        if q:
            p = p[p["Product"].str.contains(q, case=False, na=False)]
        segs = st.multiselect("Product segments", PRODUCT_SEGMENT_ORDER, default=[], key="s_p_seg",
                              placeholder="All segments")
        if segs:
            p = p[p["Segment"].isin(segs)]
        if p.empty:
            st.info("No products match.")
        else:
            top = p.nlargest(n, metric).sort_values(metric)
            fig = px.bar(top, x=metric, y="Product", orientation="h", color="Segment",
                         color_discrete_map=PRODUCT_SEGMENT_COLORS, title=f"Top {len(top)} products by {metric.lower()}",
                         hover_data=["Revenue", "Units", "Invoices"])
            plot(fig, key="s_top", height=max(380, 26 * len(top) + 140))
            with st.expander("Table & download"):
                table(p.nlargest(200, metric)[["Product", "Segment", "Revenue", "Units", "Invoices"]])
                download_csv(p, "⬇️ Download products (CSV)", "products_view.csv", "dl_s_prod")

    with t3:
        inv = (s.groupby("InvoiceNo").agg(Basket_Units=("Quantity", "sum"), Order_Value=("TotalLineRevenue", "sum"),
                                          CustomerID=("CustomerID", "first"), Country=("Country", "first"),
                                          Is_UK=("Is_UK", "first")).reset_index())
        inv = inv[(inv["Order_Value"] > 0) & (inv["Basket_Units"] > 0)]
        aov, med = inv["Order_Value"].mean(), inv["Order_Value"].median()
        c = st.columns(4)
        kpi(c[0], "Mean order", f"£{aov:,.0f}"); kpi(c[1], "Median order", f"£{med:,.0f}")
        kpi(c[2], "Avg basket (units)", f"{inv['Basket_Units'].mean():,.0f}"); kpi(c[3], "Largest order", fmt_money(inv["Order_Value"].max()))
        a, b = st.columns(2)
        scale = a.radio("Scale", ["Log", "Linear"], horizontal=True, key="s_b_scale")
        lo, hi = float(inv["Order_Value"].min()), float(inv["Order_Value"].max())
        cap = b.slider("Order value range (£)", float(math.floor(lo)), float(math.ceil(hi)),
                       (float(math.floor(lo)), float(math.ceil(hi))), step=1.0, key="s_b_rng")
        iv = inv[inv["Order_Value"].between(cap[0], cap[1])]
        if iv.empty:
            st.info("No invoices in this range.")
        else:
            if scale == "Log":
                iv = iv.assign(LogValue=np.log10(iv["Order_Value"]))
                fig = px.histogram(iv, x="LogValue", nbins=60, color_discrete_sequence=[BLUE],
                                   labels={"LogValue": "Order value (log scale)"}, title="Distribution of order values (log scale)")
                ticks = [v for v in [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
                         if iv["Order_Value"].min() / 2 <= v <= iv["Order_Value"].max() * 2]
                fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[f"{v:,}" for v in ticks])
                fig.add_vline(x=np.log10(med), line_dash="dot", line_color=ORANGE, annotation_text=f"Median £{med:,.0f}", annotation_position="top left")
                fig.add_vline(x=np.log10(aov), line_dash="dash", line_color=GREEN, annotation_text=f"Mean £{aov:,.0f}", annotation_position="top right")
            else:
                fig = px.histogram(iv, x="Order_Value", nbins=60, color_discrete_sequence=[BLUE], title="Distribution of order values")
                fig.add_vline(x=med, line_dash="dot", line_color=ORANGE, annotation_text=f"Median £{med:,.0f}", annotation_position="top left")
                fig.add_vline(x=aov, line_dash="dash", line_color=GREEN, annotation_text=f"Mean £{aov:,.0f}", annotation_position="top right")
            fig.update_yaxes(title_text="Invoices")
            plot(fig, key="s_aov", height=400)
            fig = px.scatter(iv, x="Basket_Units", y="Order_Value", color="Is_UK", opacity=0.5,
                             log_x=scale == "Log", log_y=scale == "Log", hover_data=["InvoiceNo", "Country"],
                             color_discrete_map={True: BLUE, False: ORANGE},
                             labels={"Basket_Units": "Units per invoice", "Order_Value": "Order value", "Is_UK": "UK"},
                             title="Basket size vs order value")
            plot(fig, key="s_basket", height=430)
            st.caption("The mean is higher than the median: a few large (probably wholesale) invoices pull it up, so use the median and the log view.")

    with t4:
        p = s.groupby("StockCode")["TotalLineRevenue"].sum().sort_values(ascending=False).reset_index()
        p["Cum_Revenue_%"] = p["TotalLineRevenue"].cumsum() / p["TotalLineRevenue"].sum() * 100
        p["Cum_Products_%"] = (np.arange(1, len(p) + 1)) / len(p) * 100
        thr = st.slider("Revenue threshold (%)", 50, 95, 80, key="s_par")
        k = int((p["Cum_Revenue_%"] < thr).sum()) + 1
        fig = go.Figure(go.Scatter(x=p["Cum_Products_%"], y=p["Cum_Revenue_%"], mode="lines", fill="tozeroy",
                                   line=dict(color=BLUE), name="Cumulative revenue"))
        fig.add_hline(y=thr, line_dash="dash", line_color=ORANGE)
        fig.add_vline(x=k / len(p) * 100, line_dash="dash", line_color=ORANGE)
        fig.update_layout(title="Pareto: share of products vs share of revenue", xaxis_title="% of products (best first)",
                          yaxis_title="% of revenue")
        plot(fig, key="s_pareto", height=420)
        st.info(f"💡 **{k:,} products ({k / len(p):.1%})** generate {thr}% of the revenue.")


def page_returns():
    D = get_data(); F = current_filters(D)
    hero("↩️ Returns", "How much comes back, when, and from where.")
    st.caption("Filters: " + filter_summary(F))
    s = apply_filters(D.sales, F); r = apply_filters(D.returns, F)
    empty_guard(r, "No returns for the current filters.")
    revenue = s["TotalLineRevenue"].sum()
    c = st.columns(4)
    kpi(c[0], "Returned value", fmt_money(r["Return_Value"].sum()))
    kpi(c[1], "Return lines", f"{len(r):,}")
    kpi(c[2], "Return rate (value)", f"{r['Return_Value'].sum() / revenue * 100:.1f}%" if revenue else "–")
    kpi(c[3], "Customers who returned", f"{r['CustomerID'].nunique():,}")

    cutoff = r["Return_Value"].quantile(0.999)
    extreme_share = r.loc[r["Return_Value"] > cutoff, "Return_Value"].sum() / r["Return_Value"].sum()
    excl = st.toggle(f"Exclude extreme lines (top 0.1 %, > £{cutoff:,.0f}; they carry {extreme_share:.0%} of return value)", value=False, key="r_excl")
    rr = r[r["Return_Value"] <= cutoff] if excl else r

    t1, t2, t3, t4 = st.tabs(["📅 Over time", "📦 Products", "📉 Size of returns", "👤 Customers"])
    with t1:
        ms = s.groupby("Month_Start")["TotalLineRevenue"].sum().rename("Sales")
        mr = rr.groupby("Month_Start")["Return_Value"].sum().rename("Returns")
        m = pd.concat([ms, mr], axis=1).fillna(0).reset_index()
        m["Rate_%"] = np.where(m["Sales"] > 0, m["Returns"] / m["Sales"] * 100, np.nan)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=m["Month_Start"], y=m["Sales"], name="Sales", marker_color=BLUE, opacity=0.8)
        fig.add_bar(x=m["Month_Start"], y=m["Returns"], name="Returns", marker_color=ORANGE)
        fig.add_trace(go.Scatter(x=m["Month_Start"], y=m["Rate_%"], name="Return rate %", mode="lines+markers",
                                 line=dict(color=PURPLE)), secondary_y=True)
        fig.update_layout(barmode="group", title="Monthly sales vs returns (and return rate)")
        fig.update_yaxes(title_text="Value", secondary_y=False); fig.update_yaxes(title_text="Return rate %", secondary_y=True)
        plot(fig, key="r_time", height=430)
    with t2:
        a, b, c2 = st.columns(3)
        metric = a.radio("Rank by", ["Return value", "Return units", "Return rate %"], horizontal=True, key="r_p_m")
        n = b.slider("Top N", 5, 40, 15, key="r_p_n")
        min_units = c2.slider("Min units sold (for rate)", 10, 1000, 100, step=10, key="r_p_min")
        pr = rr.groupby("StockCode").agg(Return_Value=("Return_Value", "sum"), Return_Units=("Return_Units", "sum"),
                                        Return_Lines=("InvoiceNo", "size")).reset_index()
        ps = s.groupby("StockCode").agg(Sales_Value=("TotalLineRevenue", "sum"), Sales_Units=("Quantity", "sum")).reset_index()
        pr = pr.merge(ps, on="StockCode", how="left").fillna({"Sales_Value": 0, "Sales_Units": 0})
        pr["Return rate %"] = np.where(pr["Sales_Value"] > 0, pr["Return_Value"] / pr["Sales_Value"] * 100, np.nan)
        pr["Product"] = pr["StockCode"].map(D.label)
        col = {"Return value": "Return_Value", "Return units": "Return_Units", "Return rate %": "Return rate %"}[metric]
        base = pr[(pr["Sales_Units"] >= min_units) & (pr["Return_Lines"] >= 3)] if col == "Return rate %" else pr
        top = base.nlargest(n, col).sort_values(col)
        if top.empty:
            st.info("No products meet the thresholds.")
        else:
            fig = px.bar(top, x=col, y="Product", orientation="h", color_discrete_sequence=[ORANGE],
                         hover_data=["Return_Value", "Return_Units", "Sales_Value"], title=f"Top {len(top)} products by {metric.lower()}")
            plot(fig, key="r_prod", height=max(380, 26 * len(top) + 140))
    with t3:
        scale = st.radio("Scale", ["Log10", "Linear"], horizontal=True, key="r_d_scale")
        d = rr[rr["Return_Value"] > 0]
        if scale == "Log10":
            d = d.assign(V=np.log10(d["Return_Value"]))
            fig = px.histogram(d, x="V", nbins=50, marginal="box", color_discrete_sequence=[ORANGE],
                               labels={"V": "log10(return line value)"}, title="Distribution of return-line value (log10)")
        else:
            fig = px.histogram(d, x="Return_Value", nbins=60, marginal="box", color_discrete_sequence=[ORANGE],
                               title="Distribution of return-line value")
        plot(fig, key="r_dist", height=460)
        top10 = rr.nlargest(10, "Return_Value")[["InvoiceNo", "InvoiceDate", "StockCode", "Description", "Return_Units", "UnitPrice", "Return_Value", "CustomerID"]]
        section("10 biggest return lines")
        table(top10)
    with t4:
        rc = rr.dropna(subset=["CustomerID"]).groupby("CustomerID").agg(Return_Value=("Return_Value", "sum"), Return_Invoices=("InvoiceNo", "nunique")).reset_index()
        rc["CustomerID"] = rc["CustomerID"].astype("int64")
        sc = s.dropna(subset=["CustomerID"]).groupby("CustomerID")["TotalLineRevenue"].sum().rename("Sales_Value").reset_index()
        sc["CustomerID"] = sc["CustomerID"].astype("int64")
        rc = rc.merge(sc, on="CustomerID", how="left").fillna({"Sales_Value": 0})
        rc = rc[(rc["Sales_Value"] > 0) & (rc["Return_Value"] > 0)].copy()
        rc["Return rate %"] = rc["Return_Value"] / rc["Sales_Value"] * 100
        rc["Bucket"] = pd.cut(rc["Return rate %"], [0, 5, 20, 50, np.inf], labels=["<5%", "5-20%", "20-50%", ">50%"], right=False).astype(str)
        buckets = st.multiselect("Return-rate buckets", ["<5%", "5-20%", "20-50%", ">50%"], default=["<5%", "5-20%", "20-50%", ">50%"], key="r_c_b")
        scale = st.radio("Axes", ["Log-log", "Linear"], horizontal=True, key="r_c_scale")
        rc = rc[rc["Bucket"].isin(buckets)]
        if rc.empty:
            st.info("No customers match.")
        else:
            fig = px.scatter(rc, x="Sales_Value", y="Return_Value", color="Bucket", log_x=scale == "Log-log", log_y=scale == "Log-log",
                             opacity=0.7, hover_data=["CustomerID", "Return_Invoices", "Return rate %"],
                             color_discrete_map={"<5%": "#95A5A6", "5-20%": "#F5B041", "20-50%": ORANGE, ">50%": "#C0392B"},
                             title="Customer purchases vs returns (above the dashed line = returned more than bought)")
            lo = float(min(rc["Sales_Value"].min(), rc["Return_Value"].min())); hi = float(max(rc["Sales_Value"].max(), rc["Return_Value"].max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Returns = purchases", line=dict(dash="dash", color="#7F8C8D")))
            plot(fig, key="r_cust", height=500)


def segment_filters(prefix: str, df: pd.DataFrame):
    """Filter widgets of the customer RFM page. Returns the filtered frame."""
    with st.expander("🎛️ Filters", expanded=True):
        a, b, c = st.columns(3)
        seg = a.multiselect("Main segment", SEGMENT_ORDER, default=[], key=f"{prefix}_seg", placeholder="All")
        det = b.multiselect("Sub-segment", DETAILED_ORDER, default=[], key=f"{prefix}_det", placeholder="All")
        ctry = c.multiselect("Customer country", sorted(df["Country"].dropna().unique()), default=[], key=f"{prefix}_ctry", placeholder="All")
        d, e, f = st.columns(3)
        rs = d.slider("Recency score (R)", 1, 5, (1, 5), key=f"{prefix}_r")
        fs = e.slider("Frequency score (F)", 1, 5, (1, 5), key=f"{prefix}_f")
        ms = f.slider("Monetary score (M)", 1, 5, (1, 5), key=f"{prefix}_m")
        g, h = st.columns(2)
        mlo, mhi = float(df["Monetary"].min()), float(df["Monetary"].max())
        mon = g.slider("Total spend (£)", float(math.floor(mlo)), float(math.ceil(mhi)), (float(math.floor(mlo)), float(math.ceil(mhi))), step=1.0, key=f"{prefix}_mon")
        cid = h.text_input("Find customer ID", key=f"{prefix}_cid", placeholder="e.g. 12347")
    out = df
    if seg:
        out = out[out["Segment"].isin(seg)]
    if det:
        out = out[out["Segment_Detailed"].isin(det)]
    if ctry:
        out = out[out["Country"].isin(ctry)]
    out = out[out["R_Score"].between(*rs) & out["F_Score"].between(*fs) & out["M_Score"].between(*ms) & out["Monetary"].between(*mon)]
    if cid.strip().isdigit():
        out = out[out["CustomerID"] == int(cid.strip())]
    return out


def page_customers():
    D = get_data()
    hero("👥 Customer RFM", "Who are the best customers, and who is drifting away?")
    cu = D.cust
    f = segment_filters("c", cu)
    empty_guard(f, "No customers match these filters.")
    c = st.columns(5)
    kpi(c[0], "Customers in view", f"{len(f):,}", f"{len(f) / len(cu):.0%} of all registered customers")
    kpi(c[1], "Revenue share", f"{f['Monetary'].sum() / cu['Monetary'].sum():.0%}")
    kpi(c[2], "Median recency", f"{f['Recency'].median():.0f} days")
    kpi(c[3], "Median frequency", f"{f['Frequency'].median():.0f} invoices")
    kpi(c[4], "Median spend", f"£{f['Monetary'].median():,.0f}")

    t1, t2, t3, t4, t5, t6 = st.tabs(["🧩 Segments", "🔵 Scatter", "📊 Distributions", "🔥 Heatmap", "🧊 3D", "📋 Table & playbook"])
    with t1:
        level = st.radio("Level", ["Main (4)", "Detailed (11)"], horizontal=True, key="c_level")
        col, order, cmap = ("Segment", SEGMENT_ORDER, SEGMENT_COLORS) if level.startswith("Main") else ("Segment_Detailed", DETAILED_ORDER, DETAILED_COLORS)
        g = f.groupby(col).agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"), Avg_Recency=("Recency", "mean"),
                              Avg_Frequency=("Frequency", "mean"), Avg_Monetary=("Monetary", "mean")).reindex([o for o in order if o in set(f[col])]).reset_index()
        g["Customers_%"] = g["Customers"] / g["Customers"].sum() * 100
        g["Revenue_%"] = g["Revenue"] / g["Revenue"].sum() * 100
        share = g.melt(id_vars=col, value_vars=["Customers_%", "Revenue_%"], var_name="Metric", value_name="Share")
        fig = px.bar(share, x=col, y="Share", color="Metric", barmode="group", text_auto=".1f",
                     color_discrete_map={"Customers_%": "#95A5A6", "Revenue_%": BLUE}, category_orders={col: order},
                     title="Share of customers vs share of revenue")
        plot(fig, key="c_share", height=420)
        fig = px.treemap(f, path=[px.Constant("All customers"), "Segment", "Segment_Detailed"], values="Monetary", color="Segment",
                         color_discrete_map={**SEGMENT_COLORS, "(?)": "rgba(0,0,0,0)"}, title="Revenue by segment (treemap)")
        plot(fig, key="c_tree", height=460)
        table(g.round(1))
    with t2:
        opts = ["Recency", "Frequency", "Monetary", "Avg_Order_Value"]
        a, b, c2, d = st.columns(4)
        x = a.selectbox("X axis", opts, index=1, key="c_x"); y = b.selectbox("Y axis", opts, index=2, key="c_y")
        colorby = c2.selectbox("Colour by", ["Segment", "Segment_Detailed", "R_Score", "Country"], key="c_col")
        logs = d.radio("Scale", ["Log-log", "Linear"], horizontal=True, key="c_log")
        fd = f.assign(R_Score=f["R_Score"].astype(str)) if colorby == "R_Score" else f
        kw = {}
        if colorby == "Segment":
            kw = dict(color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER})
        elif colorby == "Segment_Detailed":
            kw = dict(color_discrete_map=DETAILED_COLORS, category_orders={"Segment_Detailed": DETAILED_ORDER})
        if colorby == "Country":
            top_c = fd["Country"].value_counts().head(8).index
            fd = fd.assign(Country=np.where(fd["Country"].isin(top_c), fd["Country"], "Other"))
        fig = px.scatter(fd, x=x, y=y, color=colorby, opacity=0.65, log_x=logs == "Log-log", log_y=logs == "Log-log",
                         hover_data=["CustomerID", "Segment_Detailed", "RFM_Score"], title=f"{y} vs {x}", **kw)
        plot(fig, key="c_scatter", height=540)
    with t3:
        a, b = st.columns(2)
        metric = a.selectbox("Metric", ["Recency", "Frequency", "Monetary", "Avg_Order_Value"], key="c_dm")
        by = b.radio("Split by", ["None", "Segment"], horizontal=True, key="c_ds")
        logv = st.checkbox("Log10 scale (recommended for Frequency / Monetary)", value=metric in ("Frequency", "Monetary", "Avg_Order_Value"), key=f"c_dlog_{metric}")
        d = f[f[metric] > 0].copy()
        xcol = metric
        if logv:
            d["_x"] = np.log10(d[metric]); xcol = "_x"
        fig = px.histogram(d, x=xcol, nbins=50, marginal="box", color=None if by == "None" else "Segment",
                           color_discrete_map=SEGMENT_COLORS, labels={xcol: f"log10({metric})" if logv else metric},
                           color_discrete_sequence=[BLUE], title=f"Distribution of {metric}")
        plot(fig, key="c_hist", height=470)
    with t4:
        a, b = st.columns(2)
        val = a.radio("Cell value", ["Customers", "Median spend", "Revenue"], horizontal=True, key="c_hv")
        pair = b.radio("Axes", ["R × F", "R × M", "F × M"], horizontal=True, key="c_hp")
        ax_x, ax_y = {"R × F": ("F_Score", "R_Score"), "R × M": ("M_Score", "R_Score"), "F × M": ("M_Score", "F_Score")}[pair]
        if val == "Customers":
            piv = f.pivot_table(index=ax_y, columns=ax_x, values="CustomerID", aggfunc="count")
        elif val == "Median spend":
            piv = f.pivot_table(index=ax_y, columns=ax_x, values="Monetary", aggfunc="median")
        else:
            piv = f.pivot_table(index=ax_y, columns=ax_x, values="Monetary", aggfunc="sum")
        piv = piv.reindex(index=range(1, 6), columns=range(1, 6))
        fig = px.imshow(piv, text_auto=",.0f", origin="lower", aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x=ax_x, y=ax_y, color=val), title=f"{val} by {pair}")
        plot(fig, key="c_heat", height=450)
        top = f["RFM_Score"].value_counts().head(15).sort_values().reset_index()
        top.columns = ["RFM_Score", "Customers"]
        fig = px.bar(top, x="Customers", y="RFM_Score", orientation="h", text="Customers", color_discrete_sequence=[BLUE],
                     title="Most common RFM combinations")
        fig.update_yaxes(type="category")
        plot(fig, key="c_combo", height=460)
    with t5:
        d = f.sample(min(len(f), 4000), random_state=0) if len(f) > 4000 else f
        d = d.assign(log10_Frequency=np.log10(d["Frequency"]), log10_Monetary=np.log10(d["Monetary"]))
        fig = px.scatter_3d(d, x="Recency", y="log10_Frequency", z="log10_Monetary", color="Segment",
                            color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
                            hover_data=["CustomerID", "Segment_Detailed"], opacity=0.75, title="Customers in RFM space")
        fig.update_traces(marker_size=3)
        plot(fig, key="c_3d", height=620)
        if len(f) > 4000:
            st.caption("Showing a random sample of 4,000 customers for speed.")
    with t6:
        left, right = st.columns([2, 1])
        with right:
            section("🎯 What to do with each segment")
            for sname in SEGMENT_ORDER:
                st.markdown(f"<span class='pill' style='border:1px solid {SEGMENT_COLORS[sname]}'>{sname}</span> {PLAYBOOK[sname]}", unsafe_allow_html=True)
                st.write("")
        with left:
            cols = ["CustomerID", "Country", "Segment", "Segment_Detailed", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score"]
            cfg = {"Monetary": st.column_config.NumberColumn("Spend (£)", format="%.0f"),
                   "R_Score": st.column_config.ProgressColumn("R", min_value=0, max_value=5, format="%d"),
                   "F_Score": st.column_config.ProgressColumn("F", min_value=0, max_value=5, format="%d"),
                   "M_Score": st.column_config.ProgressColumn("M", min_value=0, max_value=5, format="%d")}
            table(f.sort_values("Monetary", ascending=False)[cols], column_config=cfg)
            download_csv(f[cols], "⬇️ Download customers in view (CSV)", "customers_view.csv", "dl_cust")


def page_products():
    D = get_data()
    hero("📦 Product RFM", "Stars, steady sellers and dead stock.")
    pr = D.prod
    with st.expander("🎛️ Filters", expanded=True):
        a, b, c = st.columns(3)
        seg = a.multiselect("Product segment", PRODUCT_SEGMENT_ORDER, default=[], key="p_seg", placeholder="All")
        q = b.text_input("Search product", key="p_q", placeholder="e.g. LUNCH BAG")
        lo, hi = float(pr["Avg_Price"].min()), float(pr["Avg_Price"].max())
        price = c.slider("Average price (£)", float(math.floor(lo)), float(math.ceil(hi)), (float(math.floor(lo)), float(math.ceil(hi))), step=0.5, key="p_price")
        d, e, f2 = st.columns(3)
        rs = d.slider("Recency score", 1, 5, (1, 5), key="p_r"); fs = e.slider("Frequency score", 1, 5, (1, 5), key="p_f"); ms = f2.slider("Monetary score", 1, 5, (1, 5), key="p_m")
    f = pr
    if seg:
        f = f[f["Segment"].isin(seg)]
    if q:
        f = f[f["Product_Label"].str.contains(q, case=False, na=False)]
    f = f[f["Avg_Price"].between(*price) & f["R_Score"].between(*rs) & f["F_Score"].between(*fs) & f["M_Score"].between(*ms)]
    empty_guard(f, "No products match these filters.")
    c = st.columns(4)
    kpi(c[0], "Products in view", f"{len(f):,}", f"{len(f) / len(pr):.0%} of all products")
    kpi(c[1], "Revenue share", f"{f['Monetary'].sum() / pr['Monetary'].sum():.0%}")
    kpi(c[2], "Units sold", f"{f['Units_Sold'].sum():,.0f}")
    kpi(c[3], "Median price", f"£{f['Avg_Price'].median():,.2f}")

    t1, t2, t3, t4, t5, t6 = st.tabs(["🧩 Segments", "🫧 Bubble", "🌳 Treemap", "⏳ Lifecycle", "🏆 Leaderboards", "🔗 Customers × Products"])
    with t1:
        g = f.groupby("Segment").agg(Products=("StockCode", "count"), Revenue=("Monetary", "sum"), Units=("Units_Sold", "sum"),
                                    Median_Price=("Avg_Price", "median")).reindex([s for s in PRODUCT_SEGMENT_ORDER if s in set(f["Segment"])]).reset_index()
        g["Products_%"] = g["Products"] / g["Products"].sum() * 100; g["Revenue_%"] = g["Revenue"] / g["Revenue"].sum() * 100
        share = g.melt(id_vars="Segment", value_vars=["Products_%", "Revenue_%"], var_name="Metric", value_name="Share")
        fig = px.bar(share, x="Segment", y="Share", color="Metric", barmode="group", text_auto=".1f",
                     color_discrete_map={"Products_%": "#95A5A6", "Revenue_%": PURPLE}, category_orders={"Segment": PRODUCT_SEGMENT_ORDER},
                     title="Share of products vs share of revenue")
        plot(fig, key="p_share", height=430)
        table(g.round(1))
    with t2:
        a, b = st.columns(2)
        sz = a.selectbox("Bubble size", ["Units_Sold", "Unique_Customers", "Monetary"], key="p_bs")
        logs = b.radio("Scale", ["Log-log", "Linear"], horizontal=True, key="p_bl")
        fig = px.scatter(f, x="Frequency", y="Monetary", size=sz, color="Segment", hover_name="Product_Label",
                         hover_data={"Recency": True, "Units_Sold": ":,", "Monetary": ":,.0f"}, color_discrete_map=PRODUCT_SEGMENT_COLORS,
                         category_orders={"Segment": PRODUCT_SEGMENT_ORDER}, log_x=logs == "Log-log", log_y=logs == "Log-log",
                         size_max=32, opacity=0.65, title="Products: frequency vs revenue")
        plot(fig, key="p_bubble", height=560)
    with t3:
        n = st.slider("Top N products by revenue", 20, 400, 150, step=10, key="p_tn")
        top = f.nlargest(n, "Monetary")
        fig = px.treemap(top, path=[px.Constant(f"Top {len(top)} products"), "Segment", "Product_Label"], values="Monetary", color="Segment",
                         color_discrete_map={**PRODUCT_SEGMENT_COLORS, "(?)": "rgba(0,0,0,0)"}, title="Revenue by segment and product")
        plot(fig, key="p_tree", height=620)
    with t4:
        fig = px.scatter(f, x="First_Sale", y="Last_Sale", color="Segment", hover_name="Product_Label", opacity=0.6,
                         color_discrete_map=PRODUCT_SEGMENT_COLORS, category_orders={"Segment": PRODUCT_SEGMENT_ORDER},
                         hover_data={"Frequency": True, "Monetary": ":,.0f"}, title="Product lifecycle: first sale vs last sale")
        plot(fig, key="p_life", height=520)
        st.caption("Products far below the diagonal's top edge stopped selling a long time ago; those on the top edge are still active.")
    with t5:
        a, b, c2 = st.columns(3)
        which = a.selectbox("Segment", PRODUCT_SEGMENT_ORDER, index=0, key="p_lb_seg")
        by = b.selectbox("Sort by", ["Monetary", "Units_Sold", "Frequency", "Recency", "Unique_Customers"], key="p_lb_by")
        n = c2.slider("Rows", 5, 50, 10, key="p_lb_n")
        lb = f[f["Segment"] == which].sort_values(by, ascending=(by == "Recency")).head(n)
        if lb.empty:
            st.info("No products in this segment for the current filters.")
        else:
            table(lb[["StockCode", "Description", "Frequency", "Monetary", "Units_Sold", "Unique_Customers", "Recency", "Avg_Price"]].round(2))
        with st.expander("All products in view"):
            table(f[["StockCode", "Description", "Segment", "Frequency", "Monetary", "Units_Sold", "Recency", "Avg_Price"]].round(2))
            download_csv(f, "⬇️ Download products in view (CSV)", "products_rfm_view.csv", "dl_prod")
    with t6:
        mode = st.radio("Show", ["Share of each customer segment's revenue", "Absolute revenue"], horizontal=True, key="p_x_mode")
        fig = px.bar(D.cross, x="Customer_Segment", y="TotalLineRevenue", color="Product_Segment", color_discrete_map=PRODUCT_SEGMENT_COLORS,
                     category_orders={"Customer_Segment": SEGMENT_ORDER, "Product_Segment": PRODUCT_SEGMENT_ORDER},
                     labels={"TotalLineRevenue": "Revenue", "Customer_Segment": "", "Product_Segment": "Product segment"},
                     title="What each customer segment buys")
        if mode.startswith("Share"):
            fig.update_layout(barnorm="percent", yaxis_title="% of segment revenue")
        plot(fig, key="p_cross", height=470)
        champ = st.selectbox("Top products bought by", SEGMENT_ORDER, key="p_x_seg")
        sc = D.cust.loc[D.cust["Segment"] == champ, "CustomerID"]
        top = (D.sales[D.sales["CustomerID"].isin(sc)].groupby("StockCode")["TotalLineRevenue"].sum().nlargest(10).reset_index())
        top["Product"] = top["StockCode"].map(D.label); top = top.sort_values("TotalLineRevenue")
        fig = px.bar(top, x="TotalLineRevenue", y="Product", orientation="h", color_discrete_sequence=[SEGMENT_COLORS[champ]],
                     labels={"TotalLineRevenue": f"Revenue from {champ}"}, title=f"Top 10 products bought by {champ}")
        plot(fig, key="p_cross_top", height=440)


# ----------------------------- Market basket -----------------------------
def _basket_matrix(tx: pd.DataFrame, label: pd.Series, min_support: float) -> pd.DataFrame:
    items = tx[["InvoiceNo", "StockCode"]].drop_duplicates()
    size = items.groupby("InvoiceNo")["StockCode"].transform("size")
    items = items[size >= 2]
    all_baskets = items["InvoiceNo"].unique()
    counts = items["StockCode"].value_counts()
    keep = counts[counts >= min_support * len(all_baskets)].index
    items = items[items["StockCode"].isin(keep)]
    m = pd.crosstab(items["InvoiceNo"], items["StockCode"]).astype(bool).reindex(all_baskets, fill_value=False)
    m.columns = [label.get(c, str(c)) for c in m.columns]
    return m


@st.cache_data(show_spinner="Mining rules…", max_entries=12)
def mine_rules(algo: str, min_support: float, min_conf: float, min_lift: float, max_len: int, scope: str, F: dict) -> dict:
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    D = get_data()
    tx = apply_filters(D.sales, F)
    if scope == "Guests only":
        tx = tx[tx["CustomerID"].isna()]
    elif scope != "All customers":
        ids = D.cust.loc[D.cust["Segment"] == scope, "CustomerID"]
        tx = tx[tx["CustomerID"].isin(ids)]
    if tx.empty:
        return dict(baskets=0, items=0, itemsets=0, rules=pd.DataFrame())
    basket = _basket_matrix(tx, D.label, min_support)
    if basket.shape[1] < 2:
        return dict(baskets=len(basket), items=basket.shape[1], itemsets=0, rules=pd.DataFrame())
    if algo == "Apriori":
        freq = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=True)
    else:
        freq = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    out = dict(baskets=len(basket), items=basket.shape[1], itemsets=len(freq), rules=pd.DataFrame())
    if freq.empty:
        return out
    try:
        rules = association_rules(freq, num_itemsets=len(basket), metric="confidence", min_threshold=min_conf)
    except TypeError:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return out
    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return out
    rules["Antecedent"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["Consequent"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    rules["Rule"] = rules["Antecedent"] + "  →  " + rules["Consequent"]
    out["rules"] = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return out


def rule_network(rules: pd.DataFrame, top_n: int) -> go.Figure | None:
    r = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()
    if r.empty:
        return None
    r["a"] = r["Antecedent"]; r["c"] = r["Consequent"]
    r["key"] = r.apply(lambda x: frozenset([x["a"], x["c"]]), axis=1)
    r = r.sort_values(["lift", "confidence"], ascending=False).drop_duplicates("key").head(top_n)
    nodes = sorted(set(r["a"]) | set(r["c"]))
    pos = {n: (math.cos(2 * math.pi * i / len(nodes)), math.sin(2 * math.pi * i / len(nodes))) for i, n in enumerate(nodes)}
    deg = pd.concat([r["a"], r["c"]]).value_counts()
    fig = go.Figure()
    lmin, lmax = r["lift"].min(), r["lift"].max()
    for row in r.itertuples():
        w = 1 + 5 * (row.lift - lmin) / (lmax - lmin + 1e-9)
        (x0, y0), (x1, y1) = pos[row.a], pos[row.c]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=w, color="rgba(127,127,127,0.55)"),
                                 hoverinfo="text", text=f"{row.a} ↔ {row.c}<br>lift {row.lift:.1f} · conf {row.confidence:.0%}",
                                 showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=[short(n.split(" - ", 1)[-1], 20) for n in nodes], textposition="top center", hovertext=nodes, hoverinfo="text",
        marker=dict(size=[10 + 4 * deg[n] for n in nodes], color=[deg[n] for n in nodes], colorscale="Blues", line=dict(width=1, color="#fff")),
        showlegend=False))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5]); fig.update_yaxes(visible=False, range=[-1.4, 1.4])
    fig.update_layout(title="Product network (edge width = lift)")
    return fig


def page_basket():
    D = get_data(); F = current_filters(D)
    hero("🛒 Market Basket (Apriori)", "Which products are bought together?")
    st.caption("Filters: " + filter_summary(F))
    try:
        import mlxtend  # noqa: F401
    except ImportError:
        st.error("This page needs `mlxtend`. Run `pip install mlxtend` and restart the app.")
        st.stop()

    with st.form("basket_form"):
        a, b, c = st.columns(3)
        scope = a.selectbox("Baskets of", ["All customers", "Champions", "Loyal", "At Risk", "Lost", "Guests only"], key="b_scope")
        algo = b.radio("Algorithm", ["FP-Growth", "Apriori"], horizontal=True, key="b_algo",
                       help="Same rules, FP-Growth is much faster and lighter on memory. Apriori runs in low-memory mode.")
        max_len = c.radio("Max items per rule", [2, 3], index=1, horizontal=True, key="b_len")
        d, e, f2 = st.columns(3)
        ms = d.slider("Min support", 0.005, 0.10, 0.02, 0.005, format="%.3f", key="b_sup", help="Share of baskets that contain the combination.")
        mc = e.slider("Min confidence", 0.10, 0.95, 0.40, 0.05, key="b_conf", help="Chance of B given A.")
        ml = f2.slider("Min lift", 1.0, 20.0, 1.5, 0.5, key="b_lift", help="> 1 means more than coincidence.")
        submitted = st.form_submit_button("🚀 Mine rules", type="primary")
    params = (algo, float(ms), float(mc), float(ml), int(max_len), scope, F)
    if submitted or "basket_res" not in st.session_state or st.session_state.get("basket_F") != F:
        st.session_state["basket_res"] = mine_rules(*params)
        st.session_state["basket_F"] = F
    res = st.session_state["basket_res"]
    rules = res["rules"]

    c = st.columns(4)
    kpi(c[0], "Baskets (2+ items)", f"{res['baskets']:,}"); kpi(c[1], "Products ≥ support", f"{res['items']:,}")
    kpi(c[2], "Frequent itemsets", f"{res['itemsets']:,}"); kpi(c[3], "Rules", f"{len(rules):,}")
    if rules.empty:
        st.warning("No rules at these thresholds. Lower the support / confidence / lift, or widen the global filters.")
        st.stop()

    t1, t2, t3, t4, t5 = st.tabs(["📜 Rules", "🔵 Support vs confidence", "🏅 Top combinations", "🕸️ Network", "💡 Recommender"])
    with t1:
        a, b, c2 = st.columns(3)
        q = a.text_input("Search product in rules", key="b_q", placeholder="e.g. TEACUP")
        sort = b.selectbox("Sort by", ["lift", "confidence", "support"], key="b_sort")
        n = c2.slider("Rows", 10, 200, 30, key="b_n")
        v = rules[rules["Rule"].str.contains(q, case=False, na=False)] if q else rules
        cols = ["Antecedent", "Consequent", "support", "confidence", "lift"]
        table(v.sort_values(sort, ascending=False)[cols].head(n).round(3),
              column_config={"support": st.column_config.ProgressColumn("support", min_value=0, max_value=float(rules["support"].max()), format="%.3f"),
                             "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f")})
        download_csv(rules[cols], "⬇️ Download all rules (CSV)", "association_rules.csv", "dl_rules")
    with t2:
        fig = px.scatter(rules, x="support", y="confidence", color="lift", size="lift", hover_name="Rule",
                         color_continuous_scale="Viridis", opacity=0.75, title="Rules: support vs confidence (colour = lift)")
        plot(fig, key="b_scatter", height=500)
    with t3:
        k = st.slider("Combinations", 5, 30, 15, key="b_k")
        u = (rules.assign(Items=rules.apply(lambda r: frozenset(r["antecedents"] | r["consequents"]), axis=1))
             .sort_values(["lift", "confidence"], ascending=False).drop_duplicates("Items").head(k).sort_values("lift"))
        fig = px.bar(u, x="lift", y="Rule", orientation="h", color="confidence", color_continuous_scale="Blues",
                     hover_data=["support", "confidence"], title=f"Top {len(u)} combinations by lift (each pair once)")
        fig.update_layout(yaxis_title="", yaxis=dict(automargin=True))
        plot(fig, key="b_top", height=max(420, 30 * len(u) + 140))
    with t4:
        k = st.slider("Rules in the network", 5, 40, 20, key="b_net")
        fig = rule_network(rules, k)
        if fig is None:
            st.info("The network needs single-product rules (one item → one item).")
        else:
            plot(fig, key="b_network", height=620)
    with t5:
        prods = sorted({p for s_ in rules["antecedents"] for p in s_})
        pick = st.selectbox("If a customer buys…", prods, key="b_pick")
        rec = rules[rules["antecedents"].apply(lambda s_: pick in s_)].sort_values("confidence", ascending=False).head(10)
        if rec.empty:
            st.info("No rules for this product.")
        else:
            rec = rec.assign(Recommend=rec["Consequent"]).sort_values("confidence")
            fig = px.bar(rec, x="confidence", y="Recommend", orientation="h", color="lift", color_continuous_scale="Viridis",
                         hover_data=["support", "lift"], title="…suggest these (by confidence)")
            fig.update_layout(yaxis=dict(automargin=True), xaxis_tickformat=".0%")
            plot(fig, key="b_rec", height=max(340, 36 * len(rec) + 140))
    st.caption("Association is not causation. Wholesale invoices inflate support, and many strong rules are just colour variants of the same product.")


def page_about():
    D = get_data()
    hero("ℹ️ Data & Method", "How the numbers are built.")
    s = D.stats
    c = st.columns(4)
    kpi(c[0], "Raw rows", f"{s['raw_rows']:,}"); kpi(c[1], "Product rows", f"{s['product_rows']:,}")
    kpi(c[2], "Clean rows", f"{s['clean_rows']:,}"); kpi(c[3], "Guest share", f"{s['guest_share']:.1%}")

    steps = pd.DataFrame({
        "Step": ["1. Raw data", "2. Real products only", "3. After dropping unrecoverable zero-price lines", "4. Valid sales (Quantity > 0)", "5. Registered customers (RFM input)"],
        "Rows": [s["raw_rows"], s["product_rows"], s["clean_rows"], s["sales_rows"], s["customer_rows"]],
    })
    fig = go.Figure(go.Funnel(y=steps["Step"], x=steps["Rows"], textinfo="value+percent initial",
                              marker=dict(color=["#2E86DE", "#3F9BE8", "#5BAEF0", "#F5B041", "#1B7F4C"])))
    fig.update_layout(title="From raw rows to the rows used in the analysis", margin=dict(l=260))
    plot(fig, key="about_funnel", height=380)

    zero = s["zero_price_rows"]
    if zero:
        fig = go.Figure(go.Waterfall(
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Zero-price lines", "Recovered: one price per description", "Recovered: nearest price per StockCode", "Dropped", "Remaining"],
            y=[zero, -s["recovered_by_description"], -s["recovered_by_nearest"], -s["dropped_zero_price"], 0],
            increasing=dict(marker=dict(color=ORANGE)), decreasing=dict(marker=dict(color=GREEN)), totals=dict(marker=dict(color=BLUE))))
        fig.update_layout(title="Zero unit prices: from problem to resolution", showlegend=False)
        plot(fig, key="about_water", height=380)

    it = pd.DataFrame({"ItemType": list(s["itemtype_counts"].keys()), "Lines": list(s["itemtype_counts"].values())}).sort_values("Lines")
    fig = px.bar(it, x="Lines", y="ItemType", orientation="h", log_x=True, text="Lines", color_discrete_sequence=[BLUE], title="Row classification (log scale)")
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    plot(fig, key="about_types", height=380)

    st.markdown("""
**Decisions (same as the notebook)**
- Shipping, fees, vouchers, adjustments and samples are separated from real products.
- One description per StockCode (the most frequent). Missing prices of 0 are recovered from the same description (if it has a single price) or the nearest price of the same product; the rest are dropped.
- **Guests and duplicate lines are flagged, not deleted.**
- Sales = positive quantity, positive price, not a cancellation. Returns = negative quantity on cancellation invoices.
- **RFM:** quintile scores on ranks (so huge customers do not squash the scale); FM = ⌈(F + M) / 2⌉; 11 sub-segments roll up to 4 main segments.
- **Basket analysis:** basket = invoice, quantity ignored, single-item invoices removed.

**Dark / light mode:** use the toggle at the bottom of the sidebar. The ⋮ menu → Settings → Theme also works.
""")


# =====================================================================================
# 6. THEME + NAVIGATION
# =====================================================================================
CSS = """
<style>
.hero{background:linear-gradient(120deg,#2E86DE 0%,#8E44AD 58%,#E67E22 100%);color:#fff;padding:1.3rem 1.6rem;border-radius:16px;margin-bottom:.8rem;}
.hero h1{margin:0;font-size:1.8rem;color:#fff;padding:0;}
.hero p{margin:.25rem 0 0;opacity:.93;font-size:1.0rem;}
.pill{display:inline-block;padding:.1rem .65rem;border-radius:999px;font-size:.78rem;font-weight:600;margin-right:.4rem;background:rgba(127,127,127,.14);}
[data-testid="stMetricValue"]{font-size:1.55rem;}
.block-container{padding-top:2.2rem;}
</style>
"""


def apply_theme() -> None:
    base = "dark" if st.session_state.get("dark") else "light"
    try:
        st._config.set_option("theme.base", base)
    except Exception:
        pass


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "dark" not in st.session_state:
        try:
            st.session_state["dark"] = st.context.theme.type == "dark"
        except Exception:
            st.session_state["dark"] = False

    D = get_data()
    pages = [
        st.Page(page_overview, title="Overview", icon="🏠", url_path="overview", default=True),
        st.Page(page_sales, title="Sales Explorer", icon="📊", url_path="sales"),
        st.Page(page_returns, title="Returns", icon="↩️", url_path="returns"),
        st.Page(page_customers, title="Customer RFM", icon="👥", url_path="customers"),
        st.Page(page_products, title="Product RFM", icon="📦", url_path="products"),
        st.Page(page_basket, title="Market Basket", icon="🛒", url_path="basket"),
        st.Page(page_about, title="Data & Method", icon="ℹ️", url_path="about"),
    ]
    pg = st.navigation(pages)
    if pg.title in {"Overview", "Sales Explorer", "Returns", "Market Basket"}:
        sidebar_filters(D)
    st.sidebar.divider()
    st.sidebar.toggle("🌙 Dark mode", key="dark", on_change=apply_theme)
    st.sidebar.caption(f"Data: {D.date_min:%d %b %Y} → {D.date_max:%d %b %Y}")
    pg.run()


main()
