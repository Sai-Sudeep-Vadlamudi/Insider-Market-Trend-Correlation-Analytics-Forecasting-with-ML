# app/app.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page config + global styling
# -----------------------------
st.set_page_config(
    page_title="Insider ↔ Market Trend Lab (JPM)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetric"] { background: #0e1117; border: 1px solid #222; padding: 14px 14px 10px 14px; border-radius: 10px; }
[data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.9; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
hr { margin: 0.6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"

# -----------------------------
# Repo-relative paths (Streamlit Cloud compatible)
# -----------------------------
APP_DIR = Path(__file__).resolve().parent      # .../repo/app
REPO_ROOT = APP_DIR.parent                     # .../repo

# Optional: allow overriding data location via environment variable
# On Streamlit Cloud you can set this in app settings (not required).
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(REPO_ROOT))).resolve()

PATH_INS_CLEAN = DATA_ROOT / "cleaned & processed" / "processed" / "jpm_insider_clean.csv"
PATH_PX_CLEAN  = DATA_ROOT / "cleaned & processed" / "processed" / "jpm_prices_clean.csv"

PATH_EVENTS    = DATA_ROOT / "eda_outputs" / "event_study_table.csv"
PATH_INS_SIG   = DATA_ROOT / "eda_outputs" / "insider_signal_with_event_day.csv"
PATH_INS_SUM   = DATA_ROOT / "eda_outputs" / "insider_summary_min5events.csv"
PATH_MASTER    = DATA_ROOT / "eda_outputs" / "master_daily_event_aligned.csv"

PATH_ML_METRICS = DATA_ROOT / "ml_outputs" / "walk_forward_metrics.csv"
PATH_ML_SUMMARY = DATA_ROOT / "ml_outputs" / "walk_forward_summary_mean.csv"

ALL_PATHS = {
    "insider_clean": PATH_INS_CLEAN,
    "prices_clean": PATH_PX_CLEAN,
    "event_study": PATH_EVENTS,
    "insider_signal": PATH_INS_SIG,
    "insider_summary": PATH_INS_SUM,
    "master_daily": PATH_MASTER,
    "ml_metrics": PATH_ML_METRICS,
    "ml_summary": PATH_ML_SUMMARY,
}

# -----------------------------
# Utilities
# -----------------------------
def fmt_money(x):
    if pd.isna(x):
        return "NA"
    x = float(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.2f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.2f}K"
    return f"${x:,.0f}"

def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100*float(x):,.2f}%"

def safe_to_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def must_exist(path: Path):
    if not path.exists():
        st.error("A required file is missing. Make sure it is committed to the GitHub repo at the same path.")
        st.code(str(path))
        st.stop()

@st.cache_data(show_spinner=False)
def load_all():
    for p in ALL_PATHS.values():
        must_exist(p)

    ins_clean  = pd.read_csv(PATH_INS_CLEAN)
    px_clean   = pd.read_csv(PATH_PX_CLEAN)
    events     = pd.read_csv(PATH_EVENTS)
    ins_sig    = pd.read_csv(PATH_INS_SIG)
    ins_sum    = pd.read_csv(PATH_INS_SUM)
    master     = pd.read_csv(PATH_MASTER)
    ml_metrics = pd.read_csv(PATH_ML_METRICS)
    ml_summary = pd.read_csv(PATH_ML_SUMMARY)

    px_clean = safe_to_datetime(px_clean, "date")
    master   = safe_to_datetime(master, "date")

    if "day" in master.columns:
        master["day"] = pd.to_datetime(master["day"], errors="coerce")
    else:
        master["day"] = master["date"].dt.normalize()

    events  = safe_to_datetime(events, "event_day")
    ins_sig = safe_to_datetime(ins_sig, "event_day")

    for c in ["trade_date", "filing_date"]:
        ins_clean = safe_to_datetime(ins_clean, c)

    # normalize event columns
    if "net_side" not in events.columns and "net_value" in events.columns:
        events["net_side"] = np.where(events["net_value"] > 0, "BUY_DOM", "SELL_DOM")
    if "abs_net_value" not in events.columns and "net_value" in events.columns:
        events["abs_net_value"] = events["net_value"].abs()

    return ins_clean, px_clean, events, ins_sig, ins_sum, master, ml_metrics, ml_summary

@st.cache_data(show_spinner=False)
def build_event_hover(ins_sig: pd.DataFrame, max_lines=6) -> pd.DataFrame:
    tmp = ins_sig.copy()
    if "trade_value" not in tmp.columns:
        tmp["trade_value"] = 0.0
    tmp["abs_value"] = tmp["trade_value"].abs()

    def row_text(r):
        name = str(r.get("insider_name", "")).strip()
        title = str(r.get("insider_title", "")).strip()
        ttype = str(r.get("trade_type", "")).strip()
        val = float(r.get("trade_value", 0.0))
        return f"{name} ({title}) {ttype} {fmt_money(val)}"

    out = []
    for d, g in tmp.groupby("event_day"):
        g = g.sort_values("abs_value", ascending=False)
        top = g.head(max_lines)
        lines = [row_text(r) for _, r in top.iterrows()]
        more = len(g) - len(top)
        if more > 0:
            lines.append(f"... +{more} more trade(s)")
        out.append({"event_day": d, "event_label": "<br>".join(lines)})
    return pd.DataFrame(out)

def choose_best_models(ml_summary: pd.DataFrame):
    best = {}
    if not {"task", "model"}.issubset(set(ml_summary.columns)):
        return best

    for task in ml_summary["task"].unique():
        sub = ml_summary[ml_summary["task"] == task].copy()

        if "cls" in str(task):
            if "roc_auc" in sub.columns:
                row = sub.sort_values("roc_auc", ascending=False).head(1)
                best[task] = (row["model"].iloc[0], "roc_auc", float(row["roc_auc"].iloc[0]))
            elif "pr_auc" in sub.columns:
                row = sub.sort_values("pr_auc", ascending=False).head(1)
                best[task] = (row["model"].iloc[0], "pr_auc", float(row["pr_auc"].iloc[0]))
        else:
            if "rmse" in sub.columns:
                row = sub.sort_values("rmse", ascending=True).head(1)
                best[task] = (row["model"].iloc[0], "rmse", float(row["rmse"].iloc[0]))
            elif "mae" in sub.columns:
                row = sub.sort_values("mae", ascending=True).head(1)
                best[task] = (row["model"].iloc[0], "mae", float(row["mae"].iloc[0]))
    return best

def ensure_mag_buckets(events: pd.DataFrame):
    if "abs_net_value" not in events.columns:
        return events, 0, 0

    q95 = events["abs_net_value"].quantile(0.95) if len(events) else 0
    q80 = events["abs_net_value"].quantile(0.80) if len(events) else 0

    def bucket(v):
        if v >= q95:
            return "Top 5%"
        if v >= q80:
            return "5–20%"
        return "Bottom 80%"

    events = events.copy()
    events["mag_bucket"] = events["abs_net_value"].apply(bucket)
    return events, q80, q95

# -----------------------------
# Load
# -----------------------------
ins_clean, px_clean, events, ins_sig, ins_sum, master, ml_metrics, ml_summary = load_all()
events, q80, q95 = ensure_mag_buckets(events)
hover_df = build_event_hover(ins_sig, max_lines=6)
best_models = choose_best_models(ml_summary)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dashboard Controls")

min_date = master["date"].min()
max_date = master["date"].max()
default_start = max(max_date - pd.Timedelta(days=365 * 2), min_date)

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[1])

side_filter = st.sidebar.multiselect("Event side", ["BUY_DOM", "SELL_DOM"], default=["BUY_DOM", "SELL_DOM"])
bucket_filter = st.sidebar.multiselect("Magnitude bucket", ["Top 5%", "5–20%", "Bottom 80%"], default=["Top 5%", "5–20%"])
min_abs_value = st.sidebar.number_input("Min |net value| ($)", min_value=0, value=0, step=1_000_000)
show_case_studies = st.sidebar.checkbox("Show Case Studies", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Start with Top 5% magnitude bucket for the highest-signal days.")

# -----------------------------
# Filter frames
# -----------------------------
master_f = master[(master["date"] >= start_date) & (master["date"] <= end_date)].copy().sort_values("date")

events_f = events[
    (events["event_day"] >= master_f["day"].min()) &
    (events["event_day"] <= master_f["day"].max())
].copy()

events_f = events_f[
    (events_f["net_side"].isin(side_filter)) &
    (events_f["mag_bucket"].isin(bucket_filter)) &
    (events_f["abs_net_value"] >= float(min_abs_value))
].copy()

# -----------------------------
# Header
# -----------------------------
st.title("Insider ↔ Market Trend Lab (JPM)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Date Range", f"{master_f['date'].min().date()} → {master_f['date'].max().date()}")
k2.metric("Trading Days", f"{len(master_f):,}")
k3.metric("Filtered Event Days", f"{events_f['event_day'].nunique():,}")
k4.metric("Filtered Net Value", fmt_money(events_f["net_value"].sum()) if "net_value" in events_f.columns else "NA")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Explorer", "Event Study", "ML Results", "Case Studies", "My Insights"])

# -----------------------------
# Explorer
# -----------------------------
with tabs[0]:
    st.subheader("Explorer: Candlestick + Insider Event Overlay")

    required = ["open", "high", "low", "price", "date", "day"]
    missing = [c for c in required if c not in master_f.columns]
    if missing:
        st.error(f"master_daily_event_aligned.csv is missing columns required for candlestick: {missing}")
        st.stop()

    events_plot = events_f.merge(hover_df, on="event_day", how="left")
    events_plot["event_label"] = events_plot["event_label"].fillna("No insider detail available")

    price_map = master_f.set_index("day")["price"]
    events_plot["marker_y"] = events_plot["event_day"].map(lambda d: price_map.get(d, np.nan))

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=master_f["date"],
        open=master_f["open"],
        high=master_f["high"],
        low=master_f["low"],
        close=master_f["price"],
        name="JPM"
    ))

    if len(events_plot) > 0:
        sizes = 6 + 18 * (events_plot["abs_net_value"] / events_plot["abs_net_value"].max()).fillna(0)
        colors = np.where(events_plot["net_value"] > 0, "#2ca02c", "#d62728")

        hover = (
            "<b>Event day:</b> %{x|%Y-%m-%d}<br>"
            "<b>Side:</b> %{customdata[0]}<br>"
            "<b>Net value:</b> %{customdata[1]}<br>"
            "<b>Trades:</b> %{customdata[2]} (Buy %{customdata[3]}, Sell %{customdata[4]})<br>"
            "<b>Unique insiders:</b> %{customdata[5]}<br>"
            "<b>Bucket:</b> %{customdata[6]}<br>"
            "<br><b>Top insider trades:</b><br>%{customdata[7]}<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=pd.to_datetime(events_plot["event_day"]),
            y=events_plot["marker_y"],
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.80, line=dict(width=1, color="rgba(0,0,0,0.2)")),
            name="Insider events",
            customdata=np.stack([
                events_plot["net_side"].values,
                [fmt_money(v) for v in events_plot["net_value"].values],
                events_plot.get("n_trades", pd.Series([0]*len(events_plot))).values,
                events_plot.get("n_buy", pd.Series([0]*len(events_plot))).values,
                events_plot.get("n_sell", pd.Series([0]*len(events_plot))).values,
                events_plot.get("unique_insiders", pd.Series([0]*len(events_plot))).values,
                events_plot["mag_bucket"].values,
                events_plot["event_label"].values,
            ], axis=1),
            hovertemplate=hover
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=680,
        title="JPM Candlestick with Insider Event Overlay",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Event Study
# -----------------------------
with tabs[1]:
    st.subheader("Event Study: Reaction vs Drift")

    cols_reaction = [c for c in events_f.columns if c.startswith("reaction")]
    cols_drift = [c for c in events_f.columns if c.startswith("drift")]

    if not cols_reaction and not cols_drift:
        st.warning("event_study_table.csv does not contain reaction/drift columns in this export.")
    else:
        show_cols = ["event_day", "net_side", "mag_bucket", "net_value", "n_trades", "unique_insiders"] + cols_reaction + cols_drift
        show_cols = [c for c in show_cols if c in events_f.columns]
        st.dataframe(events_f[show_cols].sort_values("event_day", ascending=False).head(80), use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            if "reaction_prevclose_to_close" in events_f.columns:
                fig = px.box(events_f, x="mag_bucket", y="reaction_prevclose_to_close", color="net_side",
                             points="outliers", template=PLOTLY_TEMPLATE,
                             category_orders={"mag_bucket": ["Top 5%","5–20%","Bottom 80%"]},
                             title="Event-day Reaction (prev close → close)")
                fig.add_hline(y=0, line_dash="dash", opacity=0.4)
                st.plotly_chart(fig, use_container_width=True)

        with cB:
            if "drift_5d" in events_f.columns:
                fig = px.box(events_f, x="mag_bucket", y="drift_5d", color="net_side",
                             points="outliers", template=PLOTLY_TEMPLATE,
                             category_orders={"mag_bucket": ["Top 5%","5–20%","Bottom 80%"]},
                             title="Post-event Drift (5D)")
                fig.add_hline(y=0, line_dash="dash", opacity=0.4)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ML Results
# -----------------------------
with tabs[2]:
    st.subheader("ML Results: Walk-forward Evaluation")

    st.markdown("Mean metrics across folds")
    st.dataframe(ml_summary, use_container_width=True)

    if "task" in ml_metrics.columns:
        task_options = sorted(ml_metrics["task"].unique())
        task = st.selectbox("Task", task_options, index=0)

        subm = ml_metrics[ml_metrics["task"] == task].copy()
        metric_candidates = [c for c in ["roc_auc","pr_auc","brier","rmse","mae","r2","dir_acc","accuracy","f1"] if c in subm.columns]
        metric = st.selectbox("Metric", metric_candidates, index=0) if metric_candidates else None

        if metric:
            fig = px.line(subm.sort_values("fold"), x="fold", y=metric, color="model", markers=True,
                          title=f"{task}: {metric} by fold", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Case Studies
# -----------------------------
with tabs[3]:
    st.subheader("Case Studies")

    if not show_case_studies:
        st.info("Enable case studies in the sidebar.")
    else:
        if len(events_f) == 0:
            st.info("No events under current filters.")
        else:
            topK = events_f.sort_values("abs_net_value", ascending=False).head(20).copy()
            topK["label"] = topK["event_day"].dt.strftime("%Y-%m-%d") + " | " + topK["net_side"].astype(str) + " | " + topK["abs_net_value"].apply(fmt_money)

            choice = st.selectbox("Select an event day (top 20 by magnitude)", topK["label"].tolist(), index=0)
            chosen_day = pd.to_datetime(choice.split("|")[0].strip())

            window_before = st.slider("Days before", 5, 60, 15)
            window_after  = st.slider("Days after", 5, 60, 25)

            day_idx = master_f.reset_index(drop=True)
            idx_map = {d:i for i, d in enumerate(day_idx["day"])}

            if chosen_day in idx_map:
                i = idx_map[chosen_day]
                i0 = max(0, i - window_before)
                i1 = min(len(day_idx)-1, i + window_after)
                sub = day_idx.loc[i0:i1].copy()

                fig = px.line(sub, x="date", y="price", title="Close price around event day", template=PLOTLY_TEMPLATE)
                fig.add_vline(x=day_idx.loc[i, "date"], line_dash="dash", line_width=2)
                st.plotly_chart(fig, use_container_width=True)

            trades = ins_sig[ins_sig["event_day"] == chosen_day].copy()
            if len(trades) > 0 and "trade_value" in trades.columns:
                trades["abs_value"] = trades["trade_value"].abs()
                trades = trades.sort_values("abs_value", ascending=False)

            show_cols = [c for c in ["filing_date","trade_date","insider_name","insider_title","trade_type","qty","trade_price","trade_value"] if c in trades.columns]
            st.dataframe(trades[show_cols].head(40), use_container_width=True)

# -----------------------------
# My Insights (first-person narrative)
# -----------------------------
with tabs[4]:
    st.subheader("My Insights & Business Impact")

    st.write(
        "I built this as an insider-intelligence layer: I align filings to tradable event days, quantify reaction and drift, "
        "and validate forecasting under walk-forward evaluation so the results behave like a production workflow."
    )

    st.markdown("Best models from my walk-forward summary")
    if best_models:
        for task, (model, metric, val) in best_models.items():
            st.write(f"{task}: {model} ({metric} = {val:.3f})")
    else:
        st.write("ML summary table did not include expected columns.")

    st.markdown("Largest event days (executive attention)")
    top_tbl = events.sort_values("abs_net_value", ascending=False).head(10).copy()
    top_tbl["abs_net_value_fmt"] = top_tbl["abs_net_value"].apply(fmt_money)
    top_tbl["net_value_fmt"] = top_tbl["net_value"].apply(fmt_money)

    st.dataframe(
        top_tbl[["event_day","net_side","mag_bucket","abs_net_value_fmt","net_value_fmt","n_trades","unique_insiders"]],
        use_container_width=True
    )
