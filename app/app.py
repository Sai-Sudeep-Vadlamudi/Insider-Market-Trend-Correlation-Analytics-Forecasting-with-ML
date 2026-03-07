
# /content/drive/MyDrive/Insider-Market Trend Co-relation with ML /app/app.py

import os
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

# Lightweight CSS for a more “product” feel
st.markdown("""
<style>
/* tighten page */
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }

/* nicer section titles */
h1, h2, h3 { letter-spacing: -0.02em; }

/* KPI cards spacing */
[data-testid="stMetric"] { background: #0e1117; border: 1px solid #222; padding: 14px 14px 10px 14px; border-radius: 10px; }
[data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.9; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }

/* subtle divider */
hr { margin: 0.6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"

# -----------------------------
# Paths (EXACT USER PATHS)
# -----------------------------
_DRIVE_BASE = "/content/drive/MyDrive/Insider-Market Trend Co-relation with ML "
_REPO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = _DRIVE_BASE if os.path.exists(_DRIVE_BASE) else _REPO_BASE

PATH_INS_CLEAN = BASE + "/cleaned & processed/processed/jpm_insider_clean.csv"
PATH_PX_CLEAN  = BASE + "/cleaned & processed/processed/jpm_prices_clean.csv"
PATH_EVENTS    = BASE + "/eda_outputs/event_study_table.csv"
PATH_INS_SIG   = BASE + "/eda_outputs/insider_signal_with_event_day.csv"
PATH_INS_SUM   = BASE + "/eda_outputs/insider_summary_min5events.csv"
PATH_MASTER    = BASE + "/eda_outputs/master_daily_event_aligned.csv"
PATH_ML_METRICS = BASE + "/ml_outputs/walk_forward_metrics.csv"
PATH_ML_SUMMARY = BASE + "/ml_outputs/walk_forward_summary_mean.csv"

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
def must_exist(p: str):
    if not os.path.exists(p):
        st.error(f"Missing file:\n{p}")
        st.stop()

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

@st.cache_data(show_spinner=False)
def load_all():
    for p in ALL_PATHS.values():
        must_exist(p)

    ins_clean = pd.read_csv(PATH_INS_CLEAN)
    px_clean  = pd.read_csv(PATH_PX_CLEAN)
    events    = pd.read_csv(PATH_EVENTS)
    ins_sig   = pd.read_csv(PATH_INS_SIG)
    ins_sum   = pd.read_csv(PATH_INS_SUM)
    master    = pd.read_csv(PATH_MASTER)
    ml_metrics = pd.read_csv(PATH_ML_METRICS)
    ml_summary = pd.read_csv(PATH_ML_SUMMARY)

    px_clean = safe_to_datetime(px_clean, "date")
    master   = safe_to_datetime(master, "date")
    master["day"] = pd.to_datetime(master["day"], errors="coerce") if "day" in master.columns else master["date"].dt.normalize()

    events   = safe_to_datetime(events, "event_day")
    ins_sig  = safe_to_datetime(ins_sig, "event_day")

    ins_clean = safe_to_datetime(ins_clean, "trade_date")
    ins_clean = safe_to_datetime(ins_clean, "filing_date")

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
        name = str(r.get("insider_name","")).strip()
        title = str(r.get("insider_title","")).strip()
        ttype = str(r.get("trade_type","")).strip()
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
    """
    Picks a ‘best’ model per task using common sense:
      - classification: max roc_auc if present else max pr_auc
      - regression: min rmse if present else min mae
    Returns dict task -> (model, metric_name, metric_value)
    """
    best = {}
    if not {"task","model"}.issubset(set(ml_summary.columns)):
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
        if v >= q95: return "Top 5%"
        if v >= q80: return "5–20%"
        return "Bottom 80%"

    events = events.copy()
    events["mag_bucket"] = events["abs_net_value"].apply(bucket)
    return events, q80, q95

def event_kpis(events_f: pd.DataFrame):
    out = {}
    if "net_value" in events_f.columns:
        out["net_value_sum"] = float(events_f["net_value"].sum())
        out["net_value_med"] = float(events_f["net_value"].median())
    if "event_day" in events_f.columns:
        out["event_days"] = int(events_f["event_day"].nunique())
    if "n_trades" in events_f.columns:
        out["trades_sum"] = int(events_f["n_trades"].sum())
    if "unique_insiders" in events_f.columns:
        out["insiders_med"] = float(events_f["unique_insiders"].median())
    return out


# -----------------------------
# Load
# -----------------------------
ins_clean, px_clean, events, ins_sig, ins_sum, master, ml_metrics, ml_summary = load_all()
events, q80, q95 = ensure_mag_buckets(events)
hover_df = build_event_hover(ins_sig, max_lines=6)

# -----------------------------
# Sidebar (story-friendly)
# -----------------------------
st.sidebar.title("Dashboard Controls")

min_date = master["date"].min()
max_date = master["date"].max()

default_start = max(max_date - pd.Timedelta(days=365*2), min_date)

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)
start_date = pd.Timestamp(date_range[0])
end_date   = pd.Timestamp(date_range[1])

show_only_last2y = st.sidebar.checkbox("Force last 2 years", value=False)
if show_only_last2y:
    start_date = max(max_date - pd.Timedelta(days=365*2), min_date)

side_filter = st.sidebar.multiselect(
    "Event side",
    options=["BUY_DOM","SELL_DOM"],
    default=["BUY_DOM","SELL_DOM"]
)

bucket_filter = st.sidebar.multiselect(
    "Magnitude bucket",
    options=["Top 5%","5–20%","Bottom 80%"],
    default=["Top 5%","5–20%","Bottom 80%"]
)

min_abs_value = st.sidebar.number_input(
    "Min |net value| ($)",
    min_value=0,
    value=0,
    step=1_000_000
)

show_case_studies = st.sidebar.checkbox("Show Case Studies panel", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Start with Top 5% bucket to see meaningful event days.")

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
# Header + Story summary
# -----------------------------
st.title("Insider ↔ Market Trend Lab (JPM)")
best_models = choose_best_models(ml_summary)

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Date Range", f"{master_f['date'].min().date()} → {master_f['date'].max().date()}")
k2.metric("Trading Days", f"{len(master_f):,}")
k3.metric("Filtered Event Days", f"{events_f['event_day'].nunique():,}")
k4.metric("Filtered Net Value", fmt_money(events_f["net_value"].sum()) if "net_value" in events_f.columns else "NA")

st.markdown("---")

# Story card
st.subheader("Story Summary")
left, right = st.columns([1.4, 1.0])

with left:
    st.markdown(
        """
**What this dashboard shows**
- **Event day** = the first trading day where a Form 4 filing is assumed “tradable” (your filing-time rule).
- We compare **event-day reaction** and **post-event drift**, and then test whether ML can forecast forward returns using **walk-forward** evaluation.
        """
    )
    # Compute quick story facts
    if len(events_f) > 0:
        tmp = events_f.copy()
        story_lines = []
        if "reaction_prevclose_to_close" in tmp.columns:
            buy = tmp[tmp["net_side"]=="BUY_DOM"]["reaction_prevclose_to_close"].dropna()
            sell = tmp[tmp["net_side"]=="SELL_DOM"]["reaction_prevclose_to_close"].dropna()
            if len(buy) and len(sell):
                story_lines.append(f"- Median **event-day reaction** (prev close → close): BUY_DOM {fmt_pct(buy.median())} vs SELL_DOM {fmt_pct(sell.median())}")
        if "drift_5d" in tmp.columns:
            buy = tmp[tmp["net_side"]=="BUY_DOM"]["drift_5d"].dropna()
            sell = tmp[tmp["net_side"]=="SELL_DOM"]["drift_5d"].dropna()
            if len(buy) and len(sell):
                story_lines.append(f"- Median **5D post-event drift**: BUY_DOM {fmt_pct(buy.median())} vs SELL_DOM {fmt_pct(sell.median())}")
        if story_lines:
            st.markdown("**Fast facts (filtered selection):**\n" + "\n".join(story_lines))
        else:
            st.info("My event study table doesn’t include reaction/drift columns in this export.")
    else:
        st.info("No events in current filters. Try lowering the threshold or selecting more buckets.")

with right:
    st.markdown("**Best ML models (from your walk-forward summary):**")
    if best_models:
        for task, (model, metric, val) in best_models.items():
            st.write(f"- `{task}` → **{model}** ({metric} = {val:.4f})")
    else:
        st.write("ML summary missing `task/model` columns or metric columns.")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Explorer", "Event Study", "ML Results", "Case Studies", "My Insights"])

# -----------------------------
# Tab: Explorer
# -----------------------------
with tabs[0]:
    st.subheader("Explorer: Candlestick + Insider Event Overlay")

    # Required OHLC columns
    required = ["open","high","low","price","date","day"]
    missing = [c for c in required if c not in master_f.columns]
    if missing:
        st.error(f"master_daily_event_aligned.csv is missing columns required for candlestick: {missing}")
        st.stop()

    # Merge hover labels
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
        name="JPM",
        increasing_line_color="#1f77b4",
        decreasing_line_color="#7f7f7f",
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
        title="JPM Candlestick with Insider Event Overlay (hover tells the story)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=-0.15,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Small “insight callout”
    st.info("Use the sidebar to filter to **Top 5%** events and hover markers to see *which insiders* drove each event day.")

# -----------------------------
# Tab: Event Study
# -----------------------------
with tabs[1]:
    st.subheader("Event Study: Reaction vs Drift")

    cols_reaction = [c for c in events_f.columns if c.startswith("reaction")]
    cols_drift = [c for c in events_f.columns if c.startswith("drift")]

    if not cols_reaction and not cols_drift:
        st.warning("Your event_study_table.csv export does not contain reaction/drift columns. Re-export from your EDA notebook if needed.")
    else:
        # Summary table
        show_cols = ["event_day","net_side","mag_bucket","net_value","n_trades","unique_insiders"] + cols_reaction + cols_drift
        show_cols = [c for c in show_cols if c in events_f.columns]
        st.dataframe(events_f[show_cols].sort_values("event_day", ascending=False).head(80), use_container_width=True)

        # Professional visuals: boxplots & distributions
        cA, cB = st.columns(2)

        with cA:
            if "reaction_prevclose_to_close" in events_f.columns:
                fig = px.box(
                    events_f,
                    x="mag_bucket",
                    y="reaction_prevclose_to_close",
                    color="net_side",
                    points="outliers",
                    title="Event-day Reaction (prev close → close) by Magnitude Bucket",
                    category_orders={"mag_bucket":["Top 5%","5–20%","Bottom 80%"]},
                    template=PLOTLY_TEMPLATE
                )
                fig.add_hline(y=0, line_dash="dash", line_width=1, opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            elif "reaction_open_to_close" in events_f.columns:
                fig = px.box(
                    events_f,
                    x="mag_bucket",
                    y="reaction_open_to_close",
                    color="net_side",
                    points="outliers",
                    title="Event-day Intraday Reaction (open → close) by Magnitude Bucket",
                    category_orders={"mag_bucket":["Top 5%","5–20%","Bottom 80%"]},
                    template=PLOTLY_TEMPLATE
                )
                fig.add_hline(y=0, line_dash="dash", line_width=1, opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reaction columns found in your filtered events.")

        with cB:
            if "drift_5d" in events_f.columns:
                fig = px.box(
                    events_f,
                    x="mag_bucket",
                    y="drift_5d",
                    color="net_side",
                    points="outliers",
                    title="Post-event Drift (5D) by Magnitude Bucket",
                    category_orders={"mag_bucket":["Top 5%","5–20%","Bottom 80%"]},
                    template=PLOTLY_TEMPLATE
                )
                fig.add_hline(y=0, line_dash="dash", line_width=1, opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No drift_5d column found in your filtered events.")

        st.markdown("### Distribution view")
        if "drift_5d" in events_f.columns:
            fig = px.histogram(
                events_f,
                x="drift_5d",
                color="net_side",
                nbins=50,
                barmode="overlay",
                title="Distribution: 5-day post-event drift",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig, use_container_width=True)

        # Add sample size story
        counts = events_f["net_side"].value_counts().to_dict() if "net_side" in events_f.columns else {}
        st.caption(f"Sample sizes under current filters: {counts}")

# -----------------------------
# Tab: ML Results
# -----------------------------
with tabs[2]:
    st.subheader("ML Results: Walk-forward Evaluation (stability > one-off score)")

    st.markdown("**Mean metrics across folds** (your exported summary)")
    st.dataframe(ml_summary, use_container_width=True)

    if "task" not in ml_metrics.columns:
        st.warning("walk_forward_metrics.csv missing `task` column.")
    else:
        task_options = sorted(ml_metrics["task"].unique())
        task = st.selectbox("Task", task_options, index=0)

        subm = ml_metrics[ml_metrics["task"] == task].copy()
        metric_candidates = [c for c in ["roc_auc","pr_auc","brier","rmse","mae","r2","dir_acc","accuracy","f1"] if c in subm.columns]
        metric = st.selectbox("Metric", metric_candidates, index=0) if metric_candidates else None

        if metric:
            fig = px.line(
                subm.sort_values("fold"),
                x="fold",
                y=metric,
                color="model",
                markers=True,
                title=f"{task}: {metric} by fold (walk-forward stability)",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig, use_container_width=True)

            # A story callout on stability
            if metric in ["roc_auc","pr_auc","rmse"]:
                spread = subm.groupby("model")[metric].agg(["mean","std","min","max"]).reset_index()
                st.markdown("**Stability snapshot:**")
                st.dataframe(spread, use_container_width=True)

# -----------------------------
# Tab: Case Studies
# -----------------------------
with tabs[3]:
    st.subheader("Case Studies: What happened around the biggest event days?")

    if not show_case_studies:
        st.info("Enable “Show Case Studies panel” in the sidebar.")
    else:
        if len(events_f) == 0:
            st.info("No events under current filters. Try Top 5% bucket or lower the threshold.")
        else:
            # choose event
            topK = events_f.sort_values("abs_net_value", ascending=False).head(20).copy()
            topK["label"] = topK["event_day"].dt.strftime("%Y-%m-%d") + " | " + topK["net_side"].astype(str) + " | " + topK["abs_net_value"].apply(fmt_money)

            choice = st.selectbox("Select an event day (top 20 by magnitude)", topK["label"].tolist(), index=0)
            chosen_day = pd.to_datetime(choice.split("|")[0].strip())

            st.markdown("#### Price window around event day")
            window_before = st.slider("Days before", 5, 60, 15)
            window_after  = st.slider("Days after", 5, 60, 25)

            # build window slice
            day_idx = master_f.reset_index(drop=True)
            idx_map = {d:i for i,d in enumerate(day_idx["day"])}
            if chosen_day not in idx_map:
                st.warning("Selected event day is not in the current filtered master date range.")
            else:
                i = idx_map[chosen_day]
                i0 = max(0, i - window_before)
                i1 = min(len(day_idx)-1, i + window_after)
                sub = day_idx.loc[i0:i1].copy()

                fig = px.line(sub, x="date", y="price", title="Close price around event day", template=PLOTLY_TEMPLATE)
                fig.add_vline(x=day_idx.loc[i, "date"], line_dash="dash", line_width=2)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Insider trades contributing to this event day")
            trades = ins_sig[ins_sig["event_day"] == chosen_day].copy()
            if len(trades) == 0:
                st.info("No insider_signal rows found for this event day (check your insider_signal_with_event_day.csv export).")
            else:
                # show top trades
                if "trade_value" in trades.columns:
                    trades["abs_value"] = trades["trade_value"].abs()
                    trades = trades.sort_values("abs_value", ascending=False)
                show_cols = [c for c in ["filing_date","trade_date","insider_name","insider_title","trade_type","qty","trade_price","trade_value"] if c in trades.columns]
                st.dataframe(trades[show_cols].head(40), use_container_width=True)

# -----------------------------
# Tab: Summary & Insights  
# -----------------------------
with tabs[4]:
    st.subheader("Insights & Business Impact")

    st.markdown(
        """
At a high level, I built an insider-intelligence decision layer for JPM.

I started with two real sources of information: insider filing activity and daily market price behavior. My goal was to answer a simple business question in a disciplined way: when meaningful insider activity becomes public, does the market react immediately, and does it continue to drift afterward? Then I tested whether the patterns I found could be operationalized into a forecasting signal that holds up under out-of-sample evaluation.

What makes this useful is that the output is not a static report. It is an end-to-end system: it turns raw disclosures into tradable event days, quantifies what the market did around those days, and then benchmarks predictive models using a walk-forward method that mimics how a model would behave in production.
        """
    )

    # -----------------------------
    # Insight cards (high-level)
    # -----------------------------
    a, b, c, d = st.columns(4)

    a.metric("Coverage (Trading Days)", f"{len(master):,}")
    b.metric("Signal Event Days", f"{events['event_day'].nunique():,}")

    buy_n = int((events["net_side"] == "BUY_DOM").sum()) if "net_side" in events.columns else 0
    sell_n = int((events["net_side"] == "SELL_DOM").sum()) if "net_side" in events.columns else 0
    c.metric("BUY_DOM vs SELL_DOM", f"{buy_n} vs {sell_n}")

    if "abs_net_value" in events.columns:
        d.metric("Largest Event Magnitude", fmt_money(events["abs_net_value"].max()))
    else:
        d.metric("Largest Event Magnitude", "NA")

    st.markdown("---")

    st.markdown("### Timing realism")

    if "filing_date" in ins_clean.columns:
        filing_minutes = ins_clean["filing_date"].dt.hour * 60 + ins_clean["filing_date"].dt.minute
        after_close = float((filing_minutes >= 16 * 60).mean())

        st.markdown(
            f"""
One of the most important design choices I made was aligning the analysis to how the market actually learns information.

In my dataset, about {after_close*100:.1f}% of filings arrive after market close. That means the market cannot realistically respond on the original trade date in many cases, because the public signal does not exist yet. So instead of analyzing trade dates naïvely, I created an event_day that reflects when the filing becomes actionable. This makes every downstream result more credible because the event study and the models are anchored to information availability rather than hindsight.
            """
        )
    else:
        st.info("filing_date was not found in your insider_clean file, so timing insights cannot be computed here.")

    st.markdown("---")

    st.markdown("### Reaction versus drift")

    if ("reaction_prevclose_to_close" in events.columns) or ("drift_5d" in events.columns):
        st.markdown(
            """
I separated the market response into two phases because it creates a clear business interpretation.

The first phase is the event-day reaction, which captures the immediate move when the filing becomes tradable. The second phase is post-event drift, which measures whether price behavior continues over the following days. This separation helps distinguish whether the market is digesting the information quickly or whether it plays out over time.
            """
        )

        if "reaction_prevclose_to_close" in events.columns:
            buy = events[events["net_side"]=="BUY_DOM"]["reaction_prevclose_to_close"].dropna()
            sell = events[events["net_side"]=="SELL_DOM"]["reaction_prevclose_to_close"].dropna()

            if len(buy) > 0 and len(sell) > 0:
                st.markdown(
                    f"""
On the event day itself, the median move from the previous close to the close differs across event types.

BUY_DOM median event-day move: {fmt_pct(buy.median())}  
SELL_DOM median event-day move: {fmt_pct(sell.median())}

The business takeaway is straightforward. Insider-linked events are not all treated equally by the market, and the immediate response varies with the underlying direction and composition of the event.
                    """
                )

        if "drift_5d" in events.columns:
            buy_d = events[events["net_side"]=="BUY_DOM"]["drift_5d"].dropna()
            sell_d = events[events["net_side"]=="SELL_DOM"]["drift_5d"].dropna()
            if len(buy_d) > 0 and len(sell_d) > 0:
                st.markdown(
                    f"""
After the event day, I also measure follow-through to understand whether the market continues moving beyond day zero.

BUY_DOM median 5-day drift: {fmt_pct(buy_d.median())}  
SELL_DOM median 5-day drift: {fmt_pct(sell_d.median())}

I do not interpret this as a promise of profit. I interpret it as evidence that these event days are information-dense and worth monitoring because the response can extend beyond a single session.
                    """
                )

    else:
        st.info("Your event study export doesn’t contain reaction/drift columns. Re-export from EDA if needed.")

    st.markdown("---")

    st.markdown("### Where the signal concentrates")

    if "mag_bucket" in events.columns:
        top5 = events[events["mag_bucket"]=="Top 5%"]

        st.markdown(
            f"""
Not every insider day is equally important, and I designed the analysis to make that explicit.

I bucketed event days into Top 5%, 5–20%, and Bottom 80% based on absolute net insider value. This creates a practical operational advantage: it separates routine activity from days that deserve immediate attention.

In my dataset, there are {len(top5)} days in the Top 5% bucket. Those are the days I would escalate in a monitoring workflow because they concentrate the largest, most decision-relevant events. The dashboard makes those days easy to identify, explore, and explain.
            """
        )
    else:
        st.info("Magnitude buckets not available. Ensure abs_net_value + mag_bucket exist in event table export.")

    st.markdown("---")

    st.markdown("### Forecasting layer")

    if best_models:
        st.markdown(
            """
Once the event-driven feature layer is in place, I test whether the signal holds up out-of-sample using walk-forward evaluation. That method matters because each test window is genuinely in the future relative to the training window, which approximates how forecasting works in production.

I treat the ML layer as a ranking and triage mechanism. The objective is not to predict prices perfectly. The objective is to consistently identify windows that are more likely to be informative so that decision-makers can focus attention where it matters.
            """
        )

        st.markdown("The best-performing models in my walk-forward summary are:")
        for task, (model, metric, val) in best_models.items():
            st.markdown(f"{task}: {model} ({metric} = {val:.3f})")
    else:
        st.info("Could not infer best models — check that ml_summary has task/model and metric columns.")

    st.markdown("---")

    st.markdown("### How this becomes useful inside a business")

    st.markdown(
        """
In practice, I see three concrete use cases.

First, risk and monitoring. This system surfaces unusually large sell-dominant or buy-dominant event days quickly, shows who contributed to the event, and makes it easy to track the market response.

Second, research and decision support. By separating reaction and drift and segmenting by magnitude buckets, the analysis tells a clean story about how the market digests insider information and where the strongest signals concentrate.

Third, forecasting and alerting. The walk-forward ML evaluation provides a disciplined way to convert the event signals into a repeatable, production-style indicator that can rank and prioritize attention rather than relying on intuition or ad-hoc inspection.
        """
    )

    st.markdown("---")

    st.markdown("### The event days that matter most")

    top_tbl = events.copy()
    if "abs_net_value" in top_tbl.columns:
        top_tbl = top_tbl.sort_values("abs_net_value", ascending=False).head(10)

    if len(top_tbl) > 0:
        if "abs_net_value" in top_tbl.columns:
            top_tbl["abs_net_value"] = top_tbl["abs_net_value"].apply(fmt_money)
        if "net_value" in top_tbl.columns:
            top_tbl["net_value"] = top_tbl["net_value"].apply(fmt_money)

        show_cols = [c for c in ["event_day","net_side","mag_bucket","abs_net_value","net_value","n_trades","unique_insiders"] if c in top_tbl.columns]
        st.dataframe(top_tbl[show_cols], use_container_width=True)
    else:
        st.info("Top events table not available (missing abs_net_value).")

    st.markdown("---")

    st.markdown("### Bottom line")

    st.markdown(
        """
What I achieved here is an end-to-end capability: I engineered a realistic event definition from filings, quantified market behavior around those events, and validated forecasting performance using a walk-forward setup. The dashboard then makes the output usable: it highlights which days matter, explains who drove them, and ties the behavior to measurable reaction and drift metrics.

That combination of engineering, analytics, and operational delivery is what makes this project valuable in real analytics, risk, and fintech workflows.
        """
    )
