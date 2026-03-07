Insider-Market Trend Correlation Analytics & Forecasting with ML

Event-Driven Insider Intelligence • SEC Form 4 Signal Engineering • Event Study (Reaction vs Drift) • Walk-Forward ML Validation • Streamlit Decision Dashboard

What this project is :

This project is an end-to-end insider intelligence and market signal pipeline built on JPMorgan. I take insider filing activity (Form-4 style transactions) and historical OHLCV prices, convert them into realistic tradable event days, quantify the stock’s event-day reaction and post-event drift, and then benchmark classification + regression models using walk-forward (rolling) evaluation. The result is not just a notebook—it's a decision-support workflow with a Streamlit dashboard that makes the analysis explorable and explainable.

Why it matters :

Insider activity is publicly available, but it’s rarely packaged in a way that is operational. This project turns filings into a structured signal layer that can support:

Risk monitoring and anomaly detection (high-magnitude sell clusters, unusual activity days)

Research and decision support (reaction vs drift decomposition; magnitude buckets; regime thinking)

Forecasting and triage (walk-forward validated probability/return estimates used for prioritization)

Key design principle

I anchor insider events to information availability rather than hindsight. Filings frequently arrive after market close, so the project uses an event_day definition aligned to when the market can realistically react. This keeps both the event study and the ML evaluation disciplined and production-oriented.

Repository structure :

Insider-Market Trend Co-relation with ML/
├─ app/
│  ├─ app.py
│  └─ App tunnel.ipynb
├─ cleaned & processed/
│  ├─ cleaned/
│  └─ processed/
│     ├─ jpm_insider_clean.csv
│     └─ jpm_prices_clean.csv
├─ eda_outputs/
│  ├─ event_study_table.csv
│  ├─ insider_signal_with_event_day.csv
│  ├─ insider_summary_min5events.csv
│  └─ master_daily_event_aligned.csv
├─ ml_outputs/
│  ├─ walk_forward_metrics.csv
│  └─ walk_forward_summary_mean.csv
├─ Data Quality & Cleaning.ipynb
├─ Exploratory Analysis + Event Study.ipynb
├─ ML (classification + regression) with walk-forward evaluation.ipynb
├─ JPM Insider.xlsx
└─ JPMorgan Stock Price History.xlsx