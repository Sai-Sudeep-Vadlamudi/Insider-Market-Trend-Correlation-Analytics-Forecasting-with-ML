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

<img width="653" height="426" alt="{45CB6218-17E7-4DA1-BE83-17E799905BAC}" src="https://github.com/user-attachments/assets/665b71d4-082a-4819-af7c-7b1ffb93b225" />

├─ Data Quality & Cleaning.ipynb
├─ Exploratory Analysis + Event Study.ipynb
├─ ML (classification + regression) with walk-forward evaluation.ipynb
├─ JPM Insider.xlsx
└─ JPMorgan Stock Price History.xlsx

