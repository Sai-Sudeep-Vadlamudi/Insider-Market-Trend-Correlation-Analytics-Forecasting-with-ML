Insider-Market Trend Correlation Analytics & Forecasting with ML


You can access the Interactive dashboard to explore insider event-days, market reaction vs drift, and walk-forward ML stability results here :
live Demo link : https://insider-market-trend-correlation-analytics-forecasting-with-ml.streamlit.app/


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



This repository implements a complete pipeline for analyzing and operationalizing insider filing activity as an event-driven signal for JPMorgan. The project integrates insider transactions (Form 4–style disclosures) with daily OHLCV data and places heavy emphasis on timing realism: filings are anchored to an actionable event_day rather than naïvely using the trade date, which improves interpretability and avoids hindsight-driven conclusions.

The exploratory phase builds a market micro-narrative around insider events by decomposing returns into two components: reaction on the event day (capturing how the market responds when the filing becomes tradable) and drift over the following sessions (capturing whether effects persist). The analysis includes segmentation by magnitude buckets (Top 5%, 5–20%, Bottom 80%), regime context, and role/insider attribution so the results can be understood both statistically and operationally.

The modeling phase extends the event study into forecasting tasks (classification and regression) evaluated under walk-forward validation with sequential folds. The goal is not to claim deterministic price prediction, but to test whether an insider-informed feature set can improve ranking/triage under realistic out-of-sample evaluation. Results are exported as reproducible artifacts and surfaced through a Streamlit dashboard for interactive exploration, inspection of high-impact event days, and review of model stability across time windows.

Overall, the repository demonstrates full-stack analytics work: raw data cleaning, event alignment, signal engineering, rigorous time-series evaluation, and stakeholder-ready delivery through a lightweight app.



