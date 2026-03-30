

# Insider-Market Trend Correlation Analytics & Forecasting with ML

## Event-Driven Quantitative Intelligence for Insider Activity, Market Response, and Forward Return Forecasting

This repository is a **quantitative data science and market-intelligence project** that operationalizes **insider trading disclosures** into a **timing-aware event-signal framework**, integrates those signals with historical market data, decomposes market behavior into **event-day reaction** and **post-event drift**, and extends the analysis into **walk-forward classification and regression forecasting**.

At its core, this is not a generic “data analytics” exercise. It is a **multi-layered, event-driven quantitative research pipeline** designed to emulate how real-world financial data science work is structured: **raw disclosure ingestion → signal engineering → event alignment → return decomposition → segmentation analysis → out-of-sample forecasting → interactive research delivery**.

The project uses **JPMorgan insider activity** as the research substrate and packages the resulting workflow into a stakeholder-facing **Streamlit decision interface** for inspection, triage, and interpretability.

You can access the Interactive dashboard to explore insider event-days, market reaction vs drift, and walk-forward ML stability results here :
live Demo link : https://insider-market-trend-correlation-analytics-forecasting-with-ml.streamlit.app/

---

## Executive Abstract

Public insider filings are information-rich, but in raw form they are noisy, delayed, difficult to operationalize, and often misinterpreted. Most naïve analyses commit a fundamental methodological error: they anchor the signal to the trade date rather than to the point at which the information becomes **publicly actionable**.

This project addresses that problem by constructing a **timing-realistic `event_day`**, thereby transforming insider activity into a more credible and decision-relevant market signal. From there, the repository builds an end-to-end quantitative workflow that:

- converts **Form 4–style insider transactions** into an actionable event layer
- integrates those events with **historical OHLCV market data**
- measures **immediate market reaction** versus **subsequent drift**
- segments results by **magnitude**, **regime**, and **insider-role context**
- benchmarks **classification and regression models** under **walk-forward evaluation**
- surfaces outputs through an **interactive research and storytelling dashboard**

The resulting system is best understood as a **quantitative research prototype**, **financial analytics platform**, and **decision-support artifact** in one.

---

## High level overview of Exploratory Analytics Application Hosted on Streamlit

### Explorer
![WhatsApp Image 2026-03-09 at 11 03 48 PM (7)](https://github.com/user-attachments/assets/8781bfb1-3720-4883-afa0-abd1ac2dd1fd)


### Event Study
![WhatsApp Image 2026-03-09 at 11 03 48 PM (4)](https://github.com/user-attachments/assets/2bc71a75-deb3-4994-a9be-db9ff928d273)


### ML Results
![WhatsApp Image 2026-03-09 at 11 03 48 PM (5)](https://github.com/user-attachments/assets/ea03de4c-c341-4637-9cce-4943bee21395)


### Case Studies
![WhatsApp Image 2026-03-09 at 11 03 48 PM (6)](https://github.com/user-attachments/assets/46c4c7c7-e241-4113-84b1-1f9a56743e6b)


---

## Quantitative Research Objective

The central research question is:

**Can insider filing activity, once engineered into a timing-disciplined event signal, reveal statistically and operationally meaningful structure in market behavior, and does that structure support forward-looking ranking or forecasting under realistic out-of-sample evaluation?**

This question is approached through three interconnected lenses:

### 1. Signal Interpretation
Does the market react differently to buy-dominant versus sell-dominant insider events?

### 2. Return Decomposition
Are the effects concentrated on the first tradable day, or do they persist as measurable drift over subsequent sessions?

### 3. Forecast Utility
Can insider-informed features improve classification or regression performance in a walk-forward setting where chronology is preserved?

---

## Project Scope

This repository spans the full lifecycle of a quantitative data science workflow:

### Data Layer
- Raw insider disclosure data
- Historical JPM market-price data
- Cleaned and standardized analytical tables
- Persisted intermediate CSV artifacts for downstream reuse

### Signal Engineering Layer
- Insider trade filtering and normalization
- Actionable event construction
- Side-aware event aggregation
- Net-value and trade-intensity characterization

### Event Study Layer
- Event-day reaction analysis
- Post-event drift analysis
- Magnitude-bucket segmentation
- Regime-aware contextualization
- Insider and role attribution

### Forecasting Layer
- Classification and regression tasks
- Sequential, walk-forward evaluation
- Stability-oriented out-of-sample validation
- Reusable metric exports

### Delivery Layer
- Streamlit research dashboard
- Case-study exploration
- Event overlays on price behavior
- Narrative insights for stakeholder interpretation

This breadth is what gives the project its **research depth**, **analytical legitimacy**, and **portfolio grandness**.

---

## High-Level Technical Architecture

```text
Raw Insider Filings (Form 4–style)      Historical JPM OHLCV Market Data
                    │                                   │
                    └──────────────┬────────────────────┘
                                   │
                         Data Quality & Normalization
                                   │
                    Cleaned Analytical Inputs / Reusable CSVs
                                   │
                      Timing-Realistic Event Signal Engineering
                                   │
                  Event-Day Reaction + Multi-Horizon Drift Analysis
                                   │
          Segmentation by Magnitude / Regime / Insider / Organizational Role
                                   │
                  Forecasting Tasks (Classification + Regression)
                                   │
                    Walk-Forward Out-of-Sample Evaluation
                                   │
                Exported Metrics, Research Tables, and Summary Artifacts
                                   │
                   Streamlit Decision Dashboard / Interactive Delivery
```
Key design principle

I anchor insider events to information availability rather than hindsight. Filings frequently arrive after market close, so the project uses an event_day definition aligned to when the market can realistically react. This keeps both the event study and the ML evaluation disciplined and production-oriented.



Repository structure :

<img width="653" height="426" alt="{45CB6218-17E7-4DA1-BE83-17E799905BAC}" src="https://github.com/user-attachments/assets/665b71d4-082a-4819-af7c-7b1ffb93b225" />



This repository implements a complete pipeline for analyzing and operationalizing insider filing activity as an event-driven signal for JPMorgan. The project integrates insider transactions (Form 4–style disclosures) with daily OHLCV data and places heavy emphasis on timing realism: filings are anchored to an actionable event_day rather than naïvely using the trade date, which improves interpretability and avoids hindsight-driven conclusions.

The exploratory phase builds a market micro-narrative around insider events by decomposing returns into two components: reaction on the event day (capturing how the market responds when the filing becomes tradable) and drift over the following sessions (capturing whether effects persist). The analysis includes segmentation by magnitude buckets (Top 5%, 5–20%, Bottom 80%), regime context, and role/insider attribution so the results can be understood both statistically and operationally.

The modeling phase extends the event study into forecasting tasks (classification and regression) evaluated under walk-forward validation with sequential folds. The goal is not to claim deterministic price prediction, but to test whether an insider-informed feature set can improve ranking/triage under realistic out-of-sample evaluation. Results are exported as reproducible artifacts and surfaced through a Streamlit dashboard for interactive exploration, inspection of high-impact event days, and review of model stability across time windows.

Overall, the repository demonstrates full-stack analytics work: raw data cleaning, event alignment, signal engineering, rigorous time-series evaluation, and stakeholder-ready delivery through a lightweight app.



