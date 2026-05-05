# 🏠 San Diego Airbnb — Dynamic Pricing Engine

A fully automated seasonal pricing system for San Diego Airbnb listings, powered by Meta Prophet time series forecasting. Refreshed monthly with the latest Inside Airbnb data — no manual work after deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-san-diego-pricing-engine.streamlit.app)**

---

## Overview

Static pricing leaves money on the table. This project builds an end-to-end dynamic pricing system across 11,000+ San Diego listings, forecasting nightly rates 365 days ahead using seasonality, day-of-week patterns, and local San Diego events.

Key question it answers: *What should an Airbnb host charge on any given night in San Diego — and how much does Comic-Con actually move the market?*

---

## Key Findings

- **40% seasonal swing** — summer avg $250/night vs winter $189/night
- **Peak night: July 22** — $310/night driven by Comic-Con
- **Weekend premium** — 18% higher than weekdays across all room types
- **Entire home median** — $192/night vs $79 for private rooms

---

## Features

- **365-day price forecast** with confidence intervals per room type
- **Event-aware model** — Comic-Con, Thanksgiving, New Year's Eve, Memorial Day, Labor Day, Fourth of July all modeled explicitly
- **Neighbourhood map** — avg listing price by area across San Diego
- **Room type breakdown** — separate forecast curves for entire home, private room, hotel room, shared room
- **Monthly auto-refresh** — GitHub Actions pulls new Inside Airbnb data on the 1st of every month
- **Listing distribution** — price histogram and neighbourhood deep-dive table

---

## Data Sources

| Source | Data | Update Frequency |
|---|---|---|
| [Inside Airbnb](http://insideairbnb.com/get-the-data/) | 11,000+ SD listings with price, reviews, availability | Monthly |
| San Diego event calendar | Comic-Con, local holidays, seasonal patterns | Hardcoded annually |

No API key required — Inside Airbnb data is publicly available.

---

## Project Structure

```
san-diego-pricing-engine/
├── .github/
│   └── workflows/
│       └── refresh.yml          # Monthly cron — fetch → train → commit
├── app/
│   └── dashboard.py             # Streamlit dashboard
├── scripts/
│   ├── fetch_listings.py        # Downloads latest Inside Airbnb SD dump
│   ├── model.py                 # Prophet training + forecast generation
│   └── pipeline.py              # Orchestrates fetch → model → save
├── data/
│   ├── listings_clean.csv       # Cleaned listing snapshot
│   ├── forecast.csv             # Aggregate daily price forecast
│   └── forecast_by_room.csv     # Per-room-type forecast
└── requirements.txt
```

---

## How It Works

```
GitHub Actions (cron: 1st of every month)
        ↓
fetch_listings.py scrapes Inside Airbnb index → downloads latest SD .csv.gz
        ↓
model.py builds historical time series (2022–2025)
Trains Prophet with yearly + weekly seasonality + 6 SD event holidays
Forecasts 730 days ahead → per-room-type scaling
        ↓
Writes forecast.csv, forecast_by_room.csv, listings_clean.csv
Commits → pushes to main
        ↓
Streamlit Community Cloud detects push → auto-redeploys
```

---

## Model Details

| Setting | Value |
|---|---|
| Algorithm | Meta Prophet |
| Seasonality mode | Multiplicative |
| Yearly seasonality | ✅ |
| Weekly seasonality | ✅ |
| Changepoint prior scale | 0.05 |
| Forecast horizon | 730 days |
| Training window | 2022–2025 |

**Events modeled:**

| Event | Price Multiplier |
|---|---|
| Comic-Con (July) | ×1.6 |
| New Year's Eve | ×1.5 |
| Thanksgiving | ×1.4 |
| Memorial Day | ×1.3 |
| Labor Day | ×1.3 |
| Fourth of July | ×1.3 |

**Seasonal multipliers:**

| Season | Multiplier |
|---|---|
| Summer (Jun–Aug) | ×1.35 |
| Spring Break / Holidays (Mar, Dec) | ×1.20 |
| Shoulder (Apr, May, Sep, Oct) | ×1.10 |
| Low season (Jan, Feb, Nov) | ×0.85 |
| Weekend premium | ×1.18 |

---

## Dashboard Sections

**KPI Row** — peak night, lowest night, summer avg, winter avg, seasonal swing %

**12-Month Forecast** — Prophet forecast line with confidence band and Comic-Con annotation

**Average Price by Month** — bar chart highlighting peak and trough months

**Room Type Forecast** — separate price curves for all 4 room types

**Neighbourhood Map** — scatter mapbox of avg listing price by neighbourhood, sized by listing count

**Event Calendar** — peak prices during named SD events

**Price Distribution** — histogram by room type across all 11k listings

**Neighbourhood Table** — searchable breakdown by neighbourhood + room type with avg price, median, count, reviews, availability

---

## Local Setup

### 1. Clone and create environment

```bash
git clone https://github.com/gdiaz38/san-diego-pricing-engine
cd san-diego-pricing-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run pipeline and launch dashboard

```bash
python3 scripts/pipeline.py
streamlit run app/dashboard.py
```

No API keys needed.

---

## Deployment

### GitHub Actions (auto-refresh)

No secrets required. Workflow runs on the 1st of every month, commits updated CSVs, pushes to main.

### Streamlit Community Cloud

1. Connect repo at [share.streamlit.io](https://share.streamlit.io)
2. Main file: `app/dashboard.py`
3. Deploy — no secrets needed

---

## Tech Stack

`Python 3.11` · `Streamlit` · `Plotly` · `Meta Prophet` · `Pandas` · `BeautifulSoup4` · `GitHub Actions` · `Inside Airbnb`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT
