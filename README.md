# DRISHTI ದೃಷ್ಟಿ — AI-Driven Crime Analytics & Visualization Platform

Prototype for **KSP Datathon 2026** (Karnataka State Police × Hack2skill).
DRISHTI converts raw FIR-style crime records into actionable, zone-level
intelligence: hotspot maps, ML risk prediction, demand forecasting,
emerging-cluster detection, plain-English querying, and data-driven
patrol allocation — all fully offline-capable.

## Features
| Module | What it does |
|---|---|
| Command Overview | KPIs with period-over-period deltas, weekly volume, crime mix, hour×weekday heat matrix, case-status funnel |
| Hotspot Map | Leaflet heatmap + clustered incident markers over 20 Bengaluru police-station areas |
| Prediction & Forecast | Gradient-boosted risk score (0–100) per area for any shift/day/month, plus a 14-day citywide demand forecast (validation MAE shown in-app) |
| Trends & Patterns | Monthly/seasonal trends, year-over-year comparison, DBSCAN emerging-cluster detection (last 90d vs previous 90d) |
| Ask DRISHTI | Offline natural-language queries — "vehicle theft hotspots last 3 months" → instant map/chart/metric. No API, runs air-gapped |
| Patrol Allocation | Largest-remainder allocation of N units across areas by predicted risk, with priority tiers and CSV deployment order |

## Quickstart
```bash
pip install -r requirements.txt
python generate_data.py        # builds data/crime_records.csv (synthetic)
streamlit run app.py           # (app auto-generates data if missing)
python test_core.py            # smoke tests for ML / NLQ / patrol engines
```

## Architecture
```
generate_data.py   synthetic FIR-style dataset (seed 42, 30k records)
ml_engine.py       HotspotPredictor · DemandForecaster · ClusterDetector
nlq_engine.py      offline keyword/regex NL query parser
patrol.py          largest-remainder patrol allocator
app.py             Streamlit dashboard (6 tabs)
data/              crime_records.csv
```
Stack: Python · scikit-learn · Streamlit · Folium/Leaflet · Plotly.
No external APIs — deployable on standard or air-gapped police IT
infrastructure.

## Deployment

**Option A — Streamlit Community Cloud (recommended, free)**
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → *New app* → pick the repo,
   branch `main`, file `app.py` → **Deploy**.
3. Dependencies install automatically from `requirements.txt`; the app
   generates its dataset on first boot. You get a permanent URL like
   `https://drishti-ksp.streamlit.app` for the submission form.

**Option B — Hugging Face Spaces (free backup)**
Create a Space → SDK *Streamlit* → upload these files (or connect the
GitHub repo). Same zero-config build.

**Option C — On-prem / air-gapped demo**
`pip download -r requirements.txt -d wheels/` on any online machine,
copy the folder + wheels via pen drive, then
`pip install --no-index --find-links wheels -r requirements.txt` and
`streamlit run app.py --server.address 0.0.0.0`. This mirrors real
police-network constraints and is a good live-demo talking point.

## Data & Responsible Use
- The bundled dataset is **fully synthetic** (no real FIR data),
  generated with realistic spatial/temporal patterns for Bengaluru.
  The entire pipeline runs unchanged on a real FIR CSV export with the
  same columns.
- DRISHTI performs **zone-level analytics only** — no individual-level
  prediction, profiling, or personal data anywhere in the system.
- Model cards (features, targets, validation MAE) are shown inside the
  app for transparency.
