# Infrastructure

> Everything required to run, test, and deploy Offside Market — versions, paths,
> external services, runtime processes, and the tradeoffs behind each choice.

## Runtime requirements

| Layer       | Choice                  | Why                                                                        |
| ----------- | ----------------------- | -------------------------------------------------------------------------- |
| Language    | Python 3.11+            | Type hints (`list[str]`, `X | None`), pattern matching, fast subinterps.   |
| Env mgmt    | `python -m venv .venv`  | Stdlib only, zero contributor onboarding cost.                             |
| Dep mgmt    | `pip` + `requirements.txt` | Hackathon-appropriate; floor versions only, no lockfile churn.          |
| File store  | Apache Parquet via `pyarrow` | Columnar, typed, fast; ~10× smaller than CSV for our shape.           |
| Hot config  | `data/team_name_map.json` + `pipeline/config.py` | JSON for non-code edits, Python for code defaults. |

Verified on macOS arm64 (Python 3.13.7); should work unchanged on Linux x86_64.

## Dependency graph

Grouped by layer so you know what to remove when stripping the project down.

```
data ingestion
  ├── soccerdata >= 1.6      # FBref scraper (with internal caching)
  ├── requests, httpx        # Polymarket / Kalshi REST clients
  ├── python-dotenv          # .env file loading
  └── pyarrow                # parquet read/write

modeling
  ├── pandas >= 2.1          # everything DataFrame
  ├── numpy >= 1.26          # Elo math, Monte Carlo
  ├── scipy                  # (reserved — not currently used; remove if cutting deps)
  └── scikit-learn >= 1.4    # (reserved — calibration uses ours not sklearn's)

api
  ├── fastapi >= 0.110
  ├── uvicorn[standard]      # ASGI server (websockets, http-tools)
  └── pydantic >= 2.6        # request/response schemas

app
  ├── plotly >= 5.20
  └── dash >= 2.16

testing
  ├── pytest >= 8.0
  └── httpx                  # (re-used) — TestClient backend

notebook
  └── jupyter, nbformat, nbclient (added at runtime for execute checks)
```

If you ever cut weight: `scipy` and `scikit-learn` are reserved for future use and can be dropped today without breaking anything.

## External services

| Service        | Endpoint                                                       | What we ask for                            | Auth        | Fallback                              |
| -------------- | -------------------------------------------------------------- | ------------------------------------------ | ----------- | ------------------------------------- |
| FBref          | via `soccerdata.FBref` (no public REST URL)                    | International fixtures + xG                | None        | Synthetic Poisson-generated history.  |
| Polymarket     | `https://gamma-api.polymarket.com/markets?q=World+Cup+2026...` | Outcome prices for WC winner market        | None        | Normalized synthetic prices (Euro-tilt). |
| Kalshi         | `https://api.elections.kalshi.com/trade-api/v2/markets`        | yes_bid + sub_title                        | Optional    | Normalized synthetic prices (US-tilt).   |

Every external call is wrapped in try/except and logs a `WARNING` on failure before falling back. **The pipeline never fails because of a flaky API.**

## Filesystem layout (runtime)

```
data/
├── cache/                         # soccerdata cache (gitignored)
├── matches.parquet                # raw fetch output
├── matches_clean.parquet          # post-cleaning master DataFrame
├── features.parquet               # rolling-window form features
├── ratings.parquet                # xG-adjusted Elo ratings
├── simulation.parquet             # 10k Monte Carlo aggregates
├── market_odds.parquet            # Polymarket + Kalshi snapshot
├── edges.parquet                  # model_prob - market_prob
└── team_name_map.json             # tracked, hand-maintained

analysis/output/
├── market_delta.csv               # full 48-team mispricing table
├── arbitrage.csv                  # Polymarket vs Kalshi gaps
├── calibration.csv                # reliability bins
├── model_compare.csv              # Brier scores
└── narrative.md                   # auto-written headline findings
```

All `.parquet` files and `analysis/output/` are gitignored — they regenerate from source.

## Process model

Three independent processes, none of which need each other to run.

```
                    ┌────────────────────┐
                    │ pipeline/*.py      │   (CLI scripts, run-to-completion)
                    │ analysis/*.py      │
                    └─────────┬──────────┘
                              │  writes to data/, analysis/output/
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │  data/*.parquet  +  analysis/output/*.csv               │
   │                                                         │
   └────────────┬───────────────────┬────────────────────────┘
                │                   │
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │ FastAPI      │    │ Dash app     │   (long-running daemons)
        │ uvicorn:8000 │    │ Flask:8050   │
        └──────────────┘    └──────────────┘
```

Pipelines write; daemons read. Both daemons hot-reload from disk on every request through `@lru_cache`-backed loaders, but the cache is process-lifetime — restart the process after re-running the pipeline.

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline (≈ 8 seconds end-to-end on a laptop)
python -m pipeline.fetch_data
python -m pipeline.clean_data
python -m pipeline.features
python -m pipeline.train_elo
python -m pipeline.simulate
python -m analysis.market_delta
python -m analysis.arbitrage
python -m analysis.calibration

# Servers
uvicorn api.main:app --reload          # http://127.0.0.1:8000  (Swagger at /docs)
python app/dashboard.py                # http://127.0.0.1:8050

# Tests
pytest tests/ -v                       # 31 tests, ~0.6s
```

## Deployment plan (Phase 8–10)

| Target               | Where           | What to set                                                               |
| -------------------- | --------------- | ------------------------------------------------------------------------- |
| FastAPI prod         | Zerve API block | Entry: `api/main.py`. Python 3.11. Requirements: `requirements.txt`.      |
| Dash app prod        | Zerve App Builder | Entry: `app/dashboard.py`. The `server` symbol is exposed for WSGI use.  |
| Notebook publish     | Zerve notebook  | `notebooks/analysis.ipynb`. Public visibility. Test in incognito after.   |

Capture every public URL into `STATUS.md` immediately on deploy.

## Reproducibility

Three deterministic seeds control everything stochastic:

| Seed                                | Where                      | Purpose                                  |
| ----------------------------------- | -------------------------- | ---------------------------------------- |
| `RANDOM_SEED = 42`                  | `pipeline/config.py`       | Synthetic match history, Monte Carlo.    |
| `RANDOM_SEED + 1`                   | `pipeline/fetch_data.py`   | Synthetic market snapshot.               |
| `pytest` fixture seeds (e.g. 0, 7)  | `tests/conftest.py`        | Stable test inputs.                      |

Same seeds → byte-identical parquet outputs. The `test_reproducibility` test enforces this end-to-end.

## CI / deploy hardening — when you're ready

Optional things to add before judging if time allows:

- [ ] `Makefile` with `make pipeline`, `make test`, `make app`, `make api`.
- [ ] GitHub Actions: install + `pytest` on push.
- [ ] Pre-commit hook that runs `pytest -q` and refuses on regression.
- [ ] `requirements-lock.txt` produced by `pip freeze` for fully-reproducible deploys.
- [ ] Health check curl in deployment platform pointing at `/health`.

None are required for the hackathon submission; all are easy wins if a judge tries to redeploy.
