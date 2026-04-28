# Project Status — Offside Market

> Single source of truth for "what's done, what's next, and where each piece lives."
> **Update this file every time you complete a phase, ship a deliverable, or change scope.**
>
> Sibling docs in this folder go deeper: `INFRASTRUCTURE.md` (deps, deploy targets),
> `FLOW.md` (data + execution flow), `FILE_INDEX.md` (per-file map),
> `COMPLETED.md` (detailed completion + known limitations).

Last updated: **2026-04-27**

---

## TL;DR

| Layer                     | Status        | Proof                                                            |
| ------------------------- | ------------- | ---------------------------------------------------------------- |
| Data ingestion            | DONE          | `pipeline/fetch_data.py` + `data/matches.parquet`                |
| Cleaning                  | DONE          | `pipeline/clean_data.py` + `data/matches_clean.parquet`          |
| Feature engineering       | DONE          | `pipeline/features.py` + `data/features.parquet`                 |
| xG-adjusted Elo           | DONE          | `pipeline/train_elo.py` + `data/ratings.parquet`                 |
| 10k Monte Carlo           | DONE          | `pipeline/simulate.py` + `data/simulation.parquet`               |
| Market delta              | DONE          | `analysis/market_delta.py` + `analysis/output/market_delta.csv`  |
| Cross-market arbitrage    | DONE          | `analysis/arbitrage.py` + `analysis/output/arbitrage.csv`        |
| Calibration backtest      | DONE          | `analysis/calibration.py` + `analysis/output/model_compare.csv`  |
| FastAPI                   | DONE (local)  | `api/main.py`, smoke-tested on `:8000`                           |
| Dash app (5 views)        | DONE (local)  | `app/dashboard.py`, smoke-tested on `:8050`                      |
| Test suite                | DONE          | `tests/`, 31 passing in 0.6s                                     |
| Notebook                  | DONE          | `notebooks/analysis.ipynb`, executes end-to-end                  |
| **Zerve deployment**      | NOT DONE      | Phase 10 — needs publishing + URL captured below                 |
| **FastAPI prod deploy**   | NOT DONE      | Phase 8 — needs Zerve API block + endpoint URL                   |
| **Dash prod deploy**      | NOT DONE      | Phase 9 — needs Zerve App Builder + public URL                   |
| **Demo video**            | NOT DONE      | Phase 11 — 3 min, structure in implementation guide              |
| **Devpost write-up**      | NOT DONE      | Phase 11 — 300-word summary + URLs                               |
| **Social post**           | NOT DONE      | Phase 11 — LinkedIn or X, tag @Zerve_AI                          |

---

## Phase-by-Phase Map

Maps each numbered phase from the implementation guide to the file(s) that prove it is done. When you finish a phase, flip its checkbox and link the new file.

### Phase 1 — Data collection — DONE
- [x] `pipeline/fetch_data.py` — FBref via `soccerdata`, Polymarket, Kalshi (with synthetic fallback)
- [x] `data/matches.parquet`, `data/market_odds.parquet`

### Phase 2 — Data cleaning — DONE
- [x] `pipeline/clean_data.py`
- [x] `data/team_name_map.json` (80+ aliases)
- [x] `data/matches_clean.parquet`

### Phase 3 — Feature engineering — DONE
- [x] `pipeline/features.py`
- [x] `data/features.parquet`

### Phase 4 — xG-adjusted Elo — DONE
- [x] `pipeline/train_elo.py` with margin-of-victory multiplier and xG blending
- [x] `data/ratings.parquet` (sanity check: Spain, Argentina, France, Brazil top 4)

### Phase 5 — Monte Carlo simulation — DONE
- [x] `pipeline/simulate.py` — 10 000 trials, 12 groups → R32 → … → Final
- [x] `data/simulation.parquet`, `data/edges.parquet`

### Phase 6 — Market delta + arbitrage + calibration — DONE
- [x] `analysis/market_delta.py` → `analysis/output/market_delta.csv` + `narrative.md`
- [x] `analysis/arbitrage.py` → `analysis/output/arbitrage.csv`
- [x] `analysis/calibration.py` → `analysis/output/calibration.csv` + `model_compare.csv`

### Phase 7 — Testing — DONE
- [x] `tests/test_elo.py`, `tests/test_pipeline.py`, `tests/test_simulation.py`, `tests/test_api.py`
- [x] **31 / 31 passing** in 0.6 seconds (`pytest tests/ -v`)

### Phase 8 — FastAPI — DONE (local) / NOT DEPLOYED
- [x] `api/main.py` — `/health /teams /standings /edges /arbitrage /calibration /predict/match /team/{name}`
- [x] `api/models.py` — Pydantic schemas
- [ ] **Deploy on Zerve and capture the public URL here:** `<paste URL when shipped>`

### Phase 9 — Dash app — DONE (local) / NOT DEPLOYED
- [x] `app/dashboard.py` — five tabs: Match Predictor / Mispricing / Bracket / Calibration / Arbitrage
- [x] `app/assets/styles.css` — dark-mode-aware, responsive
- [ ] **Deploy on Zerve App Builder, capture URL:** `<paste URL when shipped>`
- [ ] Verify on mobile (375px) and desktop (1440px)

### Phase 10 — Zerve notebook — DONE (local) / NOT PUBLISHED
- [x] `notebooks/analysis.ipynb` — 10 sections, executes top-to-bottom
- [ ] **Publish to Zerve, capture public URL:** `<paste URL when shipped>`
- [ ] Test the URL in an incognito window

### Phase 11 — Submission — NOT DONE
- [ ] Public Zerve project URL captured (Phase 10)
- [ ] FastAPI URL captured (Phase 8)
- [ ] Dash URL captured (Phase 9)
- [ ] Project summary (≤ 300 words) — write here or link
- [ ] Demo video (3 min, structure in implementation guide section "Demo video structure")
- [ ] Posted on LinkedIn or X tagging @Zerve_AI
- [ ] Devpost write-up filled

---

## Docs Map — what to read for what

| Doc                                        | Role                                                            | Update when                                                              | Owner                |
| ------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------- |
| `README.md` (repo root)                    | User-facing intro, quickstart, API surface, sample numbers      | Anything user-visible changes (CLI commands, endpoints, output shape)    | Hand-maintained      |
| `docs/README.md`                           | Index of every doc in this folder + update protocol             | A doc is added, renamed, or split                                        | Hand-maintained      |
| `docs/STATUS.md` (this file)               | "What's done / next" tracker + docs map + verification commands | Every phase complete, deliverable shipped, scope change                  | Hand-maintained      |
| `docs/INFRASTRUCTURE.md`                   | Runtime, deps, external services, filesystem, deployment plan   | Dependency / runtime / deploy target changes                             | Hand-maintained      |
| `docs/FLOW.md`                             | Data flow + execution order + module call graphs                | Pipeline step or endpoint added/changed; failure modes change            | Hand-maintained      |
| `docs/FILE_INDEX.md`                       | Per-file map of what every tracked file does                    | A file is added/renamed; a public symbol meaningfully changes            | Hand-maintained      |
| `docs/COMPLETED.md`                        | Phase-by-phase completion detail + known limitations            | Phase moves forward; limitation fixed; new caveat appears                | Hand-maintained      |
| `notebooks/analysis.ipynb`                 | The Zerve-published narrative; judges' first read               | Pipeline/model/headline-numbers change; before publishing                | Hand-maintained      |
| `analysis/output/narrative.md`             | Auto-generated headline findings (top 5 over/undervalued teams) | Auto: regenerated by `python -m analysis.market_delta`                   | Auto-regenerated     |
| `analysis/output/market_delta.csv`         | Full 48-team model-vs-market table                              | Auto: `python -m analysis.market_delta`                                  | Auto-regenerated     |
| `analysis/output/arbitrage.csv`            | Cross-market gap table                                          | Auto: `python -m analysis.arbitrage`                                     | Auto-regenerated     |
| `analysis/output/calibration.csv`          | Reliability-curve bins                                          | Auto: `python -m analysis.calibration`                                   | Auto-regenerated     |
| `analysis/output/model_compare.csv`        | Brier scores: naive / win-loss / xG-Elo                         | Auto: `python -m analysis.calibration`                                   | Auto-regenerated     |
| FastAPI `/docs` (Swagger UI)               | Live, interactive API contract                                  | Auto: regenerates from `api/main.py` + `api/models.py` on each boot      | Auto-regenerated     |
| FastAPI `/health`                          | Live artifact-freshness probe (`data_freshness` ISO timestamp)  | Auto: returns the most-recent artifact mtime on every call               | Auto-regenerated     |
| `data/team_name_map.json`                  | Canonical-name lookup (USA → United States, etc.)               | New aliases observed in real FBref/Polymarket/Kalshi data                | Hand-maintained      |
| `pipeline/config.py`                       | The 48 qualified teams, hyperparameters, paths                  | Bracket changes, hyperparameter tuning, new artifact paths               | Hand-maintained      |
| `.env.example`                             | Documented environment variables                                | New env var added or default changed                                     | Hand-maintained      |
| `requirements.txt`                         | Pinned-floor Python deps                                        | New import added; bump only when needed                                  | Hand-maintained      |
| `.gitignore`                               | What's tracked vs what's regenerated                            | New artifact directory added; new tracked config added                   | Hand-maintained      |
| `tests/`                                   | Behavioral contract                                             | Any model/API/pipeline behavior changes; new feature ships               | Hand-maintained      |

---

## Update protocol

Use this checklist whenever you finish a unit of work, before stepping away or pushing.

1. **If a phase moved forward**, flip the checkbox in this file and paste the path to the artifact that proves it.
2. **If you shipped a public URL** (Zerve / API / app), paste it under the relevant phase _and_ in the TL;DR table at the top.
3. **If pipeline outputs changed** (model tuning, new data, schema change):
   - Re-run the full pipeline: `python -m pipeline.fetch_data && python -m pipeline.clean_data && python -m pipeline.features && python -m pipeline.train_elo && python -m pipeline.simulate && python -m analysis.market_delta && python -m analysis.arbitrage && python -m analysis.calibration`
   - Re-execute the notebook: `jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis.ipynb`
   - Update the "Sample headline numbers" table in `README.md` if any top-5 team flipped
4. **If user-visible behavior changed** (new endpoint, new tab, new CLI flag), update `README.md`'s relevant section.
5. **Always run tests before claiming done:** `pytest tests/ -v` — should be 31/31.
6. **Update the "Last updated" date at the top of this file.**

---

## Quick verification commands

These are the commands a future contributor or judge can run to verify every claim above is true.

```bash
# 1. Full clean rebuild — should finish in < 10 seconds.
rm -rf data/*.parquet analysis/output
python -m pipeline.fetch_data
python -m pipeline.clean_data
python -m pipeline.features
python -m pipeline.train_elo
python -m pipeline.simulate
python -m analysis.market_delta
python -m analysis.arbitrage
python -m analysis.calibration

# 2. Tests — should be 31 passing.
pytest tests/ -v

# 3. API smoke test.
uvicorn api.main:app --port 8000 &
sleep 2
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict/match \
  -H "Content-Type: application/json" \
  -d '{"home":"Spain","away":"Brazil"}'

# 4. Dash app smoke test.
python app/dashboard.py    # then visit http://localhost:8050

# 5. Notebook end-to-end.
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis.ipynb
```
