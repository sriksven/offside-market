# Data + Execution Flow

> How data moves through the system, in what order, and what depends on what.

## Top-level diagram

```
                        ┌─────────────────────┐
                        │ External sources    │
                        │  • FBref            │
                        │  • Polymarket       │
                        │  • Kalshi           │
                        └──────────┬──────────┘
                                   │
                                   ▼  (synthetic fallback if any are unreachable)
              ┌─────────────────────────────────────────┐
              │  Phase 1 — pipeline/fetch_data.py       │
              └────────────┬────────────────┬───────────┘
                           │                │
                  matches.parquet     market_odds.parquet
                           │                │
                           ▼                │
       ┌────────────────────────────────┐   │
       │ Phase 2 — clean_data.py        │   │
       │  • canonicalize names           │   │
       │  • impute missing xG            │   │
       │  • apply competition weights    │   │
       └─────────────┬───────────────────┘   │
                     │                       │
              matches_clean.parquet          │
                     │                       │
        ┌────────────┴────────────┐          │
        ▼                         ▼          │
   ┌─────────────┐          ┌──────────────┐ │
   │ Phase 3     │          │ Phase 4       │ │
   │ features.py │          │ train_elo.py  │ │
   └──────┬──────┘          └──────┬───────┘ │
          │                        │         │
   features.parquet         ratings.parquet  │
          │                        │         │
          │                        ▼         │
          │                ┌────────────────┐│
          │                │ Phase 5         ││
          │                │ simulate.py     ││
          │                │  10 000 trials  ││
          │                └──────┬──────────┘│
          │                       │           │
          │              simulation.parquet   │
          │              edges.parquet ◀──────┤
          │                       │           │
          │                       ▼           │
          │       ┌────────────────────────┐  │
          │       │ Phase 6 — analysis/    │  │
          │       │  • market_delta.py     │◀─┤
          │       │  • arbitrage.py        │◀─┘
          │       │  • calibration.py      │
          │       └──────────┬─────────────┘
          │                  │
          │       analysis/output/*.csv + narrative.md
          │                  │
          ├──────────────────┴─────┐
          ▼                        ▼
   ┌─────────────┐          ┌──────────────┐
   │ FastAPI     │          │ Dash app     │
   │ api/main.py │          │ app/         │
   │             │          │  dashboard.py│
   └─────────────┘          └──────────────┘
```

## Strict execution order

The pipeline must run in this order — each step depends on the previous step's parquet output. Re-running a single step does not require re-running upstream steps.

```
fetch_data → clean_data → features    (form features, not used by Elo but used by Dash)
                       └→ train_elo → simulate → market_delta
                                              ├→ arbitrage
                                              └→ calibration
```

| Step              | Reads                                   | Writes                                                              |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------- |
| `fetch_data`      | (external sources or fallback seed)     | `matches.parquet`, `market_odds.parquet`                            |
| `clean_data`      | `matches.parquet`, `team_name_map.json` | `matches_clean.parquet`                                             |
| `features`        | `matches_clean.parquet`                 | `features.parquet`                                                  |
| `train_elo`       | `matches.parquet`                       | `ratings.parquet`                                                   |
| `simulate`        | `ratings.parquet`, `market_odds.parquet`| `simulation.parquet`, `edges.parquet`                               |
| `market_delta`    | `simulation.parquet`, `market_odds.parquet` | `market_delta.csv`, `narrative.md`, refreshes `edges.parquet`   |
| `arbitrage`       | `market_odds.parquet`                   | `arbitrage.csv`                                                     |
| `calibration`     | `matches_clean.parquet`                 | `calibration.csv`, `model_compare.csv`                              |

## Module-level call graphs

### Match prediction (FastAPI `/predict/match`)

```
POST /predict/match
   └─ api.main.predict_match
       ├─ _team_params(home)  ─ ratings_df (parquet, lru_cached)
       ├─ _team_params(away)
       ├─ _hda_probabilities
       │    ├─ pipeline.simulate._expected_goals(home, away)
       │    ├─ pipeline.simulate._expected_goals(away, home)
       │    └─ Poisson grid → home_win / draw / away_win
       └─ _confidence(home_elo, away_elo)
```

No Monte Carlo runs per request — the analytic Poisson grid resolves in microseconds.

### Tournament simulation (`pipeline.simulate.run_simulations`)

```
run_simulations(ratings, n=10_000)
   for trial in 1..n:
      └─ run_one_tournament
           ├─ draw_groups (snake-draw across 4 pots)
           ├─ for each of 12 groups:
           │     simulate_group → round-robin (6 matches)
           ├─ pick top 2 + 8 best 3rd-place teams (advancing list of 32)
           └─ knockout R32 → R16 → QF → SF → Final
                each call → simulate_match
                                 ├─ _expected_goals (both directions)
                                 ├─ rng.poisson  (goals)
                                 └─ rng.random   (penalty coin-flip if drawn)
   aggregate → counts per team per round → probabilities → DataFrame
```

### Elo training (`pipeline.train_elo.train`)

```
train(matches, k=24)
   sort matches by date
   for each match:
      ├─ expected_score(R_a, R_b, home_adv=65)
      ├─ actual_score(goals_a, goals_b, xg_a, xg_b, blend=0.55)
      │     blend(real_outcome ∈ {0, 0.5, 1}, sigmoid(xg_a - xg_b))
      ├─ mov = log(1 + |xg_a - xg_b|) + 1     (margin-of-victory multiplier)
      ├─ delta = K * mov * (s_home - e_home)
      ├─ ratings[home] += delta
      ├─ ratings[away] -= delta
      └─ update attack/defense exponential moving averages
   return DataFrame[team, rating, n_matches, last_match, attack, defense, rank]
```

## Test execution flow

```
pytest tests/
   ├─ test_elo.py        (no IO; pure function tests)
   ├─ test_pipeline.py   (uses tiny_matches fixture; tests cleaning/canonicalization)
   ├─ test_simulation.py (uses synthesized 48-team rating frame; tests structural invariants)
   └─ test_api.py        (uses fastapi.TestClient; SKIPS gracefully if pipeline outputs missing)
```

API tests have `@pytest.mark.skipif(not PIPELINE_READY, ...)` so the suite passes from a clean clone. To exercise everything, run the pipeline first, then `pytest`.

## What hot-reloads vs what doesn't

| Component        | Reload trigger                                                  |
| ---------------- | --------------------------------------------------------------- |
| FastAPI          | `--reload` flag re-imports on file change. Parquet caches survive across reloads only if the lru_cache is hit before any reload. **Restart after pipeline reruns.** |
| Dash app         | No hot-reload by default. Restart manually after pipeline reruns. |
| Pipeline scripts | N/A (run-to-completion).                                        |

## Failure modes and graceful fallbacks

| Failure                                  | What happens                                              | Where logged              |
| ---------------------------------------- | --------------------------------------------------------- | ------------------------- |
| `soccerdata` import fails                | Synthetic matches used.                                   | `offside.fetch` WARNING   |
| FBref scrape errors                      | Synthetic matches used.                                   | `offside.fetch` WARNING   |
| Polymarket REST 4xx/5xx                  | Synthetic Polymarket prices used.                         | `offside.fetch` WARNING   |
| Kalshi REST 4xx/5xx                      | Synthetic Kalshi prices used.                             | `offside.fetch` WARNING   |
| User hits API before pipeline ran        | API returns **503** with `Run the pipeline first…` hint.  | HTTPException             |
| User hits Dash before pipeline ran       | Dash refuses to boot with same hint.                      | `SystemExit` at import    |
| Calibration not yet run, `/calibration`  | API returns **503** with `Run analysis.calibration` hint. | HTTPException             |
| Invalid team name on `/predict/match`    | Returns **422** with `valid_teams_sample` and `/teams` hint. | HTTPException          |
| Same team home and away                  | Returns **422** with helpful detail.                      | HTTPException             |

The system has no silent failures — every fallback logs, every error returns an actionable code.
