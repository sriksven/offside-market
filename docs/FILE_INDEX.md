# File Index

> Every tracked file in the repo, what it contains, what it produces, and the
> public symbols you'll most likely import. Generated artifacts (`*.parquet`,
> `analysis/output/*`, `data/cache/*`) are in `INFRASTRUCTURE.md` instead.

## Top-level

| File                  | Purpose                                                                      |
| --------------------- | ---------------------------------------------------------------------------- |
| `README.md`           | User-facing intro, quick start, sample numbers. The only `.md` at the repo root. |
| `requirements.txt`    | Pinned-floor Python deps. Only edit when an import is genuinely added.       |
| `.env.example`        | Documented environment variables (API keys, seeds, simulation knobs).        |
| `.gitignore`          | Excludes generated parquet, caches, and `data/cache/`.                       |

## `pipeline/` — the data pipeline

The pipeline is a chain of run-to-completion CLI scripts. Order: `fetch_data → clean_data → features → train_elo → simulate`.

### `pipeline/__init__.py`
Empty package marker.

### `pipeline/config.py`
Constants and paths shared across all pipeline modules. Edit here for model knobs.

| Symbol                       | Type                  | What it controls                                            |
| ---------------------------- | --------------------- | ----------------------------------------------------------- |
| `WORLD_CUP_2026_TEAMS`       | `list[str]` (48 teams)| The qualified field. Roughly tier-ordered.                  |
| `MATCHES_PATH`, `RATINGS_PATH`, `MARKET_PATH`, `SIMULATION_PATH`, `EDGES_PATH` | `Path` | Where each artifact gets written. |
| `ELO_INIT = 1500.0`          | float                 | Starting rating for every team.                             |
| `ELO_K = 24.0`               | float                 | Base K-factor before margin-of-victory multiplier.          |
| `ELO_HOME_ADV = 65.0`        | float                 | Elo-point bump for the home side.                           |
| `XG_WEIGHT = 0.55`           | float                 | Blend between actual outcome (0) and xG-derived score (1).  |
| `N_SIMULATIONS = 10_000`     | int                   | Monte Carlo trial count.                                    |
| `RANDOM_SEED = 42`           | int                   | Deterministic seed for synthetic data + simulation.         |

### `pipeline/fetch_data.py`
**Phase 1.** Pulls international fixtures and live market odds; falls back to deterministic synthetic data when external sources are unreachable.

| Public symbol             | Purpose                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `fetch_fbref_matches`     | Try real FBref via `soccerdata`; returns `None` on failure.        |
| `synthesize_matches`      | Generate Poisson-based 5-year history for the 48 WC2026 teams.     |
| `load_match_history`      | Top-level match loader (real-then-fallback).                       |
| `MarketSnapshot`          | Dataclass: `polymarket: dict[str, float]`, `kalshi: dict[str, float]`. |
| `_fetch_polymarket`, `_fetch_kalshi` | Best-effort REST clients with timeouts.                |
| `synthesize_market`       | Realistic synthetic prices, normalized to ~1.05 (5% overround).    |
| `load_market_odds`        | Top-level market loader.                                           |
| `main`                    | CLI entry: writes `matches.parquet` and `market_odds.parquet`.     |

### `pipeline/clean_data.py`
**Phase 2.** Canonicalizes names, dedupes, imputes xG, applies competition weights.

| Public symbol           | Purpose                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `REQUIRED_COLS`         | List of columns guaranteed in the cleaned output.                    |
| `load_name_map`         | Reads `data/team_name_map.json`.                                     |
| `canonicalize`          | Apply name map; pass unknown names through unchanged.                |
| `COMPETITION_WEIGHTS`   | Friendly 0.3 → World Cup 1.5.                                        |
| `competition_weight`    | Lookup with substring matching.                                      |
| `impute_xg`             | Linear-fit xG from goals; flags imputed rows for 0.5× weighting.     |
| `clean_matches`         | The end-to-end function. Input: raw matches DF. Output: cleaned DF.  |
| `main`                  | CLI entry: writes `matches_clean.parquet`.                           |

### `pipeline/features.py`
**Phase 3.** Per-team rolling form features (used by Dash, not by Elo).

| Public symbol      | Purpose                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `DECAY_HALFLIFE_DAYS = 365.0` | Recency decay half-life.                                      |
| `HOME_ADV_GOALS = 0.18`       | xG-equivalent home bump for neutral adjustment.               |
| `_team_long_form`  | Reshape match table to per-team-per-match.                               |
| `_decay_weight`    | Exponential time decay relative to a reference date.                     |
| `_neutral_adjust`  | Strip home-advantage from per-row xG.                                    |
| `build_features`   | Compute rolling 5/10/20-match xG attack & defense, recent form string.   |
| `main`             | CLI entry: writes `features.parquet`.                                    |

### `pipeline/train_elo.py`
**Phase 4.** xG-adjusted Elo with margin-of-victory multiplier.

| Public symbol      | Purpose                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `expected_score`   | Standard Elo: `1 / (1 + 10^(-(R_a + home_adv - R_b)/400))`.              |
| `actual_score`     | Blend of `{0, 0.5, 1}` real outcome with sigmoid xG-differential score.  |
| `train`            | Walk matches in date order; update ratings + attack/defense EWMAs.       |
| `main`             | CLI entry: writes `ratings.parquet`.                                     |

### `pipeline/simulate.py`
**Phase 5.** 10 000-trial Monte Carlo simulation of the 2026 bracket.

| Public symbol         | Purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `TeamParams`          | Dataclass: `rating`, `attack`, `defense`.                                |
| `_expected_goals`     | Per-side Poisson rate from rating gap + attack/defense baseline.         |
| `simulate_match`      | One Poisson match with optional knockout-mode penalty resolution.        |
| `draw_groups`         | Snake-draw 48 teams into 12 groups of 4 across 4 rating-tiered pots.     |
| `simulate_group`      | Round-robin → standings tuple (team, points, GD, GF).                    |
| `run_one_tournament`  | Full bracket simulation; returns `{team: deepest_round_reached}`.        |
| `ROUND_ORDER`         | `["Group", "R32", "R16", "QF", "SF", "Final", "Champion"]`.              |
| `run_simulations`     | n trials → DataFrame[team, p_round_*, model_prob, rank].                 |
| `compute_edges`       | Join with market → `edge`, `edge_pct`.                                   |
| `main`                | CLI entry: writes `simulation.parquet`, `edges.parquet`.                 |

## `analysis/` — the analysis layer

### `analysis/__init__.py`
Package docstring; nothing executable.

### `analysis/market_delta.py`
**Phase 6.1.** Headline mispricing table + auto-written narrative.

| Public symbol      | Purpose                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `OUTPUT_DIR`, `DELTA_CSV`, `NARRATIVE_PATH` | Write paths.                                    |
| `build_delta_table`| Join sim + market, compute delta_vs_polymarket / kalshi / blend.         |
| `write_narrative`  | Render top-5 over/undervalued teams as `narrative.md`.                   |
| `main`             | CLI entry: writes `market_delta.csv`, `narrative.md`, refreshes `edges.parquet`. |

### `analysis/arbitrage.py`
**Phase 6.2.** Cross-market gap detector.

| Public symbol           | Purpose                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| `DEFAULT_FEE_THRESHOLD` | 2 percentage points after fees.                                     |
| `detect`                | DataFrame[team, polymarket, kalshi, gap, higher_market, actionable].|
| `main`                  | CLI entry: writes `arbitrage.csv`.                                  |

### `analysis/calibration.py`
**Phase 6.3.** Brier comparison + reliability curve.

| Public symbol      | Purpose                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `brier`            | Standard binary Brier score.                                             |
| `split_train_test` | Chronological split at a cutoff date.                                    |
| `_matches_to_train_format` | Adapter from cleaned schema to `train()` schema.                 |
| `evaluate`         | Train xG-Elo + win/loss-Elo on training tail; compare Brier on test.     |
| `reliability_curve`| Bin predictions and report `predicted_mean` vs `actual_rate`.            |
| `main`             | CLI entry: writes `calibration.csv`, `model_compare.csv`.                |

## `api/` — FastAPI service

### `api/__init__.py`
Empty package marker.

### `api/main.py`
**Phase 8.** The deployed prediction API.

| Endpoint                  | Purpose                                                          |
| ------------------------- | ---------------------------------------------------------------- |
| `GET /health`             | Liveness + artifact freshness.                                   |
| `GET /teams`              | The 48 canonical names (use these in `/predict`).                |
| `GET /standings`          | Sorted model-vs-market with `direction` label.                   |
| `GET /edges?limit=10`     | Top mispricings sorted by absolute edge.                         |
| `GET /arbitrage?min_gap`  | Cross-market gaps; reads `arbitrage.csv` or computes on the fly. |
| `GET /calibration`        | Brier compare + reliability curve.                               |
| `POST /predict/match`     | Two-team prediction (analytic Poisson; no per-request MC).       |
| `GET /team/{name}`        | Per-team rating + simulation breakdown + market odds.            |

| Internal symbol                         | Purpose                                                  |
| --------------------------------------- | -------------------------------------------------------- |
| `ratings_df`, `simulation_df`, `market_df`, `edges_df` | `@lru_cache(1)` parquet loaders.          |
| `_data_freshness`                       | ISO timestamp of newest artifact (UTC).                  |
| `_team_params`                          | Lookup → raises 422 with helpful hint if unknown.        |
| `_hda_probabilities`                    | Joint Poisson grid up to `cap=8` goals.                  |
| `_confidence`                           | Maps Elo gap → `low` / `medium` / `high`.                |

### `api/models.py`
Pydantic schemas: `MatchRequest`, `MatchResponse`, `StandingRow`, `ArbitrageRow`, `HealthResponse`.

## `app/` — Dash dashboard

### `app/__init__.py`
Empty package marker.

### `app/dashboard.py`
**Phase 9.** Five-tab Plotly Dash app. Bootstraps `sys.path` so it works as both `python app/dashboard.py` and `python -m app.dashboard`.

| Symbol                              | Purpose                                                    |
| ----------------------------------- | ---------------------------------------------------------- |
| `app`, `server`                     | Dash and underlying Flask WSGI app (for prod deploy).      |
| `predict_match(home, away, neutral)`| Same Poisson grid as the API; reused for `matchup` view.   |
| `matchup_view`                      | Hero view: H/D/A bars, Elo, recent form, market context.   |
| `fig_mispricing`, `fig_edge_bars`   | Mispricing dashboard (filterable).                         |
| `fig_bracket`                       | Round-by-round bars for top 16 model seeds.                |
| `fig_calibration`, `calibration_view` | Reliability curve + Brier comparison table.              |
| `arbitrage_view`                    | Sortable/filterable cross-market gap table.                |
| `render_tab`, `update_matchup`, `update_mispricing` | Dash callbacks.                            |

### `app/assets/styles.css`
Loaded automatically by Dash from `assets/`. Defines:

- CSS variables for light/dark themes (`prefers-color-scheme: dark`).
- `.om-shell`, `.om-hero`, `.om-card`, `.om-pill--good/bad/meh` utility classes.
- Mobile breakpoint at 700px (stack controls vertically).

## `tests/` — pytest suite

### `tests/__init__.py`
Empty package marker.

### `tests/conftest.py`
Shared fixtures.

| Fixture            | Provides                                                                |
| ------------------ | ----------------------------------------------------------------------- |
| `tiny_matches`     | 40-row deterministic match table covering 4 teams.                      |
| `cleaned_matches`  | Same, run through `clean_data.clean_matches`.                           |

### `tests/test_elo.py`
Pure-function tests: expected-score symmetry, xG-blend semantics at `blend=0` vs `blend=1`, training direction (winner ↑, loser ↓), chronological-order invariance.

### `tests/test_pipeline.py`
Cleaning + canonicalization tests: name-map coverage, no-nulls in required columns, result values constrained to `{0, 0.5, 1}`, competition-weight ordering, market price unit interval.

### `tests/test_simulation.py`
Monte Carlo invariants: single winner per trial, win-probs sum to 1.0, reproducibility under fixed seed, top seed clears the 1/48 baseline, monotone round probabilities (P(R32) ≥ P(R16) ≥ … ≥ P(Champion)).

### `tests/test_api.py`
TestClient-driven endpoint smoke tests with `@pytest.mark.skipif(not PIPELINE_READY)` for endpoints needing data. Covers `/health`, `/teams`, `/standings`, `/predict/match` happy + 422 paths, `/arbitrage`, `/edges`.

## `data/` — tracked configuration

### `data/team_name_map.json`
Canonical-name lookup (80+ aliases). Add new mappings here, not in code.

### `data/.gitkeep`
Keeps the directory present in clones.

## `notebooks/`

### `notebooks/analysis.ipynb`
**Phase 10.** Ten-section narrative notebook. Sections:

1. Introduction
2. Data ingestion
3. Cleaning
4. EDA
5. Model training (xG-adjusted Elo)
6. Backtest (Brier comparison: naive < win-loss < xG-Elo)
7. 2026 simulation (Monte Carlo)
8. Market delta
9. Arbitrage
10. Conclusion (3 findings)

Execute with: `jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis.ipynb`.

## `docs/` — all project documentation

### `docs/README.md`
Index of every doc in this folder + update protocol.

### `docs/STATUS.md`
Public progress tracker: TL;DR table, phase-by-phase checkboxes, docs map, verification commands. Read this first.

### `docs/INFRASTRUCTURE.md`
Versions, deps, external services, filesystem layout, process model, deployment plan, reproducibility seeds.

### `docs/FLOW.md`
Data + execution flow with diagrams. Strict step ordering, module call graphs, failure modes + fallbacks.

### `docs/FILE_INDEX.md`
This file. Per-file map of what every tracked file does and the public symbols it exposes.

### `docs/COMPLETED.md`
Phase-by-phase completion detail with implementation notes, known limitations, and the "riskiest thing right now" callout.
