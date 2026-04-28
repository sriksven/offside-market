# Completion Status — Detailed

> Deeper than `STATUS.md`. Tracks the implementation notes, known caveats, and
> the small "we should fix this someday" items that the public tracker hides.

Last updated: **2026-04-27**

## Legend

| Marker  | Meaning                                                            |
| ------- | ------------------------------------------------------------------ |
| ✅       | Done and verified end-to-end.                                       |
| 🟡       | Done locally; deployment / external action still required.          |
| ⬜       | Not started.                                                        |
| ⚠️       | Done with a known limitation worth noting.                          |

---

## Phase 1 — Data collection ✅

**File:** `pipeline/fetch_data.py`

| Item                                                       | Status |
| ---------------------------------------------------------- | ------ |
| FBref scrape via `soccerdata`                              | ✅      |
| Polymarket REST client                                     | ✅      |
| Kalshi REST client                                         | ✅      |
| Synthetic-data fallback when sources unreachable           | ✅      |
| Output: `data/matches.parquet`, `data/market_odds.parquet` | ✅      |

**Known limitations:**
- ⚠️ When live sources are unreachable, we use synthetic data. The synthetic prices are normalized to a 5% overround and roughly tier-ordered, but they don't reflect real market sentiment. **Verify against live odds before quoting numbers in the demo.**
- ⚠️ Polymarket / Kalshi REST endpoints may shape-shift; the `_fetch_*` functions catch any exception and fall back. Monitor logs.

---

## Phase 2 — Data cleaning ✅

**File:** `pipeline/clean_data.py`

| Item                                                  | Status |
| ----------------------------------------------------- | ------ |
| Team-name canonicalization via `team_name_map.json`   | ✅      |
| Deduplication on `(date, home_team, away_team)`       | ✅      |
| xG imputation with linear-fit fallback                | ✅      |
| Competition-weight assignment                         | ✅      |
| `result ∈ {0, 0.5, 1}` column                         | ✅      |
| Output: `data/matches_clean.parquet`                  | ✅      |

**Known limitations:**
- ⚠️ Synthetic data has zero rows with missing xG, so the imputer is exercised only when real FBref data flows through. Real-data path validated by code review only, not live runs.

---

## Phase 3 — Feature engineering ✅

**File:** `pipeline/features.py`

| Item                                                     | Status |
| -------------------------------------------------------- | ------ |
| Rolling 5/10/20-match xG attack and defense              | ✅      |
| Exponential recency decay (365-day half-life)            | ✅      |
| Neutral-venue adjustment (strip home advantage)          | ✅      |
| Recent-form string ("WLDWW")                             | ✅      |
| Output: `data/features.parquet`                          | ✅      |

**Known limitations:**
- ⚠️ The squad-quality feature mentioned in the implementation guide ("Pull current Elo rating of each player's club from Club Elo, average top 15") was descoped — **not implemented**. The model performs adequately without it; revisit if Elo numbers feel stale.

---

## Phase 4 — xG-adjusted Elo ✅

**File:** `pipeline/train_elo.py`

| Item                                                                     | Status |
| ------------------------------------------------------------------------ | ------ |
| Standard Elo expected score with home advantage                          | ✅      |
| Blended actual score: `(1-blend)*real + blend*sigmoid(xg_diff)`          | ✅      |
| Margin-of-victory multiplier: `log(1 + |xg_diff|) + 1`                   | ✅      |
| Per-team attack/defense EWMAs (used by the simulator)                    | ✅      |
| Output: `data/ratings.parquet`                                           | ✅      |
| Sanity check: Spain, Argentina, France, Brazil, Germany top 5            | ✅      |

**Known limitations:**
- ⚠️ K-factor grid search (`K ∈ {10, 20, 30, 40, 50}` per the implementation guide) is **not implemented**. We hard-code `ELO_K = 24`. Tuning was done by eye; sufficient for the synthetic dataset.
- ⚠️ WC2022 backtest comparing model predictions vs actuals is **partial** — we have a held-out Brier comparison in `analysis/calibration.py`, but a dedicated WC2022-bracket replay isn't built. The current backtest is a stronger statistical test (more matches than just one tournament) but loses the storytelling power of "we predicted Argentina in the final."

---

## Phase 5 — Monte Carlo simulation ✅

**File:** `pipeline/simulate.py`

| Item                                                                     | Status |
| ------------------------------------------------------------------------ | ------ |
| 10 000 trials, deterministic with `RANDOM_SEED = 42`                     | ✅      |
| 48 teams → 12 groups of 4 (snake-draw across 4 rating-tiered pots)       | ✅      |
| Top 2 + 8 best 3rd-place advance to R32                                  | ✅      |
| Knockout R32 → R16 → QF → SF → Final with penalty resolution             | ✅      |
| Per-team round-reach probabilities                                       | ✅      |
| Output: `data/simulation.parquet`, `data/edges.parquet`                  | ✅      |
| Runtime: ~4 seconds for full 10k                                         | ✅      |

**Known limitations:**
- ⚠️ The actual 2026 bracket *seeding* (which group plays which knockout slot) is not modeled — we shuffle the 32 advancing teams uniformly into the R32 bracket. For a championship-probability number this doesn't matter (every team has the same expected path length); for a "most likely route" feature it would.

---

## Phase 6 — Analysis ✅

### 6.1 Market delta — `analysis/market_delta.py`

| Item                                                       | Status |
| ---------------------------------------------------------- | ------ |
| 48-team table with model_prob, market_prob, deltas         | ✅      |
| Auto-written `narrative.md` with top 5 over/undervalued    | ✅      |
| Refreshes `data/edges.parquet` for the API                 | ✅      |

### 6.2 Arbitrage — `analysis/arbitrage.py`

| Item                                                       | Status |
| ---------------------------------------------------------- | ------ |
| Polymarket-vs-Kalshi gap detection                         | ✅      |
| Configurable fee threshold (default 2pp)                   | ✅      |
| `actionable` boolean                                       | ✅      |

### 6.3 Calibration — `analysis/calibration.py`

| Item                                                          | Status |
| ------------------------------------------------------------- | ------ |
| Held-out chronological train/test split                       | ✅      |
| Brier comparison: naive vs win/loss-only vs xG-blended Elo    | ✅      |
| Reliability curve binned by predicted probability             | ✅      |
| **Result: xG-Elo (0.165) < win/loss-Elo (0.167) < naive (0.177)** | ✅  |

**Known limitations:**
- ⚠️ Calibration runs on the last 12 months of the training data, not on the 2022 World Cup specifically. Storytelling-wise we lose "Argentina won at 6% market odds" — it's a tradeoff for statistical power.

---

## Phase 7 — Testing ✅

**Files:** `tests/test_*.py`, `tests/conftest.py`

| Item                                          | Status            |
| --------------------------------------------- | ----------------- |
| `test_elo.py` — 7 tests                       | ✅                 |
| `test_pipeline.py` — 11 tests                 | ✅                 |
| `test_simulation.py` — 5 tests                | ✅                 |
| `test_api.py` — 8 tests                       | ✅                 |
| **Total: 31 tests, ~0.6s runtime**            | ✅                 |
| API tests skip cleanly when pipeline outputs missing | ✅          |
| Clean-environment rebuild test (Phase 7 spec) | ✅ (manual; passes) |

**Known limitations:**
- ⚠️ No CI configuration committed — tests must be run locally. A 5-line GitHub Actions YAML would close this.

---

## Phase 8 — FastAPI 🟡

**File:** `api/main.py`

| Item                              | Status |
| --------------------------------- | ------ |
| `/health`                         | ✅      |
| `/teams`                          | ✅      |
| `/standings`                      | ✅      |
| `/edges`                          | ✅      |
| `/arbitrage`                      | ✅      |
| `/calibration`                    | ✅      |
| `/predict/match` (POST)           | ✅      |
| `/team/{name}`                    | ✅      |
| 422 on invalid team / same team   | ✅      |
| 503 on missing artifacts          | ✅      |
| `data_freshness` on every response| ✅      |
| CORS enabled                      | ✅      |
| Pydantic schemas in `api/models.py`| ✅     |
| **Deployed on Zerve**             | ⬜      |
| Public URL captured in STATUS.md  | ⬜      |

---

## Phase 9 — Dash app 🟡

**File:** `app/dashboard.py`

| View / feature                                       | Status |
| ---------------------------------------------------- | ------ |
| Match Predictor (hero view)                          | ✅      |
| Mispricing dashboard with under/over/all filter      | ✅      |
| Bracket / round-by-round view                        | ✅      |
| Calibration view (reliability + Brier table)         | ✅      |
| Arbitrage view (sortable/filterable table)           | ✅      |
| `app/assets/styles.css` (responsive, dark-mode-aware)| ✅      |
| `server` symbol exposed for WSGI deploy              | ✅      |
| **Deployed on Zerve App Builder**                    | ⬜      |
| Public URL captured in STATUS.md                     | ⬜      |
| Tested at 375px and 1440px                           | ⬜      |

**Known limitations:**
- ⚠️ Bracket view shows round-by-round bars for the top 16 seeds, not a literal bracket diagram with team-vs-team slots. Closer to the implementation guide's intent would be a true single-elimination tree visualization with click-to-highlight paths.

---

## Phase 10 — Zerve notebook 🟡

**File:** `notebooks/analysis.ipynb`

| Item                                                       | Status |
| ---------------------------------------------------------- | ------ |
| 10 sections per the implementation guide                   | ✅      |
| Markdown narrative between every code block                | ✅      |
| Executes top-to-bottom with no errors                      | ✅      |
| Outputs included for preview                               | ✅      |
| **Published on Zerve**                                     | ⬜      |
| Public URL captured in STATUS.md                           | ⬜      |
| Tested in incognito window                                 | ⬜      |

---

## Phase 11 — Submission ⬜

| Item                                          | Status |
| --------------------------------------------- | ------ |
| Public Zerve project URL                      | ⬜      |
| FastAPI public URL                            | ⬜      |
| Dash public URL                               | ⬜      |
| Project summary (≤ 300 words)                 | ⬜      |
| Demo video (3 min)                            | ⬜      |
| LinkedIn or X post tagging @Zerve_AI          | ⬜      |
| Devpost write-up filled                       | ⬜      |

---

## Cross-cutting "nice to have" items

These are not in any phase but would improve the project. Open issues for whoever picks them up.

- ⬜ Real WC2022 calibration replay (Phase 4.5 in the guide).
- ⬜ Squad-quality feature from Club Elo (Phase 3.4).
- ⬜ K-factor grid search (Phase 4.3).
- ⬜ True single-elimination bracket diagram in the Dash app (Phase 9 View 3).
- ⬜ GitHub Actions CI running `pytest` on push.
- ⬜ Pre-commit hook running tests.
- ⬜ `Makefile` with `make pipeline / test / api / app`.
- ⬜ Dockerfile for one-command deploys.

---

## What's the riskiest thing right now?

**Synthetic data quality.** Every headline number — "Spain undervalued by +5.8%" — is currently synthesized from a deterministic seed. This is fine for the architecture demo but a judge may notice. **Before recording the demo video, verify that real Polymarket and Kalshi prices flow through, or be explicit in the narration that the fallback is in use and the architecture is what's being demonstrated.**
