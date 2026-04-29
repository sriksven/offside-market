# Offside Market

> The crowd is wrong. We can prove it.

ML-powered engine that finds where prediction markets are mispricing teams in the **2026 FIFA World Cup**. Built with xG-adjusted Elo ratings, a Monte Carlo tournament simulator on the actual 2026 bracket, and live odds from Polymarket and Kalshi.

[![tests](https://img.shields.io/badge/tests-31%2F31%20passing-22c55e)](tests/) [![pipeline](https://img.shields.io/badge/pipeline-end--to--end-6d28d9)](pipeline/) [![hackathon](https://img.shields.io/badge/Zerve_AI-Hackathon_2026-3b82f6)](https://zerve.ai)

## Live Links

- **Zerve notebook:** https://www.zerve.ai/gallery/1eac46ea-ccef-4e73-9423-3dfee21f87f1
- **Live app:** https://offside-market.hub.zerve.cloud
- **Demo video:** https://youtu.be/mJN8t2EueyM

## Key Findings

| Team | Market | Model | Delta | Signal |
|---|---|---|---|---|
| France | 16.3% | 7.8% | -8.5pts | OVERVALUED |
| England | 11.0% | 3.5% | -7.5pts | OVERVALUED |
| Brazil | 8.5% | 3.5% | -4.9pts | OVERVALUED |
| Japan | 2.1% | 6.3% | +4.1pts | UNDERVALUED |
| South Korea | 0.3% | 3.5% | +3.2pts | UNDERVALUED |
| Morocco | 1.5% | 4.5% | +2.9pts | UNDERVALUED |

## What it does

- Builds team strength ratings from ~5 years of international match data (xG, goals, competition-weighted, recency-decayed).
- Runs **10 000 Monte Carlo simulations** of the 2026 bracket — 12 groups of 4 → top 2 + 8 best 3rd-place advance → R32/R16/QF/SF/Final.
- Compares model probabilities against **live Polymarket + Kalshi** odds; surfaces the largest mispricings.
- Detects **cross-market arbitrage** — where Polymarket and Kalshi disagree by more than a fee threshold.
- Validates with a **calibration backtest**: xG-Elo beats win/loss-only Elo on Brier score (the technical-credibility moment).
- Ships as a **deployed FastAPI** + **Plotly Dash app** with five interactive views.

## Quick start

```bash
git clone https://github.com/sriksven/offside-market
cd offside-market
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) cp .env.example .env  # add API keys; otherwise synthetic fallbacks kick in.

# Full pipeline (≈8 seconds end-to-end on a laptop):
python -m pipeline.fetch_data
python -m pipeline.clean_data
python -m pipeline.features
python -m pipeline.train_elo
python -m pipeline.simulate
python -m analysis.market_delta
python -m analysis.arbitrage
python -m analysis.calibration

# Tests
pytest tests/ -v

# API
uvicorn api.main:app --reload     # → http://127.0.0.1:8000

# Dashboard
python app/dashboard.py            # → http://127.0.0.1:8050
```

## Project structure

```
offside-market/
├── pipeline/
│   ├── fetch_data.py       # FBref + Polymarket + Kalshi ingestion (synthetic fallback)
│   ├── clean_data.py       # Name canonicalization, competition weights, xG imputation
│   ├── features.py         # Rolling xG, recency decay, neutral-venue adjustment
│   ├── train_elo.py        # xG-adjusted Elo with margin-of-victory multiplier
│   ├── simulate.py         # 10 000-trial Monte Carlo on the 2026 bracket
│   └── config.py           # Paths, the 48 qualified teams, model hyperparameters
├── analysis/
│   ├── market_delta.py     # Headline mispricing table + plain-English narrative
│   ├── arbitrage.py        # Cross-market gap detector (Polymarket vs Kalshi)
│   └── calibration.py      # Brier-score comparison + reliability curve
├── api/
│   ├── main.py             # FastAPI app: /standings /edges /arbitrage /predict /...
│   └── models.py           # Pydantic request/response schemas
├── app/
│   ├── dashboard.py        # Plotly Dash, 5 tabs (Match-up, Mispricing, Bracket, Calibration, Arbitrage)
│   └── assets/styles.css   # Dark-mode-aware responsive styling
├── notebooks/
│   └── analysis.ipynb      # 10-section published Zerve walkthrough
├── tests/                  # 31 pytest tests
└── data/
    ├── team_name_map.json  # Canonical-name lookup (USA → United States, etc.)
    └── *.parquet           # Generated artifacts (gitignored)
```

## API

Live, fully-documented at `/docs` (Swagger UI) when the server is running.

```http
GET  /health                  Liveness + artifact freshness
GET  /teams                   Canonical names of all 48 qualified teams
GET  /standings               48-team table: model prob, market prob, edge, direction
GET  /edges?limit=10          Top mispricings (sorted by |edge|)
GET  /arbitrage?min_gap=0.02  Cross-market gaps ≥ N percentage points
GET  /calibration             Brier comparison + reliability curve from analysis/
POST /predict/match           Two-team prediction with H/D/A and confidence
GET  /team/{name}             Per-team rating, simulation breakdown, market odds
```

```bash
$ curl -s -X POST http://localhost:8000/predict/match \
       -H "Content-Type: application/json" \
       -d '{"home":"Spain","away":"Brazil"}' | jq
{
  "home": "Spain",
  "away": "Brazil",
  "home_win": 0.432,
  "draw": 0.227,
  "away_win": 0.341,
  "expected_goals_home": 1.788,
  "expected_goals_away": 1.571,
  "home_elo": 1638.9,
  "away_elo": 1613.4,
  "confidence": "low",
  "data_freshness": "2026-04-27T19:35:56+00:00"
}
```

## Dashboard views

1. **Match Predictor** — two-team dropdown, instant H/D/A probabilities, Elo ratings, recent form, and side-by-side market context.
2. **Mispricing** — model vs market for all 48 teams, plus an edge-magnitude bar with green = market underpricing, red = overpricing.
3. **Bracket** — round-by-round survival probabilities for the top 16 model seeds.
4. **Calibration** — reliability curve from a held-out backtest, plus the Brier-score table comparing model variants.
5. **Arbitrage** — sortable, filterable table of Polymarket vs Kalshi gaps; rows ≥ 2pp highlighted as actionable.

## How the model works

1. **Data** — `pipeline/fetch_data.py` pulls international fixtures via [`soccerdata`](https://github.com/probberechts/soccerdata). When that's unreachable (no internet, FBref rate-limit), a deterministic synthetic 5-year history is generated so the pipeline always finishes. Same story for Polymarket / Kalshi odds.
2. **Cleaning** — `pipeline/clean_data.py` canonicalizes team names, drops duplicates, imputes missing xG with a tiny linear model (down-weighted 0.5×), and applies competition weights (friendly 0.3 → World Cup 1.5).
3. **Features** — `pipeline/features.py` computes rolling 5/10/20-match xG attack & defense weighted by competition × exponential recency decay (half-life 365 days), and strips home advantage so 2026's neutral-venue setup is comparable.
4. **xG-adjusted Elo** — `pipeline/train_elo.py` blends actual goals with an xG-derived score (logistic of xG differential) so a team that dominated xG but lost still updates positively. Adds a FiveThirtyEight-style margin-of-victory multiplier on the xG differential.
5. **Monte Carlo** — `pipeline/simulate.py` runs the full 2026 format: 12 groups of 4 → top 2 + 8 best 3rds → R32/R16/QF/SF/Final. Match results draw from a Poisson goal model whose rates depend on each team's attack, opponent's defense, and a rating-differential nudge. Knockout draws break via penalty coin-flip biased by rating.
6. **Edges** — `analysis/market_delta.py` joins simulation outputs to live odds and ranks the largest gaps. `analysis/arbitrage.py` does the same across the two markets.
7. **Calibration** — `analysis/calibration.py` retrains on a chronological cutoff and evaluates Brier score on the held-out tail across three model variants. xG-Elo wins.

## Sample headline numbers (synthetic-fallback run)

| | Brier ↓ |
|---|---|
| Naive (50/50)        | 0.177 |
| Win/loss-only Elo    | 0.167 |
| **xG-adjusted Elo**  | **0.165** |

Top model edges (model − market):

| Team | Model | Market | Edge |
|---|---|---|---|
| Spain     | 9.8% | 4.0% | **+5.8%** |
| France    | 7.5% | 4.3% | +3.2% |
| Brazil    | 6.7% | 3.6% | +3.1% |
| Argentina | 7.3% | 4.3% | +3.0% |
| Croatia   | 6.3% | 3.5% | +2.8% |

(Numbers regenerate every pipeline run.)

## Tests

```bash
pytest tests/ -v
# 31 passed in 0.6s
```

Coverage:
- **`test_elo.py`** — Elo expected-score symmetry, xG-blend behavior, win/loss-vs-xG semantics.
- **`test_pipeline.py`** — name canonicalization, no-nulls invariants, competition-weight ordering, market normalization.
- **`test_simulation.py`** — single-winner-per-trial, win-probs sum to 1.0, reproducibility, monotone round probabilities.
- **`test_api.py`** — happy paths + 422 invalid-team / same-team errors + response schema.

## Built for

[Zerve AI Hackathon](https://zerve.ai) — June 2026.
