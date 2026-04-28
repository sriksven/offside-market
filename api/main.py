"""FastAPI service exposing the offside-market model.

Endpoints
---------
GET  /health              — liveness probe + artifact freshness
GET  /teams               — list of canonical team names (use these in /predict)
GET  /standings           — ranked teams: model prob vs market prob
GET  /edges?limit=10      — biggest mispricings (sorted by |edge|)
GET  /arbitrage           — cross-market gaps (Polymarket vs Kalshi)
GET  /calibration         — model_compare + reliability curve from analysis/
POST /predict/match       — H/D/A probabilities for any two teams
GET  /team/{name}         — full per-team breakdown
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analysis.calibration import CALIBRATION_CSV, COMPARE_CSV
from analysis.arbitrage import ARBITRAGE_CSV
from api.models import (
    ArbitrageRow,
    HealthResponse,
    MatchRequest,
    MatchResponse,
    StandingRow,
)
from pipeline.config import (
    EDGES_PATH,
    MARKET_PATH,
    RATINGS_PATH,
    SIMULATION_PATH,
    WORLD_CUP_2026_TEAMS,
)
from pipeline.simulate import TeamParams, _expected_goals

app = FastAPI(
    title="Offside Market API",
    version="0.2.0",
    description="Where the crowd is wrong about the 2026 World Cup.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data accessors (cached, lazy-loaded)
# ---------------------------------------------------------------------------

def _require(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{name} not found at {path}. Run the pipeline first: "
                "python pipeline/fetch_data.py && python pipeline/train_elo.py "
                "&& python pipeline/simulate.py"
            ),
        )
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def ratings_df() -> pd.DataFrame:
    return _require(RATINGS_PATH, "Ratings")


@lru_cache(maxsize=1)
def simulation_df() -> pd.DataFrame:
    return _require(SIMULATION_PATH, "Simulation")


@lru_cache(maxsize=1)
def market_df() -> pd.DataFrame:
    return _require(MARKET_PATH, "Market odds")


@lru_cache(maxsize=1)
def edges_df() -> pd.DataFrame:
    return _require(EDGES_PATH, "Edges")


def _data_freshness() -> str:
    """ISO timestamp of the most-recent data artifact (UTC)."""
    paths = [RATINGS_PATH, SIMULATION_PATH, MARKET_PATH, EDGES_PATH]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    if not mtimes:
        return "never"
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Match prediction (analytic Poisson, fast — no Monte Carlo per request)
# ---------------------------------------------------------------------------

def _team_params(team: str) -> TeamParams:
    df = ratings_df()
    row = df.loc[df["team"] == team]
    if row.empty:
        valid = sorted(df["team"].unique().tolist())
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Unknown team: {team!r}",
                "hint": "Try GET /teams for the canonical list.",
                "valid_teams_sample": valid[:10],
            },
        )
    r = row.iloc[0]
    return TeamParams(
        rating=float(r["rating"]),
        attack=max(float(r["attack"]), 0.5),
        defense=max(float(r["defense"]), 0.5),
    )


def _hda_probabilities(home: str, away: str, neutral: bool) -> tuple[float, float, float, float, float]:
    if home == away:
        raise HTTPException(status_code=422, detail="home and away teams must differ.")
    a = _team_params(home)
    b = _team_params(away)
    lam_a = _expected_goals(a, b, neutral=neutral, is_home=True)
    lam_b = _expected_goals(b, a, neutral=neutral, is_home=False)

    cap = 8
    pa = np.array([np.exp(-lam_a) * lam_a ** k / math.factorial(k) for k in range(cap + 1)])
    pb = np.array([np.exp(-lam_b) * lam_b ** k / math.factorial(k) for k in range(cap + 1)])
    grid = np.outer(pa, pb)
    home_win = float(np.tril(grid, -1).sum())
    away_win = float(np.triu(grid, 1).sum())
    draw = float(np.trace(grid))
    total = home_win + away_win + draw
    return home_win / total, draw / total, away_win / total, lam_a, lam_b


def _confidence(home_elo: float, away_elo: float) -> str:
    gap = abs(home_elo - away_elo)
    if gap < 80:
        return "low"
    if gap < 200:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts = {
        "ratings": RATINGS_PATH.exists(),
        "simulation": SIMULATION_PATH.exists(),
        "market": MARKET_PATH.exists(),
        "edges": EDGES_PATH.exists(),
    }
    status = "ok" if all(artifacts.values()) else "degraded"
    return HealthResponse(status=status, artifacts=artifacts, data_freshness=_data_freshness())


@app.get("/teams")
def teams() -> dict:
    return {
        "qualified_2026": WORLD_CUP_2026_TEAMS,
        "in_ratings": sorted(ratings_df()["team"].unique().tolist()),
        "data_freshness": _data_freshness(),
    }


@app.get("/standings", response_model=list[StandingRow])
def standings() -> list[StandingRow]:
    df = simulation_df().merge(market_df(), on="team", how="left")
    df = df.sort_values("model_prob", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["delta"] = df["model_prob"] - df["market_prob"].fillna(0.0)

    rows: list[StandingRow] = []
    for _, r in df.iterrows():
        delta = float(r["delta"])
        direction = (
            "undervalued_by_market" if delta > 0.005
            else "overvalued_by_market" if delta < -0.005
            else "fair"
        )
        rows.append(StandingRow(
            rank=int(r["rank"]),
            team=str(r["team"]),
            model_win_prob=round(float(r["model_prob"]), 4),
            market_win_prob=round(float(r["market_prob"]), 4) if pd.notna(r["market_prob"]) else None,
            polymarket_prob=round(float(r["polymarket_prob"]), 4) if pd.notna(r["polymarket_prob"]) else None,
            kalshi_prob=round(float(r["kalshi_prob"]), 4) if pd.notna(r["kalshi_prob"]) else None,
            delta=round(delta, 4),
            direction=direction,
        ))
    return rows


@app.get("/edges")
def edges(limit: int = 10) -> dict:
    df = edges_df().copy()
    if "edge" not in df.columns:
        raise HTTPException(status_code=503, detail="edges parquet missing 'edge' column; rerun analysis/market_delta.py")
    df["abs_edge"] = df["edge"].abs()
    df = df.sort_values("abs_edge", ascending=False).head(limit)
    cols = [c for c in ["team", "model_prob", "market_prob", "polymarket_prob",
                        "kalshi_prob", "edge", "edge_pct"] if c in df.columns]
    return {
        "undervalued": df[df["edge"] > 0][cols].round(4).to_dict(orient="records"),
        "overvalued": df[df["edge"] < 0][cols].round(4).to_dict(orient="records"),
        "data_freshness": _data_freshness(),
    }


@app.get("/arbitrage", response_model=list[ArbitrageRow])
def arbitrage(min_gap: float = 0.0) -> list[ArbitrageRow]:
    if ARBITRAGE_CSV.exists():
        df = pd.read_csv(ARBITRAGE_CSV)
    else:
        # Compute on the fly if user hasn't run analysis/arbitrage.py yet.
        from analysis.arbitrage import detect
        df = detect(market_df())

    df = df[df["gap"] >= min_gap].sort_values("gap", ascending=False)
    return [
        ArbitrageRow(
            team=str(r["team"]),
            polymarket_prob=float(r["polymarket_prob"]) if pd.notna(r["polymarket_prob"]) else None,
            kalshi_prob=float(r["kalshi_prob"]) if pd.notna(r["kalshi_prob"]) else None,
            gap=round(float(r["gap"]), 4),
            higher_market=str(r["higher_market"]),
            actionable=bool(r["actionable"]),
        )
        for _, r in df.iterrows()
    ]


@app.get("/calibration")
def calibration() -> dict:
    out: dict = {"data_freshness": _data_freshness()}
    if COMPARE_CSV.exists():
        out["model_compare"] = pd.read_csv(COMPARE_CSV).round(4).to_dict(orient="records")
    if CALIBRATION_CSV.exists():
        out["reliability"] = pd.read_csv(CALIBRATION_CSV).round(4).to_dict(orient="records")
    if not out.get("model_compare") and not out.get("reliability"):
        raise HTTPException(
            status_code=503,
            detail="Calibration not computed yet. Run: python -m analysis.calibration",
        )
    return out


@app.post("/predict/match", response_model=MatchResponse)
def predict_match(req: MatchRequest) -> MatchResponse:
    h, d, a, lam_a, lam_b = _hda_probabilities(req.home, req.away, req.neutral)
    home_p = _team_params(req.home)
    away_p = _team_params(req.away)
    return MatchResponse(
        home=req.home,
        away=req.away,
        home_win=round(h, 4),
        draw=round(d, 4),
        away_win=round(a, 4),
        expected_goals_home=round(lam_a, 3),
        expected_goals_away=round(lam_b, 3),
        home_elo=round(home_p.rating, 1),
        away_elo=round(away_p.rating, 1),
        confidence=_confidence(home_p.rating, away_p.rating),
        data_freshness=_data_freshness(),
    )


@app.get("/team/{name}")
def team(name: str) -> dict:
    sim = simulation_df()
    rat = ratings_df()
    mkt = market_df()
    if name not in set(sim["team"]):
        raise HTTPException(
            status_code=422,
            detail={"error": f"Unknown team: {name!r}", "hint": "GET /teams for the canonical list."},
        )
    s = sim.loc[sim["team"] == name].iloc[0].to_dict()
    r = rat.loc[rat["team"] == name].iloc[0].to_dict()
    m = mkt.loc[mkt["team"] == name].iloc[0].to_dict()
    out = {**r, **s, **m}
    out.pop("team", None)
    out["team"] = name
    out["edge"] = round(s["model_prob"] - (m.get("market_prob") or 0.0), 4)
    out["data_freshness"] = _data_freshness()
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
