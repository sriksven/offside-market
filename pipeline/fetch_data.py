"""Ingest international match data and live prediction-market odds.

Sources (with graceful fallback):
  - Match history: FBref via the `soccerdata` package
  - Tournament-winner odds: Polymarket + Kalshi public REST APIs

When external sources are unavailable (offline, missing keys, rate limited) the
module deterministically synthesizes a realistic 5-year history and a market
snapshot so that the rest of the pipeline runs end-to-end. Every fallback is
logged so it is obvious what happened.

Outputs:
  data/matches.parquet     — one row per international match (date, teams, goals, xG)
  data/market_odds.parquet — one row per team (polymarket_prob, kalshi_prob, market_prob)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.config import (
    MARKET_PATH,
    MATCHES_PATH,
    RANDOM_SEED,
    WORLD_CUP_2026_TEAMS,
)
from pipeline.markets import (
    fetch_kalshi_world_cup_winner,
    fetch_polymarket_world_cup_winner,
)

log = logging.getLogger("offside.fetch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Match history
# ---------------------------------------------------------------------------

def fetch_fbref_matches(years_back: int = 5) -> pd.DataFrame | None:
    """Try to pull international fixtures from FBref via soccerdata.

    Returns ``None`` if the package or remote isn't reachable.
    """
    try:
        import soccerdata as sd
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("soccerdata import failed (%s); using synthetic match history", exc)
        return None

    try:
        end = datetime.utcnow().year
        seasons = [str(y) for y in range(end - years_back, end + 1)]
        fbref = sd.FBref(leagues=["INT-World Cup", "INT-European Championships", "INT-Copa America"], seasons=seasons)
        schedule = fbref.read_schedule()
    except Exception as exc:  # pragma: no cover
        log.warning("FBref fetch failed (%s); using synthetic match history", exc)
        return None

    df = schedule.reset_index()
    keep = ["date", "home_team", "away_team", "home_score", "away_score", "home_xg", "away_xg"]
    cols = [c for c in keep if c in df.columns]
    df = df.loc[:, cols].dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    if "home_xg" not in df:
        df["home_xg"] = df["home_goals"].astype(float)
    if "away_xg" not in df:
        df["away_xg"] = df["away_goals"].astype(float)
    log.info("Pulled %d real matches from FBref", len(df))
    return df


def synthesize_matches(teams: Iterable[str], years_back: int = 5, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a deterministic, realistic-looking match history.

    Each team is assigned a latent strength on a normal distribution. Match goals
    are drawn from a Poisson with rate determined by the strength differential;
    xG is sampled around the actual goals to mimic real-world noise.
    """
    rng = np.random.default_rng(seed)
    teams = list(teams)

    # Strengths roughly follow tournament-tier order so simulation feels plausible.
    base = np.linspace(1.9, 0.6, len(teams))
    jitter = rng.normal(0, 0.08, len(teams))
    strength = dict(zip(teams, base + jitter))

    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years_back)

    rows: list[dict] = []
    n_matches_per_team = 45  # ≈9/year over 5 years
    for team in teams:
        opponents = rng.choice([t for t in teams if t != team], size=n_matches_per_team, replace=True)
        for opp in opponents:
            offset_days = int(rng.integers(0, (end - start).days))
            date = start + timedelta(days=offset_days)
            home, away = (team, opp) if rng.random() < 0.5 else (opp, team)
            mu_home = max(0.2, strength[home] + 0.20)  # home advantage
            mu_away = max(0.2, strength[away])
            hg = int(rng.poisson(mu_home))
            ag = int(rng.poisson(mu_away))
            hxg = max(0.05, rng.normal(mu_home, 0.55))
            axg = max(0.05, rng.normal(mu_away, 0.55))
            rows.append({
                "date": pd.Timestamp(date),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": round(hxg, 2),
                "away_xg": round(axg, 2),
            })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info("Synthesized %d matches across %d teams", len(df), len(teams))
    return df


def load_match_history() -> pd.DataFrame:
    real = fetch_fbref_matches()
    if real is not None and len(real) > 200:
        return real
    return synthesize_matches(WORLD_CUP_2026_TEAMS)


# ---------------------------------------------------------------------------
# Market odds
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    polymarket: dict[str, float]
    kalshi: dict[str, float]


def synthesize_market(teams: Iterable[str], seed: int = RANDOM_SEED + 1) -> MarketSnapshot:
    """Generate plausible market prices that *don't* perfectly agree with the model.

    Each market sums to ~1.05 (a 5% vig / overround typical of real books).
    Polymarket is biased toward big-name European sides, Kalshi toward US-region
    teams, which gives the edge-detection step something interesting to find.
    """
    rng = np.random.default_rng(seed)
    teams = list(teams)
    base = np.linspace(0.18, 0.005, len(teams))
    poly = base * rng.normal(1.0, 0.18, len(teams))
    kalshi = base * rng.normal(1.0, 0.22, len(teams))

    region_bonus = {"USA": 1.6, "Mexico": 1.5, "Canada": 1.4, "Brazil": 1.15}
    euro_bonus = {"England": 1.3, "France": 1.2, "Spain": 1.2, "Germany": 1.25}
    for i, t in enumerate(teams):
        kalshi[i] *= region_bonus.get(t, 1.0)
        poly[i] *= euro_bonus.get(t, 1.0)

    poly = np.clip(poly, 0.001, 0.95)
    kalshi = np.clip(kalshi, 0.001, 0.95)

    # Normalize each book to ~1.05 (5% overround) so prices read naturally as
    # implied probabilities the way real Polymarket/Kalshi WC winner markets do.
    poly = poly / poly.sum() * 1.05
    kalshi = kalshi / kalshi.sum() * 1.05

    return MarketSnapshot(
        polymarket=dict(zip(teams, poly.round(4))),
        kalshi=dict(zip(teams, kalshi.round(4))),
    )


def load_market_odds() -> pd.DataFrame:
    poly = fetch_polymarket_world_cup_winner()
    kalshi = fetch_kalshi_world_cup_winner()

    if not poly and not kalshi:
        snap = synthesize_market(WORLD_CUP_2026_TEAMS)
        poly, kalshi = snap.polymarket, snap.kalshi
        log.info("Using synthetic market snapshot")
    elif not poly:
        log.warning("Polymarket returned no prices; only Kalshi data will populate the snapshot")
    elif not kalshi:
        log.warning("Kalshi returned no prices; only Polymarket data will populate the snapshot")

    rows = []
    for team in WORLD_CUP_2026_TEAMS:
        p = poly.get(team)
        k = kalshi.get(team)
        # De-vig by averaging available books (cheap but works for an MVP).
        available = [v for v in (p, k) if v is not None]
        market = float(np.mean(available)) if available else np.nan
        rows.append({
            "team": team,
            "polymarket_prob": p,
            "kalshi_prob": k,
            "market_prob": market,
        })
    df = pd.DataFrame(rows)
    # Re-normalize so probabilities sum to 1 (they should, after de-vig).
    total = df["market_prob"].sum(skipna=True)
    if total and not np.isnan(total):
        df["market_prob"] = df["market_prob"] / total
    return df


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    matches = load_match_history()
    matches.to_parquet(MATCHES_PATH, index=False)
    log.info("Wrote %s (%d rows)", MATCHES_PATH, len(matches))

    market = load_market_odds()
    market.to_parquet(MARKET_PATH, index=False)
    log.info("Wrote %s (%d rows)", MARKET_PATH, len(market))


if __name__ == "__main__":
    main()
