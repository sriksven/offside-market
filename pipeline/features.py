"""Phase 3 — Feature engineering.

Per-team form features computed from the cleaned match log:
  • Rolling 5/10/20-match xG attack and defense (weighted by match_weight)
  • Recency-decayed (exponential, half-life 365 days) means
  • Neutral-venue adjusted xG (strips home advantage)
  • Goal-differential per match

These features are not strictly required by the Elo model — the Elo update
naturally bakes in recency — but they make the dashboard match-up view far
more informative ("recent form: last 5 results") and feed into the Poisson
goal model used inside the simulator.

Output:
  data/features.parquet — one row per team.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR, WORLD_CUP_2026_TEAMS

log = logging.getLogger("offside.features")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

CLEAN_PATH = DATA_DIR / "matches_clean.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"

DECAY_HALFLIFE_DAYS = 365.0
HOME_ADV_GOALS = 0.18  # xG bump teams typically enjoy at home in internationals.


def _team_long_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape match table into a per-team-per-match long-form table."""
    a = matches[["date", "team_a", "team_b", "xg_a", "xg_b", "goals_a", "goals_b", "match_weight"]].copy()
    a.columns = ["date", "team", "opponent", "xg_for", "xg_against", "goals_for", "goals_against", "match_weight"]
    a["venue"] = "home"

    b = matches[["date", "team_b", "team_a", "xg_b", "xg_a", "goals_b", "goals_a", "match_weight"]].copy()
    b.columns = ["date", "team", "opponent", "xg_for", "xg_against", "goals_for", "goals_against", "match_weight"]
    b["venue"] = "away"

    long = pd.concat([a, b], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return long


def _decay_weight(match_date: pd.Timestamp, ref: datetime, halflife: float = DECAY_HALFLIFE_DAYS) -> float:
    days = (ref - match_date.to_pydatetime()).days
    if days <= 0:
        return 1.0
    return math.exp(-math.log(2) * days / halflife)


def _neutral_adjust(xg_for: float, xg_against: float, venue: str) -> tuple[float, float]:
    """Strip home/away advantage so values are comparable to neutral matches."""
    if venue == "home":
        return xg_for - HOME_ADV_GOALS / 2, xg_against + HOME_ADV_GOALS / 2
    if venue == "away":
        return xg_for + HOME_ADV_GOALS / 2, xg_against - HOME_ADV_GOALS / 2
    return xg_for, xg_against


def build_features(matches: pd.DataFrame, ref: datetime | None = None) -> pd.DataFrame:
    if ref is None:
        ref = datetime.utcnow()

    long = _team_long_form(matches)
    adj = long.apply(lambda r: _neutral_adjust(r["xg_for"], r["xg_against"], r["venue"]),
                     axis=1, result_type="expand")
    long["xg_for_neutral"], long["xg_against_neutral"] = adj[0], adj[1]
    long["decay"] = long["date"].apply(lambda d: _decay_weight(d, ref))
    long["effective_weight"] = long["match_weight"] * long["decay"]

    rows: list[dict] = []
    for team, g in long.groupby("team"):
        g = g.sort_values("date")
        last5 = g.tail(5)
        last10 = g.tail(10)
        last20 = g.tail(20)

        def w_mean(df: pd.DataFrame, col: str) -> float:
            w = df["effective_weight"].to_numpy()
            v = df[col].to_numpy()
            if w.sum() == 0:
                return float("nan")
            return float(np.average(v, weights=w))

        recent_results = (last5["goals_for"] - last5["goals_against"]).apply(
            lambda diff: "W" if diff > 0 else ("L" if diff < 0 else "D"))

        rows.append({
            "team": team,
            "n_matches": int(len(g)),
            "xg_attack_5":  w_mean(last5, "xg_for_neutral"),
            "xg_attack_10": w_mean(last10, "xg_for_neutral"),
            "xg_attack_20": w_mean(last20, "xg_for_neutral"),
            "xg_defense_5":  w_mean(last5, "xg_against_neutral"),
            "xg_defense_10": w_mean(last10, "xg_against_neutral"),
            "xg_defense_20": w_mean(last20, "xg_against_neutral"),
            "goal_diff_per_match": float((g["goals_for"] - g["goals_against"]).mean()),
            "recent_form": "".join(recent_results.tolist()[-5:]),
            "last_match": g["date"].max(),
        })

    df = pd.DataFrame(rows)
    df = df[df["team"].isin(WORLD_CUP_2026_TEAMS) | df["team"].notna()].reset_index(drop=True)
    return df.sort_values("team").reset_index(drop=True)


def main() -> None:
    if not CLEAN_PATH.exists():
        raise SystemExit(f"Run pipeline/clean_data.py first — {CLEAN_PATH} missing.")
    matches = pd.read_parquet(CLEAN_PATH)
    feats = build_features(matches)
    feats.to_parquet(FEATURES_PATH, index=False)
    log.info("Wrote %s (%d teams)", FEATURES_PATH, len(feats))
    qualified = feats[feats["team"].isin(WORLD_CUP_2026_TEAMS)]
    log.info("WC2026 sample form:\n%s",
             qualified.head(10)[["team", "n_matches", "xg_attack_10", "xg_defense_10", "recent_form"]]
             .to_string(index=False))


if __name__ == "__main__":
    main()
