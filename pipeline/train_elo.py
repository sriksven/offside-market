"""xG-adjusted Elo ratings for international football.

Standard Elo would update a team's rating purely off the win/draw/loss outcome.
We extend it two ways:

  1. xG blending — instead of using actual goals to compute the "score"
     (1 / 0.5 / 0), we blend actual goals with xG. Teams that consistently
     out-perform their xG (lucky) gain less rating; teams that out-create their
     scoreline (unlucky) lose less.
  2. Home advantage — opponents' effective rating gets a +HOME bump when
     playing at home.

Outputs:
  data/ratings.parquet — team, rating, n_matches, last_match, attack, defense
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.config import (
    ELO_HOME_ADV,
    ELO_INIT,
    ELO_K,
    MATCHES_PATH,
    RATINGS_PATH,
    WORLD_CUP_2026_TEAMS,
    XG_WEIGHT,
)

log = logging.getLogger("offside.elo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def expected_score(rating_a: float, rating_b: float, home_adv: float = ELO_HOME_ADV) -> float:
    """Expected score for team A (home) vs team B (away)."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a + home_adv - rating_b) / 400.0))


def actual_score(home_goals: float, away_goals: float, home_xg: float, away_xg: float, blend: float = XG_WEIGHT) -> float:
    """Blended Elo "actual" score using both goals and xG.

    Returns a value in [0, 1] representing the home team's outcome share.
    """
    def to_score(hg: float, ag: float) -> float:
        if hg > ag:
            return 1.0
        if hg < ag:
            return 0.0
        return 0.5

    real = to_score(home_goals, away_goals)
    # Soft xG-based score: logistic of xG differential.
    xg_diff = home_xg - away_xg
    xg_score = 1.0 / (1.0 + np.exp(-1.6 * xg_diff))
    return float((1 - blend) * real + blend * xg_score)


def train(matches: pd.DataFrame, k: float = ELO_K) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]) | set(WORLD_CUP_2026_TEAMS))

    ratings = {t: ELO_INIT for t in teams}
    n_played = {t: 0 for t in teams}
    last_played = {t: pd.NaT for t in teams}
    attack = {t: 0.0 for t in teams}
    defense = {t: 0.0 for t in teams}

    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in ratings or a not in ratings:
            continue

        e_home = expected_score(ratings[h], ratings[a])
        s_home = actual_score(row["home_goals"], row["away_goals"], row["home_xg"], row["away_xg"])

        # Margin-of-victory multiplier (FiveThirtyEight-style).
        margin = abs(float(row["home_xg"]) - float(row["away_xg"]))
        mov = np.log(1.0 + margin) + 1.0

        delta = k * mov * (s_home - e_home)
        ratings[h] += delta
        ratings[a] -= delta
        n_played[h] += 1
        n_played[a] += 1
        last_played[h] = row["date"]
        last_played[a] = row["date"]

        # Track separate attack/defense rates (goals/xG per match) for the Poisson sim.
        attack[h] = 0.9 * attack[h] + 0.1 * float(row["home_xg"])
        attack[a] = 0.9 * attack[a] + 0.1 * float(row["away_xg"])
        defense[h] = 0.9 * defense[h] + 0.1 * float(row["away_xg"])
        defense[a] = 0.9 * defense[a] + 0.1 * float(row["home_xg"])

    df = pd.DataFrame({
        "team": list(ratings.keys()),
        "rating": list(ratings.values()),
        "n_matches": [n_played[t] for t in ratings],
        "last_match": [last_played[t] for t in ratings],
        "attack": [round(attack[t], 3) for t in ratings],
        "defense": [round(defense[t], 3) for t in ratings],
    })
    df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def main() -> None:
    matches = pd.read_parquet(MATCHES_PATH)
    log.info("Loaded %d matches", len(matches))
    ratings = train(matches)
    ratings.to_parquet(RATINGS_PATH, index=False)
    log.info("Wrote %s\n%s", RATINGS_PATH, ratings.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
