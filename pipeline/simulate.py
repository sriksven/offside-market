"""Monte Carlo simulation of the 2026 FIFA World Cup bracket.

Approach:
  1. Seed 48 teams into 12 groups of 4 (drawn deterministically from rating tier).
  2. Round-robin within each group → top 2 + 8 best 3rd-placed teams advance to R32.
  3. Single-elimination from R32 → R16 → QF → SF → Final.
  4. Match outcomes use a bivariate Poisson goal model whose rates come from each
     team's attack/defense + a rating differential nudge. Penalty shootouts are
     a 50/50 coin flip when the knockout match ends level.

Outputs:
  data/simulation.parquet — per-team round-reach probabilities + win probability
  data/edges.parquet      — model_prob vs market_prob, sorted by edge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import (
    EDGES_PATH,
    ELO_HOME_ADV,
    MARKET_PATH,
    N_SIMULATIONS,
    RANDOM_SEED,
    RATINGS_PATH,
    SIMULATION_PATH,
    WORLD_CUP_2026_TEAMS,
)

log = logging.getLogger("offside.sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Match model
# ---------------------------------------------------------------------------

@dataclass
class TeamParams:
    rating: float
    attack: float
    defense: float


def _expected_goals(team: TeamParams, opponent: TeamParams, neutral: bool = True, is_home: bool = False) -> float:
    """Compute expected goals scored by ``team`` against ``opponent``."""
    rating_gap = team.rating - opponent.rating + (ELO_HOME_ADV if (is_home and not neutral) else 0.0)
    rating_factor = 1.0 + rating_gap / 600.0  # mild scaling, capped by clip below
    base = max(0.25, 0.5 * (team.attack + opponent.defense))
    return float(np.clip(base * rating_factor, 0.15, 4.0))


def simulate_match(team_a: TeamParams, team_b: TeamParams, rng: np.random.Generator,
                   knockout: bool = False, neutral: bool = True) -> tuple[int, int, str]:
    """Simulate one match. Returns (goals_a, goals_b, winner)."""
    lam_a = _expected_goals(team_a, team_b, neutral=neutral, is_home=True)
    lam_b = _expected_goals(team_b, team_a, neutral=neutral, is_home=False)
    ga = int(rng.poisson(lam_a))
    gb = int(rng.poisson(lam_b))
    if ga > gb:
        return ga, gb, "A"
    if gb > ga:
        return ga, gb, "B"
    if not knockout:
        return ga, gb, "DRAW"
    # Penalties: tilt slightly toward higher-rated team but mostly coin flip.
    p_a = 1.0 / (1.0 + 10.0 ** (-(team_a.rating - team_b.rating) / 1200.0))
    return ga, gb, "A" if rng.random() < p_a else "B"


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------

def draw_groups(teams: list[str], rng: np.random.Generator) -> list[list[str]]:
    """Snake-draw 48 teams into 12 groups of 4 across 4 rating-based pots."""
    n_groups = 12
    pot_size = 4
    pots = [teams[i * n_groups:(i + 1) * n_groups] for i in range(pot_size)]
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for pot in pots:
        order = rng.permutation(n_groups)
        for slot, gi in enumerate(order):
            groups[gi].append(pot[slot])
    return groups


def simulate_group(group: list[str], params: dict[str, TeamParams], rng: np.random.Generator) -> list[tuple[str, int, int, int]]:
    """Round-robin within a group. Returns sorted standings:
    list of (team, points, goal_diff, goals_for)."""
    pts = {t: 0 for t in group}
    gd = {t: 0 for t in group}
    gf = {t: 0 for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            ga, gb, winner = simulate_match(params[a], params[b], rng, knockout=False)
            gf[a] += ga; gf[b] += gb
            gd[a] += ga - gb; gd[b] += gb - ga
            if winner == "A":
                pts[a] += 3
            elif winner == "B":
                pts[b] += 3
            else:
                pts[a] += 1; pts[b] += 1
    standings = sorted(group, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return [(t, pts[t], gd[t], gf[t]) for t in standings]


def run_one_tournament(params: dict[str, TeamParams], rng: np.random.Generator) -> dict[str, str]:
    """Run a single tournament and return ``{team: deepest_round_reached}``."""
    teams = sorted(WORLD_CUP_2026_TEAMS, key=lambda t: -params[t].rating)
    groups = draw_groups(teams, rng)

    reached: dict[str, str] = {t: "Group" for t in teams}

    advancing: list[str] = []
    third_place_pool: list[tuple[str, int, int, int]] = []
    for group in groups:
        standings = simulate_group(group, params, rng)
        advancing.append(standings[0][0])
        advancing.append(standings[1][0])
        third_place_pool.append(standings[2])

    third_place_pool.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    advancing.extend(t for t, *_ in third_place_pool[:8])
    for t in advancing:
        reached[t] = "R32"

    def knockout_round(bracket: list[str], label: str) -> list[str]:
        winners: list[str] = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            _, _, w = simulate_match(params[a], params[b], rng, knockout=True)
            winner = a if w == "A" else b
            winners.append(winner)
            reached[winner] = label
        return winners

    rng.shuffle(advancing)
    r16 = knockout_round(advancing, "R16")
    qf = knockout_round(r16, "QF")
    sf = knockout_round(qf, "SF")
    final = knockout_round(sf, "Final")
    champion = final[0]
    reached[champion] = "Champion"
    return reached


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Champion"]


def run_simulations(ratings: pd.DataFrame, n: int = N_SIMULATIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    params: dict[str, TeamParams] = {
        row.team: TeamParams(rating=row.rating, attack=max(row.attack, 0.5), defense=max(row.defense, 0.5))
        for row in ratings.itertuples()
        if row.team in WORLD_CUP_2026_TEAMS
    }
    rng = np.random.default_rng(seed)

    counts = {t: {r: 0 for r in ROUND_ORDER} for t in params}
    for i in range(n):
        if i and i % 1000 == 0:
            log.info("  simulated %d / %d tournaments", i, n)
        result = run_one_tournament(params, rng)
        for team, deepest in result.items():
            idx = ROUND_ORDER.index(deepest)
            for r in ROUND_ORDER[: idx + 1]:
                counts[team][r] += 1

    rows = []
    for team, c in counts.items():
        row = {"team": team}
        for r in ROUND_ORDER:
            row[f"p_{r.lower()}"] = c[r] / n
        row["model_prob"] = c["Champion"] / n
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("model_prob", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def compute_edges(simulation: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = simulation.merge(market, on="team", how="left")
    df["edge"] = df["model_prob"] - df["market_prob"]
    df["edge_pct"] = (df["edge"] / df["market_prob"].replace(0, np.nan)) * 100
    df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    ratings = pd.read_parquet(RATINGS_PATH)
    market = pd.read_parquet(MARKET_PATH)
    log.info("Running %d Monte Carlo tournaments", N_SIMULATIONS)
    sim = run_simulations(ratings, n=N_SIMULATIONS, seed=RANDOM_SEED)
    sim.to_parquet(SIMULATION_PATH, index=False)
    edges = compute_edges(sim, market)
    edges.to_parquet(EDGES_PATH, index=False)

    log.info("Top 10 by championship probability:\n%s", sim.head(10).to_string(index=False))
    log.info("Top 5 model edges (undervalued):\n%s",
             edges.head(5)[["team", "model_prob", "market_prob", "edge", "edge_pct"]].to_string(index=False))
    log.info("Top 5 model fades (overvalued):\n%s",
             edges.tail(5)[["team", "model_prob", "market_prob", "edge", "edge_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
