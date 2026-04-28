"""Distributional tests for the Monte Carlo tournament simulator.

These run with a much smaller N than production (10 000) to keep CI fast — but
the structural invariants we check don't depend on N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import WORLD_CUP_2026_TEAMS
from pipeline.simulate import (
    TeamParams,
    run_one_tournament,
    run_simulations,
)


@pytest.fixture
def synth_ratings() -> pd.DataFrame:
    """Make-believe ratings spanning a realistic range so the sim is non-trivial."""
    rng = np.random.default_rng(0)
    base = np.linspace(1900, 1400, len(WORLD_CUP_2026_TEAMS))
    base = base + rng.normal(0, 25, len(WORLD_CUP_2026_TEAMS))
    return pd.DataFrame({
        "team": WORLD_CUP_2026_TEAMS,
        "rating": base,
        "n_matches": [40] * len(WORLD_CUP_2026_TEAMS),
        "last_match": [pd.Timestamp("2025-01-01")] * len(WORLD_CUP_2026_TEAMS),
        "attack": np.abs(rng.normal(1.4, 0.2, len(WORLD_CUP_2026_TEAMS))),
        "defense": np.abs(rng.normal(1.0, 0.2, len(WORLD_CUP_2026_TEAMS))),
    })


def test_single_winner_per_tournament(synth_ratings: pd.DataFrame) -> None:
    params = {
        row.team: TeamParams(rating=float(row.rating),
                             attack=max(float(row.attack), 0.5),
                             defense=max(float(row.defense), 0.5))
        for row in synth_ratings.itertuples()
    }
    rng = np.random.default_rng(1)
    for _ in range(20):
        result = run_one_tournament(params, rng)
        champions = [t for t, deepest in result.items() if deepest == "Champion"]
        assert len(champions) == 1


def test_win_probs_sum_to_one(synth_ratings: pd.DataFrame) -> None:
    sim = run_simulations(synth_ratings, n=200, seed=42)
    total = sim["model_prob"].sum()
    assert abs(total - 1.0) < 0.01, f"expected probs to sum to 1, got {total:.4f}"


def test_reproducibility(synth_ratings: pd.DataFrame) -> None:
    a = run_simulations(synth_ratings, n=100, seed=7)
    b = run_simulations(synth_ratings, n=100, seed=7)
    pd.testing.assert_frame_equal(
        a.sort_values("team").reset_index(drop=True),
        b.sort_values("team").reset_index(drop=True),
    )


def test_strong_team_favored_over_weak_team(synth_ratings: pd.DataFrame) -> None:
    sim = run_simulations(synth_ratings, n=300, seed=42).sort_values("model_prob", ascending=False)
    top = sim.iloc[0]
    bottom = sim.iloc[-1]
    assert top["model_prob"] > bottom["model_prob"]
    # Top seed should clear naive uniform 1/48 baseline by a wide margin.
    assert top["model_prob"] > 1.0 / 48 * 2


def test_round_probabilities_are_monotone(synth_ratings: pd.DataFrame) -> None:
    """For every team, P(R32) >= P(R16) >= P(QF) >= ... >= P(Champion)."""
    sim = run_simulations(synth_ratings, n=200, seed=42)
    for _, r in sim.iterrows():
        assert r["p_r32"] >= r["p_r16"] - 1e-9
        assert r["p_r16"] >= r["p_qf"] - 1e-9
        assert r["p_qf"] >= r["p_sf"] - 1e-9
        assert r["p_sf"] >= r["p_final"] - 1e-9
        assert r["p_final"] >= r["p_champion"] - 1e-9
