"""Unit tests for the xG-adjusted Elo update."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import ELO_INIT, ELO_K
from pipeline.train_elo import actual_score, expected_score, train


def _make_match(home="Argentina", away="Brazil", hg=2, ag=1, hxg=1.8, axg=0.9, date="2024-01-01"):
    return pd.DataFrame([{
        "date": pd.Timestamp(date),
        "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag,
        "home_xg": hxg, "away_xg": axg,
    }])


def test_expected_score_symmetry() -> None:
    """expected_score(A,B) + expected_score(B,A) ≈ 1 (with no home advantage)."""
    e_ab = expected_score(1700.0, 1500.0, home_adv=0.0)
    e_ba = expected_score(1500.0, 1700.0, home_adv=0.0)
    assert abs(e_ab + e_ba - 1.0) < 1e-9


def test_expected_score_higher_rating_wins() -> None:
    """Higher rated team has expected score > 0.5."""
    assert expected_score(1700.0, 1500.0, home_adv=0.0) > 0.5
    assert expected_score(1500.0, 1700.0, home_adv=0.0) < 0.5


def test_actual_score_pure_winloss_at_blend_zero() -> None:
    """At blend=0 (no xG), actual_score is just W/D/L."""
    assert actual_score(2, 1, 0.5, 0.5, blend=0.0) == 1.0
    assert actual_score(1, 1, 0.5, 0.5, blend=0.0) == 0.5
    assert actual_score(0, 1, 0.5, 0.5, blend=0.0) == 0.0


def test_actual_score_pure_xg_at_blend_one() -> None:
    """At blend=1, actual_score depends entirely on xG differential, not goals."""
    favored = actual_score(0, 0, 2.0, 0.5, blend=1.0)  # team dominated xG but didn't score
    disfavored = actual_score(0, 0, 0.5, 2.0, blend=1.0)
    assert favored > 0.5
    assert disfavored < 0.5


def test_train_winner_rating_goes_up() -> None:
    """Winner should gain rating, loser should lose rating."""
    matches = _make_match()
    ratings = train(matches, k=30.0)
    arg = ratings.loc[ratings["team"] == "Argentina", "rating"].iloc[0]
    bra = ratings.loc[ratings["team"] == "Brazil", "rating"].iloc[0]
    assert arg > ELO_INIT
    assert bra < ELO_INIT


def test_xg_blend_rewards_dominant_xg_despite_loss() -> None:
    """The differentiator: actual_score blends real-outcome with xG-outcome,
    so a team that dominated xG but lost on the scoreline still gets a positive
    score share (>0.5). We test the function directly so home-advantage and
    margin-of-victory multipliers don't confound the assertion."""
    s = actual_score(home_goals=0, away_goals=1, home_xg=2.5, away_xg=0.4, blend=0.7)
    assert s > 0.5, "xG-dominant loser should have actual_score > 0.5 with high blend"

    s_winloss = actual_score(home_goals=0, away_goals=1, home_xg=2.5, away_xg=0.4, blend=0.0)
    assert s_winloss == 0.0, "win/loss-only Elo would record this as a flat loss"


def test_train_chronological_order_matters() -> None:
    """Two matches in different orders → same final ratings (commutative within a chunk
    is not guaranteed in Elo, but reversing the chronological order does change them).
    Stronger property: training is deterministic and ordered by date."""
    m = pd.concat([
        _make_match(date="2024-01-01"),
        _make_match(home="Brazil", away="Argentina", hg=2, ag=0, hxg=2.1, axg=0.8, date="2024-01-15"),
    ], ignore_index=True)
    r1 = train(m).set_index("team")["rating"]
    # Shuffle input order — training should sort by date, so result is identical.
    r2 = train(m.iloc[::-1]).set_index("team")["rating"]
    assert np.allclose(r1.loc["Argentina"], r2.loc["Argentina"])
    assert np.allclose(r1.loc["Brazil"], r2.loc["Brazil"])
