"""Shared pytest fixtures.

Most pipeline tests use a small deterministic synthetic match table so they run
fast and don't depend on data files being present.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_matches() -> pd.DataFrame:
    """A 20-row deterministic match table covering 4 teams."""
    rng = np.random.default_rng(0)
    teams = ["Argentina", "France", "Spain", "Brazil"]
    rows = []
    start = datetime(2022, 1, 1)
    for i in range(40):
        h, a = rng.choice(teams, size=2, replace=False)
        date = start + timedelta(days=i * 7)
        hg = int(rng.poisson(1.4))
        ag = int(rng.poisson(1.1))
        rows.append({
            "date": pd.Timestamp(date),
            "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag,
            "home_xg": round(hg + rng.normal(0, 0.3), 2),
            "away_xg": round(ag + rng.normal(0, 0.3), 2),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def cleaned_matches(tiny_matches: pd.DataFrame) -> pd.DataFrame:
    """Tiny match table run through clean_data.clean_matches."""
    from pipeline.clean_data import clean_matches
    df = tiny_matches.copy()
    df["competition"] = "Friendly"
    return clean_matches(df)
