"""Pipeline integration tests — name canonicalization, cleaning, market normalization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline.clean_data import (
    REQUIRED_COLS,
    canonicalize,
    clean_matches,
    competition_weight,
    load_name_map,
)
from pipeline.fetch_data import synthesize_market
from pipeline.config import WORLD_CUP_2026_TEAMS


# ---------------------------------------------------------------------------
# Name canonicalization
# ---------------------------------------------------------------------------

def test_team_name_map_loads() -> None:
    name_map = load_name_map()
    assert isinstance(name_map, dict)
    assert "USA" in name_map
    assert name_map["USA"] == "United States"


def test_canonicalize_handles_aliases() -> None:
    name_map = load_name_map()
    assert canonicalize("USA", name_map) == "United States"
    assert canonicalize("Korea Republic", name_map) == "South Korea"
    assert canonicalize("Türkiye", name_map) == "Turkey"


def test_canonicalize_passes_unknown_names_through() -> None:
    """Unknown names should pass through unchanged so we don't silently drop matches."""
    name_map = load_name_map()
    assert canonicalize("Madeup-istan", name_map) == "Madeup-istan"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_no_nulls_in_required_columns(cleaned_matches: pd.DataFrame) -> None:
    for col in REQUIRED_COLS:
        assert col in cleaned_matches.columns
        assert cleaned_matches[col].isna().sum() == 0, f"nulls found in {col}"


def test_result_values_only_in_set(cleaned_matches: pd.DataFrame) -> None:
    assert set(cleaned_matches["result"].unique()).issubset({0.0, 0.5, 1.0})


def test_match_weights_positive(cleaned_matches: pd.DataFrame) -> None:
    assert (cleaned_matches["match_weight"] > 0).all()


def test_dates_sortable(cleaned_matches: pd.DataFrame) -> None:
    assert cleaned_matches["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Competition weighting
# ---------------------------------------------------------------------------

def test_competition_weights_strict_ordering() -> None:
    assert competition_weight("Friendly") < competition_weight("World Cup Qualifying")
    assert competition_weight("World Cup Qualifying") < competition_weight("FIFA World Cup")
    assert competition_weight("UEFA European Championships") > competition_weight("Friendly")


def test_competition_weight_unknown_defaults_to_one() -> None:
    assert competition_weight("Some Random League") == 1.0
    assert competition_weight(None) == 1.0


# ---------------------------------------------------------------------------
# Market odds normalization
# ---------------------------------------------------------------------------

def test_synth_market_covers_all_teams() -> None:
    snap = synthesize_market(WORLD_CUP_2026_TEAMS)
    assert set(snap.polymarket.keys()) == set(WORLD_CUP_2026_TEAMS)
    assert set(snap.kalshi.keys()) == set(WORLD_CUP_2026_TEAMS)


def test_synth_market_prices_in_unit_interval() -> None:
    snap = synthesize_market(WORLD_CUP_2026_TEAMS)
    for v in snap.polymarket.values():
        assert 0 <= v <= 1
    for v in snap.kalshi.values():
        assert 0 <= v <= 1
