"""Shared paths and constants for the offside-market pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

MATCHES_PATH = DATA_DIR / "matches.parquet"
RATINGS_PATH = DATA_DIR / "ratings.parquet"
MARKET_PATH = DATA_DIR / "market_odds.parquet"
SIMULATION_PATH = DATA_DIR / "simulation.parquet"
EDGES_PATH = DATA_DIR / "edges.parquet"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 48-team field for the 2026 World Cup. Order is roughly by FIFA strength tier so that
# the synthetic-data fallback produces realistic-feeling ratings.
WORLD_CUP_2026_TEAMS: list[str] = [
    "Argentina", "France", "Brazil", "Spain", "England", "Portugal",
    "Netherlands", "Germany", "Belgium", "Italy", "Croatia", "Uruguay",
    "Colombia", "Morocco", "USA", "Mexico", "Switzerland", "Denmark",
    "Senegal", "Japan", "South Korea", "Australia", "Iran", "Ecuador",
    "Canada", "Poland", "Serbia", "Wales", "Tunisia", "Cameroon",
    "Ghana", "Nigeria", "Egypt", "Algeria", "Ivory Coast", "Saudi Arabia",
    "Qatar", "Costa Rica", "Panama", "Jamaica", "Peru", "Chile",
    "Paraguay", "Norway", "Sweden", "Turkey", "Ukraine", "New Zealand",
]

# Default Elo parameters (tuned by hand; train_elo.py exposes them as CLI flags).
ELO_INIT = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 65.0
XG_WEIGHT = 0.55  # blend factor between actual goals (0) and xG (1)

# Monte Carlo defaults.
N_SIMULATIONS = 10_000
RANDOM_SEED = 42
