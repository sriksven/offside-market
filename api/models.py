"""Pydantic schemas shared across API routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    home: str = Field(..., examples=["Spain"])
    away: str = Field(..., examples=["Brazil"])
    neutral: bool = Field(True, description="True for tournament-neutral venues (default for WC2026).")


class MatchResponse(BaseModel):
    home: str
    away: str
    home_win: float
    draw: float
    away_win: float
    expected_goals_home: float
    expected_goals_away: float
    home_elo: float
    away_elo: float
    confidence: Literal["low", "medium", "high"]
    data_freshness: str


class StandingRow(BaseModel):
    rank: int
    team: str
    model_win_prob: float
    market_win_prob: float | None
    polymarket_prob: float | None
    kalshi_prob: float | None
    delta: float
    direction: Literal["undervalued_by_market", "overvalued_by_market", "fair"]


class ArbitrageRow(BaseModel):
    team: str
    polymarket_prob: float | None
    kalshi_prob: float | None
    gap: float
    higher_market: Literal["polymarket", "kalshi"]
    actionable: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    artifacts: dict[str, bool]
    data_freshness: str
