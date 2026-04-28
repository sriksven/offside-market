"""FastAPI endpoint tests using TestClient.

We require pipeline data to exist; otherwise the API correctly returns 503.
The tests skip themselves in that case rather than failing CI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from pipeline.config import EDGES_PATH, MARKET_PATH, RATINGS_PATH, SIMULATION_PATH

PIPELINE_READY = all(p.exists() for p in (RATINGS_PATH, SIMULATION_PATH, MARKET_PATH, EDGES_PATH))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_200(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "data_freshness" in body


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_teams_endpoint(client: TestClient) -> None:
    r = client.get("/teams")
    assert r.status_code == 200
    body = r.json()
    assert len(body["qualified_2026"]) == 48


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_standings_returns_48(client: TestClient) -> None:
    r = client.get("/standings")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 48
    total = sum(row["model_win_prob"] for row in rows)
    assert abs(total - 1.0) < 0.05


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_predict_match_happy_path(client: TestClient) -> None:
    r = client.post("/predict/match", json={"home": "Spain", "away": "Brazil"})
    assert r.status_code == 200
    body = r.json()
    assert abs(body["home_win"] + body["draw"] + body["away_win"] - 1.0) < 0.01
    assert body["confidence"] in {"low", "medium", "high"}


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_predict_match_invalid_team_returns_422(client: TestClient) -> None:
    r = client.post("/predict/match", json={"home": "Atlantis", "away": "Brazil"})
    assert r.status_code == 422


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_predict_match_same_team_returns_422(client: TestClient) -> None:
    r = client.post("/predict/match", json={"home": "Brazil", "away": "Brazil"})
    assert r.status_code == 422


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_arbitrage_endpoint(client: TestClient) -> None:
    r = client.get("/arbitrage")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    if rows:
        assert "gap" in rows[0]
        assert rows[0]["gap"] >= 0


@pytest.mark.skipif(not PIPELINE_READY, reason="pipeline outputs missing")
def test_edges_endpoint(client: TestClient) -> None:
    r = client.get("/edges?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "undervalued" in body
    assert "overvalued" in body
