"""Live prediction-market clients for Polymarket and Kalshi.

Two thin, dependency-light fetchers that return ``{team: implied_prob}`` for
the 2026 FIFA World Cup winner. Both functions are designed to be safe to
call without credentials and without a network — they swallow every failure
and return an empty dict, letting ``fetch_data.py`` fall back to its
synthetic snapshot.

Polymarket
----------
The 2026 winner market is a 60-leg neg-risk event on the public Gamma REST
API. No auth needed for read access — the credentials in ``.env`` are only
required if/when we want to place orders via the CLOB.

Kalshi
------
The matching event is ``KXMENWORLDCUP-26`` on the v2 trade API. Reading
markets requires the standard authenticated request: every call carries a
timestamp, an RSA-PSS signature over ``timestamp + method + path``, and the
caller's API key id.

Both endpoints, the slugs/tickers, and the path to the Kalshi private key are
configurable via environment variables (loaded from ``.env``) so nothing
sensitive is hard-coded.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pulls KALSHI_*, POLYMARKET_* into os.environ when present.

log = logging.getLogger("offside.markets")


# ---------------------------------------------------------------------------
# Polymarket — public Gamma REST endpoint, no auth required.
# ---------------------------------------------------------------------------

POLYMARKET_GAMMA_URL = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")
POLYMARKET_WC_SLUG = os.getenv("POLYMARKET_WC_SLUG", "2026-fifa-world-cup-winner-595")


def fetch_polymarket_world_cup_winner() -> dict[str, float]:
    """Pull the {team -> Yes-side implied prob} dict from the Polymarket event.

    The 2026 winner event is structured as a *negative-risk* group of 60
    Yes/No markets — one per team. We read each leg's ``outcomePrices`` and
    take the Yes price (already in [0, 1]) as the implied probability that
    the team wins the World Cup.
    """
    try:
        import httpx
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("httpx unavailable (%s); skipping Polymarket fetch", exc)
        return {}

    url = f"{POLYMARKET_GAMMA_URL.rstrip('/')}/events/slug/{POLYMARKET_WC_SLUG}"
    try:
        r = httpx.get(url, timeout=15)
        r.raise_for_status()
        event = r.json()
    except Exception as exc:  # pragma: no cover
        log.warning("Polymarket fetch failed (%s)", exc)
        return {}

    out: dict[str, float] = {}
    for m in event.get("markets", []) or []:
        team = m.get("groupItemTitle") or m.get("yes_sub_title")
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = None
        if not team or not prices:
            continue
        try:
            yes_price = float(prices[0])
        except (TypeError, ValueError, IndexError):
            continue
        out[team.strip()] = yes_price

    if out:
        log.info("Polymarket: %d World Cup winner markets pulled", len(out))
    return out


# ---------------------------------------------------------------------------
# Kalshi — RSA-signed v2 trade API.
# ---------------------------------------------------------------------------

KALSHI_API_BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com")
KALSHI_WC_EVENT = os.getenv("KALSHI_WC_EVENT", "KXMENWORLDCUP-26")
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")


@dataclass
class _KalshiSigner:
    """Holds the loaded RSA private key and stamps requests with v2 headers."""

    key_id: str
    private_key: object  # cryptography RSAPrivateKey

    @classmethod
    def from_env(cls) -> "_KalshiSigner | None":
        if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
            log.info("Kalshi credentials not set; skipping Kalshi fetch")
            return None
        key_path = Path(KALSHI_PRIVATE_KEY_PATH)
        if not key_path.is_absolute():
            key_path = Path(__file__).resolve().parent.parent / key_path
        if not key_path.exists():
            log.warning("Kalshi private key not found at %s; skipping Kalshi fetch", key_path)
            return None
        try:
            from cryptography.hazmat.primitives import serialization
        except Exception as exc:  # pragma: no cover - depends on env
            log.warning("cryptography unavailable (%s); skipping Kalshi fetch", exc)
            return None
        try:
            pk = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        except Exception as exc:
            log.warning("Failed to load Kalshi private key (%s); skipping Kalshi fetch", exc)
            return None
        return cls(key_id=KALSHI_API_KEY_ID, private_key=pk)

    def headers(self, method: str, path: str) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode()
        signature = self.private_key.sign(  # type: ignore[attr-defined]
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "accept": "application/json",
        }


def _kalshi_dollar_to_float(value: object) -> float | None:
    """Kalshi reports prices as strings in dollar units (e.g. '0.1680' = 16.8c)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_kalshi_world_cup_winner() -> dict[str, float]:
    """Pull the {team -> Yes-side implied prob} dict from the Kalshi event.

    Probability is the midpoint of the best Yes bid/ask in dollar units
    (Kalshi 1-cent contracts → [0, 1]). Falls back to last trade price for
    illiquid legs and skips any market with no observable price at all.
    """
    signer = _KalshiSigner.from_env()
    if signer is None:
        return {}
    try:
        import httpx
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("httpx unavailable (%s); skipping Kalshi fetch", exc)
        return {}

    path = f"/trade-api/v2/events/{KALSHI_WC_EVENT}?with_nested_markets=true"
    url = KALSHI_API_BASE.rstrip("/") + path
    try:
        r = httpx.get(url, headers=signer.headers("GET", path), timeout=15)
        r.raise_for_status()
        event = r.json().get("event", {})
    except Exception as exc:  # pragma: no cover
        log.warning("Kalshi fetch failed (%s)", exc)
        return {}

    out: dict[str, float] = {}
    for m in event.get("markets", []) or []:
        team = m.get("yes_sub_title")
        if not team:
            continue
        bid = _kalshi_dollar_to_float(m.get("yes_bid_dollars"))
        ask = _kalshi_dollar_to_float(m.get("yes_ask_dollars"))
        last = _kalshi_dollar_to_float(m.get("last_price_dollars"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            prob = (bid + ask) / 2.0
        elif last is not None and last > 0:
            prob = last
        else:
            continue
        out[team.strip()] = prob

    if out:
        log.info("Kalshi: %d World Cup winner markets pulled (event=%s)", len(out), KALSHI_WC_EVENT)
    return out


__all__ = [
    "fetch_polymarket_world_cup_winner",
    "fetch_kalshi_world_cup_winner",
]
