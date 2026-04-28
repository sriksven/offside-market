"""Phase 6.2 — Cross-market arbitrage detector.

For each WC2026 team we have an implied probability on Polymarket and on Kalshi.
When the two markets disagree by more than a fee threshold, that's a (theoretical)
arbitrage opportunity: buy the lower one + sell the higher one and lock in a
spread regardless of the actual outcome.

Caveats — and we surface them explicitly in the output:
  • Real arbitrage requires both venues to clear at quoted prices, with size.
  • Liquidity asymmetry between the two markets caps practical edge.
  • Each platform takes a fee; we filter gaps below 5 cents by default.

Output:
  analysis/output/arbitrage.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import MARKET_PATH

log = logging.getLogger("offside.arb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARBITRAGE_CSV = OUTPUT_DIR / "arbitrage.csv"

DEFAULT_FEE_THRESHOLD = 0.02  # 2 percentage points after fees.


def detect(market: pd.DataFrame, fee_threshold: float = DEFAULT_FEE_THRESHOLD) -> pd.DataFrame:
    df = market.copy()
    df["gap"] = (df["polymarket_prob"] - df["kalshi_prob"]).abs()
    df["higher_market"] = np.where(df["polymarket_prob"] >= df["kalshi_prob"], "polymarket", "kalshi")
    df["actionable"] = df["gap"] >= fee_threshold

    cols = ["team", "polymarket_prob", "kalshi_prob", "market_prob",
            "gap", "higher_market", "actionable"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("gap", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    market = pd.read_parquet(MARKET_PATH)
    arb = detect(market)
    arb.to_csv(ARBITRAGE_CSV, index=False)
    log.info("Wrote %s (%d rows; %d actionable)",
             ARBITRAGE_CSV, len(arb), int(arb["actionable"].sum()))
    log.info("Top 5 cross-market gaps:\n%s",
             arb.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
