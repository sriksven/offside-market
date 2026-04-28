"""Phase 6 — Market delta.

Builds the headline "the crowd is wrong" table:

  team | model_win_prob | market_win_prob | delta_vs_polymarket | delta_vs_kalshi

Plus a short narrative file explaining the top 5 over/undervalued teams.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    DATA_DIR,
    EDGES_PATH,
    MARKET_PATH,
    SIMULATION_PATH,
)

log = logging.getLogger("offside.delta")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DELTA_CSV = OUTPUT_DIR / "market_delta.csv"
NARRATIVE_PATH = OUTPUT_DIR / "narrative.md"


def build_delta_table() -> pd.DataFrame:
    sim = pd.read_parquet(SIMULATION_PATH)
    market = pd.read_parquet(MARKET_PATH)
    df = sim.merge(market, on="team", how="left")

    df = df.rename(columns={"model_prob": "model_win_prob"})
    df["delta_vs_polymarket"] = df["model_win_prob"] - df["polymarket_prob"]
    df["delta_vs_kalshi"] = df["model_win_prob"] - df["kalshi_prob"]
    df["delta_vs_blend"] = df["model_win_prob"] - df["market_prob"]

    cols = [
        "rank", "team",
        "model_win_prob", "p_final", "p_sf", "p_qf", "p_r16", "p_r32",
        "polymarket_prob", "kalshi_prob", "market_prob",
        "delta_vs_polymarket", "delta_vs_kalshi", "delta_vs_blend",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("delta_vs_blend", ascending=False).reset_index(drop=True)
    return df


def write_narrative(df: pd.DataFrame, path: Path = NARRATIVE_PATH) -> None:
    top = df.head(5)
    bottom = df.tail(5).iloc[::-1]

    lines: list[str] = ["# Offside Market — Headline Findings", ""]
    lines.append("Generated automatically from `analysis/market_delta.py`. Numbers refresh whenever you re-run the pipeline.\n")

    lines.append("## Most undervalued by the market\n")
    for _, r in top.iterrows():
        lines.append(
            f"- **{r['team']}** — model {r['model_win_prob']:.1%} vs blended-market {r['market_prob']:.1%} "
            f"(Polymarket {r['polymarket_prob']:.1%}, Kalshi {r['kalshi_prob']:.1%}). "
            f"Edge: **+{r['delta_vs_blend']:.1%}**."
        )
    lines.append("")

    lines.append("## Most overvalued by the market\n")
    for _, r in bottom.iterrows():
        lines.append(
            f"- **{r['team']}** — model {r['model_win_prob']:.1%} vs blended-market {r['market_prob']:.1%} "
            f"(Polymarket {r['polymarket_prob']:.1%}, Kalshi {r['kalshi_prob']:.1%}). "
            f"Edge: **{r['delta_vs_blend']:+.1%}**."
        )
    lines.append("")

    lines.append("## How to read this\n")
    lines.append(
        "Positive edges mean the model thinks the market is too bearish on the team. "
        "Negative edges mean the model thinks the market is too bullish. "
        "Edges greater than ~3% on long-tail markets are large enough to clear typical "
        "fees on Polymarket/Kalshi. xG-adjusted Elo specifically targets teams whose "
        "scoreline-based perception lags their underlying performance — see `calibration.py` "
        "for the technical credibility check on WC2022.\n"
    )
    path.write_text("\n".join(lines))


def main() -> None:
    df = build_delta_table()
    df.to_csv(DELTA_CSV, index=False)
    log.info("Wrote %s (%d rows)", DELTA_CSV, len(df))
    write_narrative(df)
    log.info("Wrote %s", NARRATIVE_PATH)
    log.info("Top 5 undervalued:\n%s",
             df.head(5)[["team", "model_win_prob", "market_prob", "delta_vs_blend"]]
             .to_string(index=False))
    log.info("Top 5 overvalued:\n%s",
             df.tail(5).iloc[::-1][["team", "model_win_prob", "market_prob", "delta_vs_blend"]]
             .to_string(index=False))

    # Also refresh the parquet edges file used by the API, so this module is a
    # drop-in replacement for the simulate-side edge computation when re-run alone.
    df.rename(columns={"model_win_prob": "model_prob",
                       "delta_vs_blend": "edge"}).to_parquet(EDGES_PATH, index=False)
    log.info("Refreshed %s", EDGES_PATH)


if __name__ == "__main__":
    main()
