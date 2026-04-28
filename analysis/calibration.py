"""Phase 6.3 — Calibration analysis.

Two questions:

1. Is our model better than win/loss-only Elo on held-out matches?
   We retrain on data older than a cutoff and evaluate Brier score on the held-out
   tail. The xG-blended model should clear the win/loss baseline.

2. How well-calibrated are the prediction markets, historically?
   We synthesize realistic-feeling 2022 Polymarket-style winner odds and compare
   to actual outcomes. (Real 2022 odds can be plumbed in via FBref scrapes — the
   structure here is set up to drop those in.)

Outputs:
  analysis/output/calibration.csv  — per-bin actual vs predicted frequency
  analysis/output/model_compare.csv — Brier scores across model variants
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR
from pipeline.train_elo import expected_score, train

log = logging.getLogger("offside.calib")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_CSV = OUTPUT_DIR / "calibration.csv"
COMPARE_CSV = OUTPUT_DIR / "model_compare.csv"

CLEAN_PATH = DATA_DIR / "matches_clean.parquet"


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Standard Brier for binary outcomes."""
    return float(np.mean((predicted - actual) ** 2))


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def split_train_test(matches: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = matches.sort_values("date")
    train_df = matches[matches["date"] < cutoff].copy()
    test_df = matches[matches["date"] >= cutoff].copy()
    return train_df, test_df


def _matches_to_train_format(clean: pd.DataFrame) -> pd.DataFrame:
    """Convert ``team_a`` / ``team_b`` schema to the ``home_team`` / ``away_team``
    format that ``pipeline.train_elo.train`` expects."""
    out = clean.rename(columns={
        "team_a": "home_team",
        "team_b": "away_team",
        "xg_a": "home_xg",
        "xg_b": "away_xg",
        "goals_a": "home_goals",
        "goals_b": "away_goals",
    })
    return out


def evaluate(matches: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    if cutoff is None:
        # Evaluate on the most-recent 12 months of data.
        cutoff = matches["date"].max() - pd.Timedelta(days=365)

    train_clean, test_clean = split_train_test(matches, cutoff)
    if len(test_clean) < 30:
        log.warning("Only %d test matches; results may be noisy", len(test_clean))

    train_input = _matches_to_train_format(train_clean)
    ratings_full = train(train_input)  # xG-blended (default)
    rmap = dict(zip(ratings_full["team"], ratings_full["rating"]))

    # Naive baseline: every team is rated equally → expected score is always 0.5.
    preds_xg, preds_naive, preds_winloss, actuals = [], [], [], []

    # For "winloss-only" Elo we re-train with blend=0 by hand.
    from pipeline.train_elo import actual_score, expected_score as _exp
    from pipeline.config import ELO_INIT, ELO_K, ELO_HOME_ADV
    wl_ratings: dict[str, float] = {t: ELO_INIT for t in rmap}
    train_input = train_input.sort_values("date").reset_index(drop=True)
    for _, row in train_input.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in wl_ratings or a not in wl_ratings:
            continue
        e_home = _exp(wl_ratings[h], wl_ratings[a])
        s_home = actual_score(row["home_goals"], row["away_goals"],
                              row["home_xg"], row["away_xg"], blend=0.0)
        delta = ELO_K * (s_home - e_home)
        wl_ratings[h] += delta
        wl_ratings[a] -= delta

    for _, row in test_clean.iterrows():
        a, b = row["team_a"], row["team_b"]
        # Use Elo win-share semantics: win=1.0, draw=0.5, loss=0.0.
        actual_outcome = float(row["result"])
        preds_naive.append(0.5)
        preds_xg.append(expected_score(rmap.get(a, ELO_INIT), rmap.get(b, ELO_INIT)))
        preds_winloss.append(expected_score(wl_ratings.get(a, ELO_INIT), wl_ratings.get(b, ELO_INIT)))
        actuals.append(actual_outcome)

    out = pd.DataFrame([
        {"model": "naive (50/50)", "brier": brier(np.array(preds_naive), np.array(actuals)), "n": len(actuals)},
        {"model": "win/loss elo",   "brier": brier(np.array(preds_winloss), np.array(actuals)), "n": len(actuals)},
        {"model": "xg-adjusted elo", "brier": brier(np.array(preds_xg), np.array(actuals)), "n": len(actuals)},
    ])
    return out


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------

def reliability_curve(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    binned = np.digitize(predicted, bins) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = binned == i
        if mask.sum() == 0:
            rows.append({"bin_lower": bins[i], "bin_upper": bins[i + 1],
                         "n": 0, "predicted_mean": np.nan, "actual_rate": np.nan})
            continue
        rows.append({
            "bin_lower": bins[i],
            "bin_upper": bins[i + 1],
            "n": int(mask.sum()),
            "predicted_mean": float(predicted[mask].mean()),
            "actual_rate": float(actual[mask].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not CLEAN_PATH.exists():
        raise SystemExit(f"Run pipeline/clean_data.py first — {CLEAN_PATH} missing.")
    matches = pd.read_parquet(CLEAN_PATH)

    log.info("Comparing model variants on held-out tail of data")
    compare = evaluate(matches)
    compare.to_csv(COMPARE_CSV, index=False)
    log.info("Brier comparison:\n%s", compare.to_string(index=False))

    cutoff = matches["date"].max() - pd.Timedelta(days=365)
    train_clean, test_clean = split_train_test(matches, cutoff)
    ratings_df = train(_matches_to_train_format(train_clean))
    rmap = dict(zip(ratings_df["team"], ratings_df["rating"]))
    preds = np.array([expected_score(rmap.get(r["team_a"], 1500.0),
                                     rmap.get(r["team_b"], 1500.0))
                      for _, r in test_clean.iterrows()])
    # Reliability curve uses Elo win-share semantics (1 / 0.5 / 0) so the
    # interpretation matches what predicted is forecasting: expected points share.
    actuals = test_clean["result"].astype(float).to_numpy()
    rel = reliability_curve(preds, actuals)
    rel.to_csv(CALIBRATION_CSV, index=False)
    log.info("Wrote %s\n%s", CALIBRATION_CSV, rel.to_string(index=False))


if __name__ == "__main__":
    main()
