"""Interactive Plotly Dash dashboard for Offside Market.

Five views, switched via tabs:
  • Match Predictor   — pick any two teams, get H/D/A probabilities (hero view)
  • Mispricing        — model vs market for all 48 teams, sorted by edge
  • Bracket           — tournament bracket with per-team championship odds
  • Calibration       — Brier-score comparison + reliability curve
  • Arbitrage         — Polymarket vs Kalshi gaps
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make sure the project root is on sys.path so ``python app/dashboard.py``
# works the same as ``python -m app.dashboard``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from analysis.arbitrage import ARBITRAGE_CSV, detect as detect_arbitrage
from analysis.calibration import CALIBRATION_CSV, COMPARE_CSV
from pipeline.config import (
    DATA_DIR,
    EDGES_PATH,
    MARKET_PATH,
    RATINGS_PATH,
    SIMULATION_PATH,
    WORLD_CUP_2026_TEAMS,
)
from pipeline.simulate import TeamParams, _expected_goals


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in (RATINGS_PATH, SIMULATION_PATH, MARKET_PATH, EDGES_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise SystemExit(
            f"Missing data files: {names}.\n"
            "Run the pipeline first:\n"
            "  python pipeline/fetch_data.py && "
            "python pipeline/clean_data.py && "
            "python pipeline/train_elo.py && "
            "python pipeline/simulate.py && "
            "python -m analysis.market_delta && "
            "python -m analysis.arbitrage"
        )
    return (
        pd.read_parquet(RATINGS_PATH),
        pd.read_parquet(SIMULATION_PATH),
        pd.read_parquet(MARKET_PATH),
        pd.read_parquet(EDGES_PATH),
    )


ratings, simulation, market, edges = _load()
team_params: dict[str, TeamParams] = {
    row.team: TeamParams(
        rating=float(row.rating),
        attack=max(float(row.attack), 0.5),
        defense=max(float(row.defense), 0.5),
    )
    for row in ratings.itertuples()
}

# Features may not exist if the user ran the legacy pipeline; tolerate missing.
FEATURES_PATH = DATA_DIR / "features.parquet"
features: pd.DataFrame | None = (
    pd.read_parquet(FEATURES_PATH) if FEATURES_PATH.exists() else None
)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

C_GREEN = "#22c55e"
C_RED = "#ef4444"
C_BLUE = "#3b82f6"
C_GRAY = "#94a3b8"
C_INK = "#0f172a"
C_PURPLE = "#6d28d9"


# ---------------------------------------------------------------------------
# View 1: Match Predictor
# ---------------------------------------------------------------------------

def predict_match(home: str, away: str, neutral: bool = True) -> dict:
    a, b = team_params[home], team_params[away]
    lam_a = _expected_goals(a, b, neutral=neutral, is_home=True)
    lam_b = _expected_goals(b, a, neutral=neutral, is_home=False)
    cap = 8
    pa = np.array([np.exp(-lam_a) * lam_a ** k / math.factorial(k) for k in range(cap + 1)])
    pb = np.array([np.exp(-lam_b) * lam_b ** k / math.factorial(k) for k in range(cap + 1)])
    grid = np.outer(pa, pb)
    h = float(np.tril(grid, -1).sum())
    d = float(np.trace(grid))
    aw = float(np.triu(grid, 1).sum())
    s = h + d + aw
    return {
        "home_win": h / s,
        "draw": d / s,
        "away_win": aw / s,
        "xg_home": lam_a,
        "xg_away": lam_b,
        "elo_home": a.rating,
        "elo_away": b.rating,
    }


def matchup_view(home: str, away: str) -> html.Div:
    if not home or not away or home == away:
        return html.Div("Pick two different teams.", className="om-card")

    p = predict_match(home, away)
    labels = [f"{home} win", "Draw", f"{away} win"]
    values = [p["home_win"], p["draw"], p["away_win"]]
    colors = [C_GREEN, C_GRAY, C_BLUE]

    fig = go.Figure()
    fig.add_bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.0%}" for v in values], textposition="outside",
    )
    fig.update_layout(
        title=f"{home} vs {away} — model probabilities",
        yaxis_title="Probability",
        yaxis=dict(tickformat=".0%", range=[0, max(values) * 1.3]),
        height=380, template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    h_form = (features.loc[features["team"] == home, "recent_form"].iloc[0]
              if features is not None and home in features["team"].values else "—")
    a_form = (features.loc[features["team"] == away, "recent_form"].iloc[0]
              if features is not None and away in features["team"].values else "—")

    market_row = market.set_index("team").reindex([home, away])
    market_home = market_row.loc[home, "market_prob"] if home in market_row.index else None
    market_away = market_row.loc[away, "market_prob"] if away in market_row.index else None

    return html.Div([
        dcc.Graph(figure=fig),
        html.Div([
            html.Div([
                html.Div(home, style={"fontWeight": 700, "fontSize": 20}),
                html.Div(f"Elo: {p['elo_home']:.0f}", style={"color": C_GRAY}),
                html.Div(f"Form: {h_form}", style={"color": C_GRAY, "fontFamily": "monospace"}),
                html.Div(f"xG (this match): {p['xg_home']:.2f}", style={"color": C_GRAY}),
                html.Div(f"WC2026 odds (model): {market_home:.1%}" if market_home is not None else "",
                         style={"color": C_INK, "marginTop": 8, "fontWeight": 600}),
            ], className="om-card", style={"flex": 1}),
            html.Div([
                html.Div(away, style={"fontWeight": 700, "fontSize": 20}),
                html.Div(f"Elo: {p['elo_away']:.0f}", style={"color": C_GRAY}),
                html.Div(f"Form: {a_form}", style={"color": C_GRAY, "fontFamily": "monospace"}),
                html.Div(f"xG (this match): {p['xg_away']:.2f}", style={"color": C_GRAY}),
                html.Div(f"WC2026 odds (market): {market_away:.1%}" if market_away is not None else "",
                         style={"color": C_INK, "marginTop": 8, "fontWeight": 600}),
            ], className="om-card", style={"flex": 1}),
        ], style={"display": "flex", "gap": 12, "marginTop": 16, "flexWrap": "wrap"}),
    ])


# ---------------------------------------------------------------------------
# View 2: Mispricing dashboard
# ---------------------------------------------------------------------------

def fig_mispricing(filter_mode: str = "all") -> go.Figure:
    df = simulation.merge(market, on="team", how="left").copy()
    df["delta"] = df["model_prob"] - df["market_prob"].fillna(0.0)
    if filter_mode == "under":
        df = df[df["delta"] > 0]
    elif filter_mode == "over":
        df = df[df["delta"] < 0]
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values("abs_delta", ascending=True).tail(30)

    colors = [C_GREEN if d >= 0 else C_RED for d in df["delta"]]
    fig = go.Figure()
    fig.add_bar(
        y=df["team"], x=df["model_prob"], orientation="h", name="Model",
        marker_color=C_PURPLE, opacity=0.85,
        hovertemplate="%{y} (model): %{x:.1%}<extra></extra>",
    )
    fig.add_bar(
        y=df["team"], x=df["market_prob"], orientation="h", name="Market",
        marker_color=C_GRAY, opacity=0.85,
        hovertemplate="%{y} (market): %{x:.1%}<extra></extra>",
    )
    fig.update_layout(
        barmode="group",
        title="Championship probability — Model vs Market",
        xaxis_title="P(win World Cup)",
        height=720, margin=dict(l=120, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def fig_edge_bars() -> go.Figure:
    df = simulation.merge(market, on="team", how="left").copy()
    df["delta"] = df["model_prob"] - df["market_prob"].fillna(0.0)
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values("abs_delta", ascending=False).head(20).sort_values("delta")
    colors = [C_RED if e < 0 else C_GREEN for e in df["delta"]]
    fig = go.Figure(go.Bar(
        y=df["team"], x=df["delta"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Model %{customdata[0]:.1%} • Market %{customdata[1]:.1%}<br>Edge: %{x:+.2%}<extra></extra>",
        customdata=df[["model_prob", "market_prob"]].values,
    ))
    fig.update_layout(
        title="Edge — green = market underpricing, red = overpricing",
        xaxis_title="Model − Market probability",
        height=560, margin=dict(l=120, r=20, t=60, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(tickformat="+.1%", zeroline=True, zerolinewidth=2, zerolinecolor=C_INK)
    return fig


# ---------------------------------------------------------------------------
# View 3: Bracket
# ---------------------------------------------------------------------------

def fig_bracket() -> go.Figure:
    """Visualization of round-by-round survivor probabilities for the top 16 seeds."""
    cols = [c for c in ["p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"] if c in simulation.columns]
    pretty = {"p_r32": "R32", "p_r16": "R16", "p_qf": "QF", "p_sf": "SF",
              "p_final": "Final", "p_champion": "Champion"}
    df = simulation.copy().sort_values("model_prob", ascending=False).head(16)

    fig = go.Figure()
    for col in cols:
        fig.add_bar(
            x=df["team"], y=df[col], name=pretty[col],
            hovertemplate="%{x}<br>" + pretty[col] + ": %{y:.1%}<extra></extra>",
        )
    fig.update_layout(
        barmode="overlay",
        title="Round-by-round probability — top 16 by model",
        yaxis_title="Probability of reaching round",
        height=560, template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=120),
        xaxis=dict(tickangle=-30),
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(opacity=0.65)
    return fig


# ---------------------------------------------------------------------------
# View 4: Calibration
# ---------------------------------------------------------------------------

def fig_calibration() -> go.Figure:
    if not CALIBRATION_CSV.exists():
        return go.Figure().update_layout(
            title="Run `python -m analysis.calibration` to populate this view.",
            template="plotly_white",
        )
    df = pd.read_csv(CALIBRATION_CSV).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["predicted_mean"], y=df["actual_rate"], mode="markers+lines",
        marker=dict(size=df["n"].clip(lower=4) ** 0.5 * 3 + 6, color=C_PURPLE),
        name="Empirical", hovertemplate="P̂=%{x:.0%}<br>Actual=%{y:.0%}<br>n=%{text}<extra></extra>",
        text=df["n"],
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(dash="dash", color=C_GRAY),
    ))
    fig.update_layout(
        title="Reliability curve — held-out international matches",
        xaxis_title="Predicted home-win probability",
        yaxis_title="Observed home-win frequency",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=520, template="plotly_white",
    )
    return fig


def calibration_view() -> html.Div:
    children: list = [dcc.Graph(figure=fig_calibration())]
    if COMPARE_CSV.exists():
        compare = pd.read_csv(COMPARE_CSV).round(4)
        children.append(html.H3("Brier score by model variant", style={"marginTop": 16}))
        children.append(dash_table.DataTable(
            data=compare.to_dict(orient="records"),
            columns=[{"name": c, "id": c} for c in compare.columns],
            style_cell={"fontFamily": "Inter, sans-serif", "padding": "8px"},
            style_header={"fontWeight": 700, "backgroundColor": "#f8fafc"},
            style_table={"overflowX": "auto"},
        ))
    else:
        children.append(html.Div("Run `python -m analysis.calibration` to compute Brier comparison.",
                                 className="om-card", style={"marginTop": 12}))
    return html.Div(children)


# ---------------------------------------------------------------------------
# View 5: Arbitrage
# ---------------------------------------------------------------------------

def arbitrage_view() -> html.Div:
    if ARBITRAGE_CSV.exists():
        df = pd.read_csv(ARBITRAGE_CSV)
    else:
        df = detect_arbitrage(market)
    df = df.sort_values("gap", ascending=False).copy()
    df["polymarket_prob"] = (df["polymarket_prob"] * 100).round(2)
    df["kalshi_prob"] = (df["kalshi_prob"] * 100).round(2)
    df["gap"] = (df["gap"] * 100).round(2)
    df = df.rename(columns={
        "polymarket_prob": "Polymarket %",
        "kalshi_prob": "Kalshi %",
        "gap": "Gap (pp)",
        "higher_market": "Higher",
        "team": "Team",
        "actionable": "≥2pp",
    })
    df = df[["Team", "Polymarket %", "Kalshi %", "Gap (pp)", "Higher", "≥2pp"]]

    return html.Div([
        html.P(
            "Cross-market gaps between Polymarket and Kalshi. Gaps ≥ 2pp survive typical fees on both venues.",
            style={"color": C_GRAY},
        ),
        dash_table.DataTable(
            data=df.to_dict(orient="records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            sort_action="native", filter_action="native", page_size=20,
            style_cell={"fontFamily": "Inter, sans-serif", "padding": "8px", "textAlign": "left"},
            style_header={"fontWeight": 700, "backgroundColor": "#f8fafc"},
            style_data_conditional=[
                {"if": {"filter_query": "{≥2pp} eq true"},
                 "backgroundColor": "rgba(34, 197, 94, 0.10)"},
            ],
            style_table={"overflowX": "auto"},
        ),
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Offside Market", suppress_callback_exceptions=True)
server = app.server  # Expose for production deployments (Zerve, gunicorn, etc.)
TEAM_OPTIONS = [{"label": t, "value": t} for t in WORLD_CUP_2026_TEAMS]

app.layout = html.Div(
    className="om-shell",
    children=[
        html.Div(className="om-hero", children=[
            html.H1("Offside Market"),
            html.P("The crowd is wrong. We can prove it.", className="om-tag"),
        ]),
        dcc.Tabs(id="tabs", value="matchup", children=[
            dcc.Tab(label="Match Predictor", value="matchup"),
            dcc.Tab(label="Mispricing", value="mispricing"),
            dcc.Tab(label="Bracket", value="bracket"),
            dcc.Tab(label="Calibration", value="calibration"),
            dcc.Tab(label="Arbitrage", value="arbitrage"),
        ], style={"marginTop": 16}),
        html.Div(id="tab-content", style={"marginTop": 24}),
    ],
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "matchup":
        return html.Div([
            html.Div(className="om-controls", style={
                "display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"
            }, children=[
                html.Label("Home", style={"fontWeight": 600, "marginRight": 8}),
                dcc.Dropdown(id="dd-home", options=TEAM_OPTIONS, value="Spain",
                             style={"width": 240, "display": "inline-block"}),
                html.Span("vs", style={"margin": "0 16px"}),
                html.Label("Away", style={"fontWeight": 600, "marginRight": 8}),
                dcc.Dropdown(id="dd-away", options=TEAM_OPTIONS, value="Brazil",
                             style={"width": 240, "display": "inline-block"}),
            ]),
            html.Div(id="matchup-container", style={"marginTop": 16}),
        ])
    if tab == "mispricing":
        return html.Div([
            html.Div([
                html.Label("Filter", style={"fontWeight": 600, "marginRight": 8}),
                dcc.RadioItems(
                    id="rad-filter",
                    options=[{"label": " All", "value": "all"},
                             {"label": " Undervalued", "value": "under"},
                             {"label": " Overvalued", "value": "over"}],
                    value="all", inline=True,
                    inputStyle={"marginLeft": 12, "marginRight": 4},
                ),
            ], style={"marginBottom": 12}),
            dcc.Graph(id="mispricing-graph"),
            dcc.Graph(figure=fig_edge_bars()),
        ])
    if tab == "bracket":
        return html.Div([
            html.P("Higher bars = greater chance of reaching that round. Top-16 model seeds shown.",
                   style={"color": C_GRAY}),
            dcc.Graph(figure=fig_bracket()),
        ])
    if tab == "calibration":
        return calibration_view()
    if tab == "arbitrage":
        return arbitrage_view()
    return html.Div("Unknown tab")


@app.callback(
    Output("matchup-container", "children"),
    Input("dd-home", "value"),
    Input("dd-away", "value"),
)
def update_matchup(home: str, away: str):
    return matchup_view(home, away)


@app.callback(
    Output("mispricing-graph", "figure"),
    Input("rad-filter", "value"),
)
def update_mispricing(filter_mode: str):
    return fig_mispricing(filter_mode or "all")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
