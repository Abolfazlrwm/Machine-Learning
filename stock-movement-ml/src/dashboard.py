"""Plotly dashboard: price chart, equity curves, fold metrics, feature importance.

All figures are combined into a single self-contained HTML file
(``reports/dashboard.html``) using the Plotly CDN so it stays small
enough to commit to GitHub.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_dashboard(
    price_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    feature_importance: pd.Series | None,
    ticker: str,
    out_path: str | Path = "reports/dashboard.html",
) -> go.Figure:
    """Build the 2x2 dashboard and save it as one HTML file."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"{ticker} — Close & moving averages",
            "Out-of-sample equity: Strategy vs Buy & Hold",
            "Walk-forward metrics per fold",
            "Feature importance (final model)",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )

    # 1. price + MAs -----------------------------------------------------
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Close"],
                             name="Close", line=dict(width=1.2, color="#1f77b4")),
                  row=1, col=1)
    if "SMA_20" in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df["SMA_20"],
                                 name="SMA 20", line=dict(width=1, dash="dot")),
                      row=1, col=1)
    if "EMA_20" in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df["EMA_20"],
                                 name="EMA 20", line=dict(width=1, dash="dash")),
                      row=1, col=1)

    # 2. equity curves ----------------------------------------------------
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["Strategy"],
                             name="Strategy", line=dict(width=1.6, color="#2ca02c")),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["BuyHold"],
                             name="Buy & Hold", line=dict(width=1.2, color="#7f7f7f")),
                  row=1, col=2)

    # 3. fold metrics ------------------------------------------------------
    metric_cols = [c for c in ("Accuracy", "F1-Score", "ROC-AUC") if c in folds_df.columns]
    fold_df = folds_df.reset_index()
    fold_df = fold_df[fold_df["Fold"] != "mean"]  # numeric folds only
    x = [str(f) for f in fold_df["Fold"]]
    for col, color in zip(metric_cols, ("#1f77b4", "#ff7f0e", "#9467bd")):
        fig.add_trace(go.Bar(x=x, y=fold_df[col], name=col, marker_color=color),
                      row=2, col=1)

    # 4. feature importance -------------------------------------------------
    if feature_importance is not None and len(feature_importance):
        imp = feature_importance.sort_values().tail(12)
        fig.add_trace(go.Bar(x=imp.values, y=list(imp.index),
                             name="importance", orientation="h",
                             marker_color="#d62728", showlegend=False),
                      row=2, col=2)

    fig.update_layout(
        height=780, width=1280,
        title_text=f"Stock Movement ML — {ticker} (walk-forward out-of-sample)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text="Fold", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return fig
