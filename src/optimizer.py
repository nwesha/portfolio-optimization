"""
Portfolio Optimizer — Constrained Markowitz mean-variance optimization.
"""

import numpy as np
import pandas as pd
import cvxpy as cp

from src.config import (
    STOCK_SECTORS,
    ALL_TICKERS,
    DEFAULT_RISK_AVERSION,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MAX_SECTOR_EXPOSURE,
)


# ── Risk-tolerance → lambda mapping ──────────────────────────────────────────
RISK_TOLERANCE_MAP = {
    "low": 6,        # high lambda = very risk averse
    "medium": 3,     # moderate
    "high": 1,       # low lambda = risk seeking
}


def optimize_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str] | None = None,
    risk_tolerance: str = "medium",
    max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
    max_sector_exposure: float = DEFAULT_MAX_SECTOR_EXPOSURE,
) -> pd.DataFrame:
    """
    Solve the constrained Markowitz mean-variance optimisation.

    Parameters
    ----------
    mean_returns : np.ndarray
        Expected (mean) monthly return per asset.
    cov_matrix : np.ndarray
        Covariance matrix of monthly returns (symmetrised).
    tickers : list[str] | None
        Ordered list of ticker symbols matching mean_returns.
    risk_tolerance : str
        One of "low", "medium", "high".
    max_position_size : float
        Maximum weight for any single stock (0-1).
    max_sector_exposure : float
        Maximum total weight for any sector (0-1).

    Returns
    -------
    pd.DataFrame
        Columns: Ticker, Weight, Sector — sorted by weight descending.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    lambda_val = RISK_TOLERANCE_MAP.get(risk_tolerance, DEFAULT_RISK_AVERSION)
    n = len(tickers)

    w = cp.Variable(n)

    # Core constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_position_size,
    ]

    # Sector concentration limits
    for _sector, sector_tickers in STOCK_SECTORS.items():
        indices = [tickers.index(t) for t in sector_tickers if t in tickers]
        if indices:
            constraints.append(cp.sum(w[indices]) <= max_sector_exposure)

    # Objective: maximise  μᵀw − (λ/2) wᵀΣw
    objective = cp.Maximize(
        mean_returns @ w
        - 0.5 * lambda_val * cp.quad_form(w, cp.psd_wrap(cov_matrix))
    )

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Optimisation failed: {problem.status}")

    optimal_weights = np.maximum(w.value, 0)
    optimal_weights = optimal_weights / optimal_weights.sum()

    # Build a ticker → sector lookup
    ticker_to_sector = {
        ticker: sector
        for sector, stickers in STOCK_SECTORS.items()
        for ticker in stickers
    }

    result = pd.DataFrame({
        "Ticker": tickers,
        "Weight": optimal_weights,
    })
    result["Sector"] = result["Ticker"].map(ticker_to_sector)
    result = result.sort_values("Weight", ascending=False).reset_index(drop=True)

    return result


def format_weights_summary(weights_df: pd.DataFrame) -> str:
    """
    Return a human-readable summary of portfolio weights.
    Filters out near-zero positions for clarity.
    """
    lines = ["**Optimal Portfolio Weights:**\n"]
    significant = weights_df[weights_df["Weight"] > 0.005]

    for _, row in significant.iterrows():
        pct = row["Weight"] * 100
        lines.append(f"- **{row['Ticker']}** ({row['Sector']}): {pct:.1f}%")

    # Sector totals
    lines.append("\n**Sector Exposure:**")
    sector_totals = significant.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
    for sector, weight in sector_totals.items():
        lines.append(f"- {sector}: {weight * 100:.1f}%")

    return "\n".join(lines)
