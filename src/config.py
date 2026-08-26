"""
Configuration — Stock universe, sector mappings, and default parameters.
"""

# ── Stock Universe ────────────────────────────────────────────────────────────
STOCK_SECTORS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA"],
    "Finance": ["JPM", "BAC", "GS"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer Staples": ["PG", "KO", "WMT"],
    "Healthcare": ["JNJ", "UNH", "PFE"],
    "Industrial": ["CAT", "BA", "MMM"],
}

ALL_TICKERS: list[str] = [
    ticker
    for tickers in STOCK_SECTORS.values()
    for ticker in tickers
]

# ── Default Date Range ────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2019-04-01"
DEFAULT_END_DATE = "2025-03-31"

# ── Optimization Defaults ─────────────────────────────────────────────────────
DEFAULT_RISK_AVERSION = 3          # lambda in mean-variance objective
DEFAULT_MAX_POSITION_SIZE = 0.10   # max weight per individual stock
DEFAULT_MAX_SECTOR_EXPOSURE = 0.25 # max weight per sector

# ── ML Defaults ───────────────────────────────────────────────────────────────
ML_TRAIN_END_DATE = "2022-12-31"
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_LEAF = 8
RF_RANDOM_STATE = 42

# ── Risk-free Rate Proxy ──────────────────────────────────────────────────────
RISK_FREE_TICKER = "^TNX"  # 10-Year US Treasury yield
