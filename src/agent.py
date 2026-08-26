"""
AI Portfolio Advisor Agent — Gemini-powered conversational interface
that orchestrates data fetching, risk analysis, optimization, and ML
forecasting through tool-calling.
"""

import json
import os

import google.generativeai as genai

from src.config import ALL_TICKERS, STOCK_SECTORS
from src.data_fetcher import fetch_stock_data, get_monthly_prices
from src.risk_analysis import (
    compute_monthly_returns,
    compute_covariance_matrix,
    compute_risk_return_table,
)
from src.optimizer import optimize_portfolio, format_weights_summary
from src.ml_forecaster import (
    build_ml_dataset,
    train_and_predict,
    format_forecast_summary,
)


# Tool Definitions (Gemini function-calling schema)

_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "fetch_market_data",
                "description": (
                    "Fetch historical stock prices for given tickers "
                    "and date range from Yahoo Finance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock ticker symbols (e.g. ['AAPL','MSFT']). "
                                           "Leave empty for the full 18-stock universe.",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format. Default: 2019-04-01",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format. Default: 2025-03-31",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "analyze_risk_return",
                "description": (
                    "Compute annualised returns, volatility, Sharpe ratios, "
                    "and the correlation/covariance matrix for a set of stocks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tickers to analyse. Leave empty for full universe.",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "optimize_portfolio",
                "description": (
                    "Run Markowitz mean-variance optimisation with sector "
                    "and position constraints. Returns optimal weights."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Stocks to include. Leave empty for full universe.",
                        },
                        "risk_tolerance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Risk tolerance level: low, medium, or high.",
                        },
                        "max_position_size": {
                            "type": "number",
                            "description": "Max weight per stock (0-1). Default 0.10.",
                        },
                        "max_sector_exposure": {
                            "type": "number",
                            "description": "Max weight per sector (0-1). Default 0.25.",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "forecast_returns",
                "description": (
                    "Use a trained Random Forest ML model to predict "
                    "next-month returns for the stock universe. "
                    "Also returns model evaluation metrics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Stocks to forecast. Leave empty for full universe.",
                        },
                    },
                    "required": [],
                },
            },
        ]
    }
]


# Tool Implementations

def _execute_fetch_market_data(args: dict) -> str:
    tickers = args.get("tickers") or None
    start = args.get("start_date", "2019-04-01")
    end = args.get("end_date", "2025-03-31")

    prices = fetch_stock_data(tickers=tickers, start=start, end=end)
    n_rows = len(prices)
    n_cols = len(prices.columns)
    latest = prices.tail(3).to_string()

    return (
        f"Successfully fetched {n_rows} days of price data for "
        f"{n_cols} stocks ({start} to {end}).\n\n"
        f"Latest prices:\n{latest}"
    )


def _execute_analyze_risk_return(args: dict) -> str:
    tickers = args.get("tickers") or None

    prices = fetch_stock_data(tickers=tickers)
    monthly = get_monthly_prices(prices)
    returns = compute_monthly_returns(monthly)
    table = compute_risk_return_table(returns)

    # Sort by Sharpe
    table = table.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)

    summary_lines = ["**Risk/Return Analysis:**\n"]
    for _, row in table.iterrows():
        ann_ret = row["Mean Monthly Return"] * 12 * 100
        ann_vol = row["Monthly Volatility"] * (12 ** 0.5) * 100
        sharpe = row["Sharpe Ratio"]
        summary_lines.append(
            f"- **{row['Ticker']}**: "
            f"Annual Return {ann_ret:.1f}%, "
            f"Annual Volatility {ann_vol:.1f}%, "
            f"Sharpe {sharpe:.3f}"
        )

    return "\n".join(summary_lines)


def _execute_optimize_portfolio(args: dict) -> str:
    tickers = args.get("tickers") or ALL_TICKERS
    risk_tol = args.get("risk_tolerance", "medium")
    max_pos = args.get("max_position_size", 0.10)
    max_sec = args.get("max_sector_exposure", 0.25)

    prices = fetch_stock_data(tickers=tickers)
    monthly = get_monthly_prices(prices)
    returns = compute_monthly_returns(monthly)

    mean_ret = returns.mean().loc[tickers].values
    cov = compute_covariance_matrix(returns[tickers])

    weights_df = optimize_portfolio(
        mean_returns=mean_ret,
        cov_matrix=cov,
        tickers=tickers,
        risk_tolerance=risk_tol,
        max_position_size=max_pos,
        max_sector_exposure=max_sec,
    )

    return format_weights_summary(weights_df)


def _execute_forecast_returns(args: dict) -> str:
    tickers = args.get("tickers") or None

    prices = fetch_stock_data(tickers=tickers)
    monthly = get_monthly_prices(prices)
    returns = compute_monthly_returns(monthly)

    dataset = build_ml_dataset(returns, tickers=tickers)
    result = train_and_predict(dataset)

    return format_forecast_summary(result)


_TOOL_DISPATCH = {
    "fetch_market_data": _execute_fetch_market_data,
    "analyze_risk_return": _execute_analyze_risk_return,
    "optimize_portfolio": _execute_optimize_portfolio,
    "forecast_returns": _execute_forecast_returns,
}


# Agent Class

SYSTEM_PROMPT = """\
You are an AI-powered Portfolio Advisor. You help users build and analyse \
investment portfolios using quantitative finance methods.

You have access to four tools:
1. **fetch_market_data** — Download historical stock prices.
2. **analyze_risk_return** — Compute risk/return metrics and Sharpe ratios.
3. **optimize_portfolio** — Run Markowitz mean-variance optimisation.
4. **forecast_returns** — Generate ML-based return forecasts (Random Forest).

The stock universe covers 18 US large-caps across 6 sectors:
- Technology: AAPL, MSFT, NVDA
- Finance: JPM, BAC, GS
- Energy: XOM, CVX, COP
- Consumer Staples: PG, KO, WMT
- Healthcare: JNJ, UNH, PFE
- Industrial: CAT, BA, MMM

Guidelines:
- Always use the tools to provide data-backed answers.
- When users ask for portfolio recommendations, call optimize_portfolio.
- Explain results clearly and concisely using financial concepts.
- If a user's question is ambiguous, ask for clarification.
- Never invent data — only use tool results.
- Format numbers clearly (percentages, 2 decimal places).
"""


class PortfolioAgent:
    """Gemini-powered portfolio advisor with tool-calling."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "Please set the GEMINI_API_KEY environment variable "
                "or pass api_key to PortfolioAgent."
            )
        genai.configure(api_key=key)

        self.model = genai.GenerativeModel(
            model_name="gemini-3.6-flash",
            tools=_TOOLS,
            system_instruction=SYSTEM_PROMPT,
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=False)

    def run(self, user_message: str) -> str:
        """
        Send a user message and handle any tool calls the model requests.
        Returns the final text response.
        """
        response = self.chat.send_message(user_message)

        # Loop to handle tool calls (the model may chain multiple)
        while response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts

            # Check if any part is a function call
            function_calls = [p for p in parts if p.function_call.name]
            if not function_calls:
                # No tool calls — return the text
                return self._extract_text(parts)

            # Execute each tool call and collect results
            tool_responses = []
            for part in function_calls:
                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args) if part.function_call.args else {}

                if fn_name in _TOOL_DISPATCH:
                    try:
                        result = _TOOL_DISPATCH[fn_name](fn_args)
                    except Exception as e:
                        result = f"Error executing {fn_name}: {str(e)}"
                else:
                    result = f"Unknown tool: {fn_name}"

                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn_name,
                            response={"result": result},
                        )
                    )
                )

            # Send tool results back to the model
            response = self.chat.send_message(
                genai.protos.Content(parts=tool_responses)
            )

        return self._extract_text(response.candidates[0].content.parts)

    @staticmethod
    def _extract_text(parts) -> str:
        """Extract text from response parts."""
        texts = []
        for p in parts:
            if hasattr(p, "text") and p.text:
                texts.append(p.text)
        return "\n".join(texts) if texts else "I couldn't generate a response. Please try again."
