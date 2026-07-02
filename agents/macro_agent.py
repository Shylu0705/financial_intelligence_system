import os

import requests
import yfinance as yf
from dotenv import load_dotenv

from core.state import FinancialState

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",      # Federal Funds Effective Rate
    "cpi":            "CPIAUCSL",      # Consumer Price Index (All Urban)
    "treasury_10y":   "DGS10",         # 10-Year Treasury Constant Maturity Rate
}

# Market benchmarks
BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
}


def _fetch_fred_series(series_id: str, limit: int = 1) -> list[float]:
    """
    Fetches the most recent `limit` observations for a FRED series.
    Returns a list of floats (most recent first), empty list on failure.
    """
    try:
        response = requests.get(
            FRED_BASE_URL,
            params={
                "series_id":         series_id,
                "api_key":           FRED_API_KEY,
                "file_type":         "json",
                "sort_order":        "desc",
                "limit":             limit,
                "observation_start": "2020-01-01",
            },
            timeout=10,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
        values = []
        for obs in observations:
            v = obs.get("value", ".")
            if v != ".":
                values.append(float(v))
        return values
    except Exception as e:
        print(f"[Macro Agent] FRED fetch error for {series_id}: {e}")
        return []


def _fetch_cpi_yoy() -> float | None:
    """
    Fetches recent CPIAUCSL observations and computes YoY % change.
    Requests 18 observations to ensure we have 13 valid non-dot values
    even if FRED returns dots for unreleased/pending months.
    """
    values = _fetch_fred_series(FRED_SERIES["cpi"], limit=18)
    if len(values) < 13:
        print(f"[Macro Agent] Not enough CPI observations ({len(values)}) to compute YoY.")
        return None
    current  = values[0]   # most recent valid month
    year_ago = values[12]  # same month last year
    return round(((current - year_ago) / year_ago) * 100, 2)


def _benchmark_return(ticker: str, period: str = "6mo") -> float | None:
    """Fetches 3-month return for a benchmark ETF (SPY or QQQ)."""
    try:
        data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if data.empty or len(data) < 63:
            return None
        start_price = data["Close"].iloc[-63]
        end_price   = data["Close"].iloc[-1]
        return round(((end_price - start_price) / start_price) * 100, 2)
    except Exception as e:
        print(f"[Macro Agent] Benchmark fetch error for {ticker}: {e}")
        return None


def _rate_environment(fed_rate: float | None, treasury_10y: float | None) -> str:
    """Classifies the current interest rate environment in plain language."""
    if fed_rate is None:
        return "Unknown"
    if fed_rate >= 4.5:
        return "High rates — expensive borrowing, headwind for high-P/E growth stocks"
    if fed_rate >= 2.5:
        return "Moderate rates — neutral environment"
    return "Low rates — supportive for growth stocks and high P/E valuations"


def _inflation_environment(cpi_yoy: float | None) -> str:
    """Classifies the inflation environment from a YoY % change figure."""
    if cpi_yoy is None:
        return "Unknown"
    if cpi_yoy >= 5.0:
        return f"High inflation ({cpi_yoy}% YoY) — margin pressure, Fed likely to keep rates elevated"
    if cpi_yoy >= 2.5:
        return f"Moderate inflation ({cpi_yoy}% YoY) — within manageable range"
    return f"Low inflation ({cpi_yoy}% YoY) — supportive macro environment"


def macro_agent_node(state: FinancialState) -> dict:
    """
    Agent 5 — Macro & Market Context
    Two responsibilities combined into one lightweight node:

    1. MACRO CONTEXT (FRED API — requires FRED_API_KEY in .env):
       - Fed Funds Rate      → rate environment classification
       - CPI                 → inflation environment classification
       - 10Y Treasury Yield  → risk-free rate benchmark

    2. MARKET CONTEXT (yfinance — no API key needed):
       - Stock 3-month return vs SPY (S&P 500)
       - Stock 3-month return vs QQQ (NASDAQ 100)

    Degrades gracefully: if FRED_API_KEY is missing, macro fields return None
    and the pipeline continues with market context only.
    """
    ticker       = state["ticker"]
    stock_return = state.get("stock_3m_return")   # already computed by sector_agent

    print(f"--- [Agent 5] Fetching macro & market context for {ticker}... ---")

    # ── 1. FRED Macro Data ───────────────────────────────────────────────────
    if not FRED_API_KEY:
        print("[Agent 5] FRED_API_KEY not set — skipping macro data.")
        fed_rate     = None
        cpi          = None
        treasury_10y = None
    else:
        fed_values   = _fetch_fred_series(FRED_SERIES["fed_funds_rate"])
        t10y_values  = _fetch_fred_series(FRED_SERIES["treasury_10y"])
        fed_rate     = fed_values[0]  if fed_values  else None
        treasury_10y = t10y_values[0] if t10y_values else None
        cpi          = _fetch_cpi_yoy()   # YoY % change, not raw index
        print(f"--- [Agent 5] Fed: {fed_rate}% | CPI YoY: {cpi}% | 10Y: {treasury_10y}% ---")

    rate_env      = _rate_environment(fed_rate, treasury_10y)
    inflation_env = _inflation_environment(cpi)

    # ── 2. Market Context (SPY + QQQ) ────────────────────────────────────────
    spy_return = _benchmark_return("SPY")
    qqq_return = _benchmark_return("QQQ")

    # Stock vs benchmark deltas (None if stock_return or benchmark unavailable)
    def vs_benchmark(benchmark_return):
        if stock_return is None or benchmark_return is None:
            return None
        return round(stock_return - benchmark_return, 2)

    vs_spy = vs_benchmark(spy_return)
    vs_qqq = vs_benchmark(qqq_return)

    def market_label(delta):
        if delta is None:
            return "Unknown"
        if delta > 15:
            return "Strong Outperformer"
        if delta > 8:
            return "Outperformer"
        if delta > -8:
            return "In-line"
        if delta > -15:
            return "Underperformer"
        return "Strong Underperformer"

    print(
        f"--- [Agent 5] vs SPY: {vs_spy:+.2f}% ({market_label(vs_spy)}) | "
        f"vs QQQ: {vs_qqq:+.2f}% ({market_label(vs_qqq)}) ---"
    )

    return {
        # Macro
        "fed_funds_rate":     fed_rate,
        "cpi":                cpi,
        "treasury_10y":       treasury_10y,
        "rate_environment":   rate_env,
        "inflation_environment": inflation_env,
        # Market context
        "spy_3m_return":      spy_return,
        "qqq_3m_return":      qqq_return,
        "vs_spy":             vs_spy,
        "vs_qqq":             vs_qqq,
        "vs_spy_label":       market_label(vs_spy),
        "vs_qqq_label":       market_label(vs_qqq),
    }
