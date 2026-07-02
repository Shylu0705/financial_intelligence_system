import yfinance as yf

from core.state import FinancialState

# Relative strength tiers (stock return minus sector ETF return)
STRONG_OUTPERFORM =  15.0
OUTPERFORM        =   8.0
UNDERPERFORM      =  -8.0
STRONG_UNDERPERFORM = -15.0

# yfinance sector name → SPDR sector ETF ticker
SECTOR_ETF_MAP = {
    "Technology":            "XLK",
    "Healthcare":            "XLV",
    "Financial Services":    "XLF",
    "Financials":            "XLF",
    "Energy":                "XLE",
    "Consumer Cyclical":     "XLY",
    "Consumer Defensive":    "XLP",
    "Industrials":           "XLI",
    "Real Estate":           "XLRE",
    "Utilities":             "XLU",
    "Basic Materials":       "XLB",
    "Communication Services":"XLC",
}


def _three_month_return(df) -> float:
    """
    Calculates the percentage return over the last ~63 trading days (3 months).
    Uses the last available closing price vs the price 63 bars ago.
    """
    if len(df) < 63:
        # Fewer than 63 bars — use whatever we have
        start_price = df["Close"].iloc[0]
    else:
        start_price = df["Close"].iloc[-63]

    end_price = df["Close"].iloc[-1]
    return round(((end_price - start_price) / start_price) * 100, 2)


def _sector_label(relative_strength: float) -> str:
    if relative_strength > STRONG_OUTPERFORM:
        return "Strong Outperformer"
    if relative_strength > OUTPERFORM:
        return "Outperformer"
    if relative_strength > UNDERPERFORM:
        return "In-line"
    if relative_strength > STRONG_UNDERPERFORM:
        return "Underperformer"
    return "Strong Underperformer"


def sector_agent_node(state: FinancialState) -> dict:
    """
    Agent 4 — Sector Relative Strength
    Compares the stock's 3-month return against its sector ETF benchmark.
    Requires no new API key — uses yfinance and data already in state.

    Tiers:
        > +15%          → Strong Outperformer
        +8% to +15%     → Outperformer
        -8% to +8%      → In-line
        -15% to -8%     → Underperformer
        < -15%          → Strong Underperformer
    """
    ticker  = state["ticker"]
    sector  = state.get("fundamental_metrics", {}).get("sector", "Unknown")

    print(f"--- [Agent 4] Sector relative strength for {ticker} (sector: {sector})... ---")

    # ── 1. Map sector → ETF ──────────────────────────────────────────────────
    etf_ticker = SECTOR_ETF_MAP.get(sector)

    if not etf_ticker:
        print(f"[Agent 4] No ETF mapping for sector '{sector}'. Skipping.")
        return {
            "sector_etf":          "N/A",
            "stock_3m_return":     None,
            "sector_3m_return":    None,
            "relative_strength":   None,
            "sector_label":        "Unknown",
        }

    # ── 2. Stock 3-month return (from state — no extra API call) ────────────
    stock_return = _three_month_return(state["historical_data"])

    # ── 3. Sector ETF 3-month return ─────────────────────────────────────────
    try:
        etf_data = yf.Ticker(etf_ticker).history(period="6mo", auto_adjust=True)
        if etf_data.empty:
            raise ValueError(f"No data returned for ETF {etf_ticker}")
        sector_return = _three_month_return(etf_data)
    except Exception as e:
        print(f"[Agent 4] ETF fetch error for {etf_ticker}: {e}")
        return {
            "sector_etf":        etf_ticker,
            "stock_3m_return":   stock_return,
            "sector_3m_return":  None,
            "relative_strength": None,
            "sector_label":      "Unknown",
        }

    # ── 4. Relative strength & label ─────────────────────────────────────────
    relative_strength = round(stock_return - sector_return, 2)
    label = _sector_label(relative_strength)

    print(
        f"--- [Agent 4] {ticker}: {stock_return:+.2f}% | "
        f"{etf_ticker}: {sector_return:+.2f}% | "
        f"RS: {relative_strength:+.2f}% → {label} ---"
    )

    return {
        "sector_etf":        etf_ticker,
        "stock_3m_return":   stock_return,
        "sector_3m_return":  sector_return,
        "relative_strength": relative_strength,
        "sector_label":      label,
    }
