from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict


class FinancialState(TypedDict):
    # --- Input ---
    ticker: str

    # --- Agent 1: Data Ingestion ---
    start_date: str
    end_date: str
    historical_data: Annotated[Any, "Raw pandas DataFrame (in-memory only)"]

    # --- Agent 2: Technical & Fundamental Analysis ---
    fundamental_metrics: Dict[str, Any]   # P/E, market cap, sector, etc.
    technical_indicators: Dict[str, Any]  # RSI, MACD, trend, current price

    # --- Agent 3: Risk ---
    risk_metrics: Dict[str, Any]          # sharpe, max_drawdown, volatility

    # --- Agent 4: Sector Relative Strength ---
    sector_etf:        str                # e.g. "XLK"
    stock_3m_return:   float              # stock's 3-month % return
    sector_3m_return:  float              # sector ETF's 3-month % return
    relative_strength: float              # stock_3m_return - sector_3m_return
    sector_label:      str                # "Strong Outperformer" | "Outperformer" | "In-line" | etc.

    # --- Agent 5: Macro & Market Context ---
    fed_funds_rate:          float              # current Fed Funds Rate %
    cpi:                     float              # latest CPI reading
    treasury_10y:            float              # 10-year Treasury yield %
    rate_environment:        str                # plain-language rate classification
    inflation_environment:   str                # plain-language inflation classification
    spy_3m_return:           float              # S&P 500 3-month return %
    qqq_3m_return:           float              # NASDAQ 100 3-month return %
    vs_spy:                  float              # stock return minus SPY return
    vs_qqq:                  float              # stock return minus QQQ return
    vs_spy_label:            str                # "Outperformer" | "In-line" | etc.
    vs_qqq_label:            str                # "Outperformer" | "In-line" | etc.

    # --- Agent 6: Sentiment ---
    news_headlines: List[str]             # raw headlines passed to Gemini
    sentiment_score: float                # -1.0 (bearish) → +1.0 (bullish)
    sentiment_label: str                  # "Bullish" | "Neutral" | "Bearish"
    sentiment_summary: str                # Gemini's 2-3 sentence theme summary

    # --- Agent 5: Chart Vision ---
    chart_image_path: str                 # path to the close-up PNG (for UI display)
    chart_bias: str                       # "Bullish" | "Bearish" | "Neutral"
    chart_confidence: str                 # "Low" | "Medium" | "High"
    chart_patterns_detected: List[str]    # e.g. ["Convergence forming", "Inside Day"]
    chart_analysis: str                   # Gemini's visual narrative
    fib_levels: Dict[str, float]          # {"0%": x, "23.6%": x, "38.2%": x, "50%": x, "61.8%": x, "100%": x}

    # --- Agent 6: Earnings & Financials ---
    earnings_history: List[Dict[str, Any]]  # last N quarters of financial data
    next_earnings_date: str                  # e.g. "2025-07-22"
    days_to_earnings: int                    # calendar days until next report
    earnings_warning: str                    # caution message if earnings are imminent

    # --- Portfolio context (optional — provided by user at analysis time) ---
    owns_stock:  bool                     # True if user already holds this ticker
    buy_price:   float                    # cost basis per share (0.0 if not owned)
    shares_held: float                    # number of shares held (0.0 if not owned)

    # --- Price Recommendation (deterministic, post-synthesis) ---
    entry_low:      float          # bottom of entry zone ($)
    entry_high:     float          # top of entry zone ($)
    stop_loss:      float          # stop loss price ($)
    target_1:       float          # conservative target ($)
    target_2:       float          # aggressive target ($)
    risk_reward:    float          # (target_1 - entry_mid) / (entry_mid - stop_loss)
    price_rec_note: str            # short explanation of how levels were set

    # --- Agent 7: Synthesis (hybrid gate + Gemini tree output) ---
    recommendation: str                   # "BUY" | "SELL" | "HOLD"
    risk_level: str                       # "Low" | "Medium" | "High" | "Extreme"
    reasoning: str                        # Concise LLM explanation
    key_drivers: List[str]                # 3-5 data-point strings
    tree_path: List[str]                  # decision tree steps Gemini followed
    final_report: str                     # Formatted markdown string
