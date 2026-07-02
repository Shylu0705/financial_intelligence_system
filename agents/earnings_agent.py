import os
from datetime import datetime, timezone

import requests
import yfinance as yf
from dotenv import load_dotenv

from core.state import FinancialState

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

MAX_QUARTERS = 8

# Days threshold below which we flag an earnings warning
EARNINGS_CAUTION_DAYS = 14


def _safe_millions(value) -> str:
    """Converts a raw numeric value to a rounded millions string, or 'N/A'."""
    try:
        return f"${round(float(value) / 1_000_000, 2)}M"
    except (TypeError, ValueError):
        return "N/A"


def _days_from_now(date_str: str) -> int:
    """Returns calendar days from today to a YYYY-MM-DD date string."""
    target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((target - datetime.now(tz=timezone.utc)).days, 0)


def _get_next_earnings_yfinance(stock: yf.Ticker) -> tuple[str, int]:
    """Try yfinance calendar first."""
    try:
        cal = stock.calendar
        if cal is None:
            return "Unknown", -1

        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
        else:
            earnings_date = cal.loc["Earnings Date"].iloc[0] if "Earnings Date" in cal.index else None

        if earnings_date is None:
            return "Unknown", -1

        if hasattr(earnings_date, "__iter__") and not isinstance(earnings_date, str):
            earnings_date = list(earnings_date)[0]

        if hasattr(earnings_date, "tzinfo") and earnings_date.tzinfo is None:
            earnings_date = earnings_date.replace(tzinfo=timezone.utc)

        date_str  = earnings_date.strftime("%Y-%m-%d")
        days_away = _days_from_now(date_str)
        return date_str, days_away

    except Exception:
        return "Unknown", -1


def _get_next_earnings_finnhub(ticker: str) -> tuple[str, int]:
    """Fallback: Finnhub /stock/earnings returns upcoming earnings dates."""
    if not FINNHUB_API_KEY:
        return "Unknown", -1
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={
                "symbol": ticker,
                "from":   datetime.now().strftime("%Y-%m-%d"),
                "to":     "2026-12-31",
                "token":  FINNHUB_API_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()
        earnings_calendar = response.json().get("earningsCalendar", [])
        if not earnings_calendar:
            return "Unknown", -1

        # First result is the soonest upcoming date
        date_str  = earnings_calendar[0]["date"]
        days_away = _days_from_now(date_str)
        return date_str, days_away

    except Exception:
        return "Unknown", -1


def _get_next_earnings_date(stock: yf.Ticker, ticker: str) -> tuple[str, int]:
    """
    Tries yfinance first, falls back to Finnhub if yfinance returns Unknown.
    """
    date_str, days_away = _get_next_earnings_yfinance(stock)
    if date_str != "Unknown":
        return date_str, days_away

    print(f"[Agent 6] yfinance calendar unavailable — trying Finnhub fallback...")
    return _get_next_earnings_finnhub(ticker)


def _build_earnings_warning(days: int, date_str: str) -> str:
    if days < 0 or date_str == "Unknown":
        return "Next earnings date unavailable."
    if days <= 3:
        return f"⚠️  EARNINGS IN {days} DAY(S) ({date_str}) — extreme volatility likely. Avoid new positions."
    if days <= 7:
        return f"⚠️  Earnings in {days} days ({date_str}) — elevated risk. Consider waiting for the report."
    if days <= EARNINGS_CAUTION_DAYS:
        return f"Caution: Earnings in {days} days ({date_str}). Factor this into position sizing."
    return f"Next earnings: {date_str} ({days} days away). Sufficient runway for a new position."


def earnings_agent_node(state: FinancialState) -> dict:
    """
    Agent 6 — Earnings & Financials
    Fetches the last N quarterly financial results (up to MAX_QUARTERS) and
    the next scheduled earnings date. Handles companies with fewer than
    MAX_QUARTERS of history gracefully.

    Fields per quarter:
        - Sales (Total Revenue)
        - Net Profit (Net Income)
        - Dividend (summed from dividend series)
        - Equity (Common Stock / Share Capital)
        - Reserves & Surplus (Retained Earnings)
        - Networth (Total Stockholders Equity)
        - Debt (Total Debt)
        - Share Price (quarter-end closing price)
    """
    ticker = state["ticker"]
    print(f"--- [Agent 6] Fetching earnings & financials for {ticker}... ---")

    stock = yf.Ticker(ticker)
    history = state["historical_data"]

    # ── 1. Next earnings date ────────────────────────────────────────────────
    next_date, days_away = _get_next_earnings_date(stock, ticker)
    warning = _build_earnings_warning(days_away, next_date)
    print(f"--- [Agent 6] Next earnings: {next_date} ({days_away} days) ---")

    # ── 2. Quarterly financials ──────────────────────────────────────────────
    try:
        income_stmt  = stock.quarterly_income_stmt
        balance_sheet = stock.quarterly_balance_sheet
    except Exception as e:
        print(f"[Agent 6] Failed to fetch financials: {e}")
        return {
            "earnings_history":   [],
            "next_earnings_date": next_date,
            "days_to_earnings":   days_away,
            "earnings_warning":   warning,
        }

    if income_stmt is None or income_stmt.empty:
        print(f"[Agent 6] No quarterly income data for {ticker}.")
        return {
            "earnings_history":   [],
            "next_earnings_date": next_date,
            "days_to_earnings":   days_away,
            "earnings_warning":   warning,
        }

    # Quarters are columns; take the most recent MAX_QUARTERS
    quarters = income_stmt.columns[:MAX_QUARTERS]
    n_found  = len(quarters)

    if n_found < MAX_QUARTERS:
        print(f"[Agent 6] Only {n_found} quarters available (requested {MAX_QUARTERS}) — using all.")

    # ── 3. Dividends — aggregate per quarter ────────────────────────────────
    dividends = stock.dividends
    if dividends is not None and not dividends.empty:
        dividends.index = dividends.index.tz_localize(None)

    def _dividends_for_quarter(q_end):
        """Sum dividends paid in the 3 months up to and including q_end."""
        try:
            if dividends is None or dividends.empty:
                return "N/A"
            q_start = q_end - pd.DateOffset(months=3)
            mask = (dividends.index > q_start) & (dividends.index <= q_end)
            total = dividends[mask].sum()
            return f"${round(total, 4)}" if total > 0 else "$0.00"
        except Exception:
            return "N/A"

    # ── 4. Build quarter records ─────────────────────────────────────────────
    import pandas as pd

    earnings_history = []

    for q_date in quarters:
        def get_row(df, *keys):
            """Try multiple key variants — yfinance column names vary by ticker."""
            for key in keys:
                if key in df.index:
                    val = df.loc[key, q_date]
                    if pd.notna(val):
                        return val
            return None

        # Income statement fields
        revenue    = get_row(income_stmt, "Total Revenue", "Revenue")
        net_profit = get_row(income_stmt, "Net Income", "Net Income Common Stockholders")

        # Balance sheet fields
        equity            = get_row(balance_sheet, "Common Stock", "Share Capital", "Common Stock Equity")
        retained_earnings = get_row(balance_sheet, "Retained Earnings", "Retained Earnings / Accumulated Deficit")
        total_equity      = get_row(balance_sheet, "Stockholders Equity", "Total Equity Gross Minority Interest",
                                    "Common Stock Equity")
        total_debt        = get_row(balance_sheet, "Total Debt", "Long Term Debt And Capital Lease Obligation",
                                    "Long Term Debt")

        # Quarter-end share price: use the closing price nearest to q_date
        try:
            q_date_naive = q_date.tz_localize(None) if q_date.tzinfo else q_date
            nearest_price = history["Close"].asof(q_date_naive)
            share_price = f"${round(float(nearest_price), 2)}" if pd.notna(nearest_price) else "N/A"
        except Exception:
            share_price = "N/A"

        earnings_history.append({
            "quarter":           q_date.strftime("%b %Y"),
            "sales":             _safe_millions(revenue),
            "net_profit":        _safe_millions(net_profit),
            "dividend":          _dividends_for_quarter(q_date_naive if 'q_date_naive' in dir() else q_date),
            "equity":            _safe_millions(equity),
            "reserves_surplus":  _safe_millions(retained_earnings),
            "networth":          _safe_millions(total_equity),
            "debt":              _safe_millions(total_debt),
            "share_price":       share_price,
        })

    print(f"--- [Agent 6] Built {len(earnings_history)} quarter records ---")

    return {
        "earnings_history":   earnings_history,
        "next_earnings_date": next_date,
        "days_to_earnings":   days_away,
        "earnings_warning":   warning,
    }
