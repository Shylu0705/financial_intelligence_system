import numpy as np
import pandas as pd
import yfinance as yf

from core.state import FinancialState


def _compute_adx(df, period: int = 14) -> dict:
    """
    Computes ADX-14, +DI-14, and -DI-14.
    ADX > 25 → trending market; ADX < 20 → ranging/choppy market.

    Uses pandas-native operations throughout to preserve the DatetimeIndex
    and avoid the NaN-propagation bug that occurs when wrapping numpy arrays
    in pd.Series before passing to .ewm().
    """
    import pandas as pd

    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # True Range: max of the three span definitions.
    # tr2/tr3 have NaN at row 0 (no previous close); fill with simple High-Low
    # so EWM sees a valid seed value at position 0 instead of propagating NaN.
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(tr1)

    # Directional Movement (pandas diff keeps index alignment)
    up_move   = high.diff()
    down_move = -low.diff()   # equivalent to low.shift(1) - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # Wilder smoothing — alpha = 1/period is identical to Wilder's method
    tr_s       = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_s  = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    denom    = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / denom
    adx      = dx.ewm(alpha=1 / period, adjust=False).mean()

    latest_adx      = round(float(adx.iloc[-1]),  2)
    latest_plus_di  = round(float(plus_di.iloc[-1]),  2)
    latest_minus_di = round(float(minus_di.iloc[-1]), 2)

    if latest_adx >= 25:
        regime = "Trending"
    elif latest_adx <= 20:
        regime = "Ranging"
    else:
        regime = "Weak Trend"

    return {
        "adx_14":     latest_adx,
        "plus_di":    latest_plus_di,
        "minus_di":   latest_minus_di,
        "adx_regime": regime,
    }


def analysis_node(state: FinancialState) -> dict:
    """
    Agent 2 — Technical & Fundamental Analysis
    Calculates RSI-14, MACD, and ADX-14 from historical price data.
    Fetches fundamentals (P/E, market cap, sector) from yfinance info.
    """
    ticker = state["ticker"]
    data = state["historical_data"]

    print(f"--- [Agent 2] Analyzing {ticker}... ---")

    # --- Fundamental Metrics ---
    info = yf.Ticker(ticker).info
    fundamentals = {
        "market_cap":  info.get("marketCap", "N/A"),
        "pe_ratio":    info.get("trailingPE", "N/A"),
        "forward_pe":  info.get("forwardPE", "N/A"),
        "sector":      info.get("sector", "Unknown"),
    }

    # --- Technical Indicators ---
    df = data.copy()

    # RSI-14: exponential-weighted gain/loss ratio
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD: 12-day EMA minus 26-day EMA, with a 9-day signal line
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema_12 - ema_26
    df["Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()

    # ADX-14: trend strength + regime filter
    adx_data = _compute_adx(df)

    # ATR-14: average daily range in dollars — used for stop loss sizing
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(tr1)
    atr_14 = round(float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]), 4)

    # MA dollar values (last 252 days = 1-year window, matching the chart)
    year_data = df.tail(252)
    ma9_price   = round(float(year_data["Close"].rolling(9).mean().iloc[-1]),   2)
    ma45_price  = round(float(year_data["Close"].rolling(45).mean().iloc[-1]),  2)
    ma180_price = round(float(year_data["Close"].rolling(180).mean().iloc[-1]), 2)

    # 52-week high / low
    week52_high = round(float(year_data["High"].max()),  2)
    week52_low  = round(float(year_data["Low"].min()),   2)

    latest = df.iloc[-1]
    technicals = {
        "current_price": round(float(latest["Close"]), 2),
        "rsi_14":        round(latest["RSI"], 2),
        "macd_line":     round(latest["MACD_Line"], 2),
        "signal_line":   round(latest["Signal_Line"], 2),
        "trend":         "Bullish" if latest["MACD_Line"] > latest["Signal_Line"] else "Bearish",
        "atr_14":        atr_14,
        "ma9_price":     ma9_price,
        "ma45_price":    ma45_price,
        "ma180_price":   ma180_price,
        "week52_high":   week52_high,
        "week52_low":    week52_low,
        **adx_data,
    }

    print(
        f"--- [Agent 2] RSI: {technicals['rsi_14']} | "
        f"Trend: {technicals['trend']} | "
        f"ADX: {technicals['adx_14']} ({technicals['adx_regime']}) ---"
    )

    return {
        "fundamental_metrics":  fundamentals,
        "technical_indicators": technicals,
    }
