"""
Backtesting engine — deterministic signal replay, no LLM calls.

Uses the same numerical indicators as the live pipeline (RSI, MACD, ADX,
MA45/MA180, ATR) with a rule-based synthesis to generate weekly signals,
then simulates trades and computes performance metrics.

Signal rules (simplified decision tree):
  BUY  — MACD line > Signal line  AND  Close > MA180  AND  RSI < 70
  SELL — MACD line < Signal line  AND  Close < MA45   AND  RSI > 30
  HOLD — everything else

Trade management:
  - Signals checked weekly (last trading day of each week)
  - Stop loss: entry_price - atr_mult × ATR14 (checked daily via row["Low"])
  - Max holding period: configurable in days (default 100)
  - Open position at end of period is closed at last available price
"""

import numpy as np
import pandas as pd
import yfinance as yf

# How many calendar days of extra history to fetch before start_date
# so that MA180 has enough data for its first valid value.
_WARMUP_DAYS = 300


# ── Indicator computation ────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI-14
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD (12/26/9)
    ema12         = df["Close"].ewm(span=12, adjust=False).mean()
    ema26         = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]    = ema12 - ema26
    df["MACDSig"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Moving averages
    df["MA45"]  = df["Close"].rolling(45).mean()
    df["MA180"] = df["Close"].rolling(180).mean()

    # ATR-14 (reuses the NaN-safe formula from analysis_agent)
    tr1     = df["High"] - df["Low"]
    tr2     = (df["High"] - df["Close"].shift(1)).abs()
    tr3     = (df["Low"]  - df["Close"].shift(1)).abs()
    tr      = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(tr1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # ADX-14 (used informally — trend filter could be added later)
    up_move  = df["High"].diff()
    dn_move  = -df["Low"].diff()
    plus_dm  = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    tr_s     = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_s
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_s
    denom    = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / denom
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Signal generation
    buy_cond  = (df["MACD"] > df["MACDSig"]) & (df["Close"] > df["MA180"]) & (df["RSI"] < 70)
    sell_cond = (df["MACD"] < df["MACDSig"]) & (df["Close"] < df["MA45"])  & (df["RSI"] > 30)

    df["TradeSignal"] = "HOLD"
    df.loc[buy_cond,  "TradeSignal"] = "BUY"
    df.loc[sell_cond, "TradeSignal"] = "SELL"

    return df


# ── Trade simulation ─────────────────────────────────────────────────────────

def _simulate(
    df: pd.DataFrame,
    atr_mult:      float = 1.5,
    max_hold_days: int   = 100,
) -> tuple[list[dict], pd.Series]:
    """
    Single daily pass through the data.
    Signals are only *acted on* at the last trading day of each week (Friday
    or the closest preceding day) — daily data drives the equity curve and
    stop-loss checks, but entry/exit decisions happen weekly.
    """
    # Build a set of weekly sample dates (last trading day of each week)
    weekly_dates = set(
        df.resample("W-FRI").last().dropna(subset=["Close"]).index.normalize()
    )

    in_position = False
    entry_price = 0.0
    entry_date  = None
    stop_price  = 0.0
    cash        = 10_000.0
    shares      = 0.0

    trades: list[dict] = []
    equity_values      = []

    for date, row in df.iterrows():
        price = float(row["Close"])
        low   = float(row["Low"])
        atr   = float(row["ATR"]) if not np.isnan(row["ATR"]) else 0.0

        # Daily stop-loss check (even between signal dates)
        if in_position and low <= stop_price:
            exit_price  = stop_price
            hold_days   = (date - entry_date).days
            ret_pct     = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "entry_date":   entry_date.strftime("%Y-%m-%d"),
                "exit_date":    date.strftime("%Y-%m-%d"),
                "entry_price":  round(entry_price, 2),
                "exit_price":   round(exit_price,  2),
                "exit_reason":  "Stop loss",
                "return_pct":   round(ret_pct, 2),
                "holding_days": hold_days,
            })
            cash        = shares * exit_price
            shares      = 0.0
            in_position = False
            stop_price  = 0.0

        # Weekly signal check
        if date.normalize() in weekly_dates:
            signal    = row["TradeSignal"]
            hold_days = (date - entry_date).days if entry_date else 0

            if not in_position and signal == "BUY" and not np.isnan(row["MA180"]):
                shares      = cash / price
                cash        = 0.0
                entry_price = price
                entry_date  = date
                stop_price  = price - atr_mult * atr
                in_position = True

            elif in_position:
                hit_max  = hold_days >= max_hold_days
                hit_sell = signal == "SELL"

                if hit_max or hit_sell:
                    ret_pct = (price - entry_price) / entry_price * 100
                    reason  = "Max hold" if hit_max else "SELL signal"
                    trades.append({
                        "entry_date":   entry_date.strftime("%Y-%m-%d"),
                        "exit_date":    date.strftime("%Y-%m-%d"),
                        "entry_price":  round(entry_price, 2),
                        "exit_price":   round(price, 2),
                        "exit_reason":  reason,
                        "return_pct":   round(ret_pct, 2),
                        "holding_days": hold_days,
                    })
                    cash        = shares * price
                    shares      = 0.0
                    in_position = False
                    stop_price  = 0.0

        # Daily portfolio value for equity curve
        pv = shares * price if in_position else cash
        equity_values.append((date, round(pv, 2)))

    # Close any open position at end of period
    if in_position:
        last_date  = df.index[-1]
        last_price = float(df.iloc[-1]["Close"])
        ret_pct    = (last_price - entry_price) / entry_price * 100
        trades.append({
            "entry_date":   entry_date.strftime("%Y-%m-%d"),
            "exit_date":    last_date.strftime("%Y-%m-%d"),
            "entry_price":  round(entry_price,  2),
            "exit_price":   round(last_price,   2),
            "exit_reason":  "End of period",
            "return_pct":   round(ret_pct, 2),
            "holding_days": (last_date - entry_date).days,
        })

    equity_curve = pd.Series(
        [v for _, v in equity_values],
        index=[d for d, _ in equity_values],
    )
    return trades, equity_curve


# ── Performance metrics ──────────────────────────────────────────────────────

def _compute_metrics(
    trades:       list[dict],
    equity_curve: pd.Series,
    df:           pd.DataFrame,
) -> dict:
    if not trades:
        return {
            "error":               "No trades generated in this period — signals never triggered.",
            "total_return_pct":    0.0,
            "buy_hold_return_pct": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
            "num_trades":          0,
        }

    returns = [t["return_pct"] for t in trades]
    winners = [r for r in returns if r > 0]
    losers  = [r for r in returns if r <= 0]

    total_return   = round((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100, 2)
    bh_return      = round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)
    win_rate       = round(len(winners) / len(returns) * 100, 1) if returns else 0.0
    avg_win        = round(float(np.mean(winners)), 2) if winners else 0.0
    avg_loss       = round(float(np.mean(losers)),  2) if losers  else 0.0
    sum_loss       = abs(sum(losers))
    profit_factor  = round(sum(winners) / sum_loss, 2) if sum_loss > 0 else float("inf")

    rolling_max    = equity_curve.cummax()
    drawdowns      = (equity_curve - rolling_max) / rolling_max * 100
    max_dd         = round(float(drawdowns.min()), 2)

    daily_ret      = equity_curve.pct_change().dropna()
    sharpe         = round(
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)), 2
    ) if daily_ret.std() > 0 else 0.0

    avg_hold = round(float(np.mean([t["holding_days"] for t in trades])), 1)

    return {
        "total_return_pct":    total_return,
        "buy_hold_return_pct": bh_return,
        "win_rate_pct":        win_rate,
        "avg_win_pct":         avg_win,
        "avg_loss_pct":        avg_loss,
        "profit_factor":       profit_factor,
        "max_drawdown_pct":    max_dd,
        "sharpe_ratio":        sharpe,
        "num_trades":          len(trades),
        "avg_hold_days":       avg_hold,
    }


# ── Public entry point ───────────────────────────────────────────────────────

def run_backtest(
    ticker:        str,
    start_date:    str,
    end_date:      str,
    max_hold_days: int   = 100,
    atr_mult:      float = 1.5,
) -> dict:
    """
    Main backtest entry point. Fetches data, computes indicators,
    simulates trades, and returns metrics + equity curve + trades list.
    """
    print(f"\n📊 Backtesting {ticker} | {start_date} → {end_date}")

    warmup_start = (
        pd.Timestamp(start_date) - pd.DateOffset(days=_WARMUP_DAYS)
    ).strftime("%Y-%m-%d")

    raw = yf.download(ticker, start=warmup_start, end=end_date, progress=False)

    if raw.empty:
        return {"error": f"No market data found for '{ticker}'."}

    # yfinance sometimes returns a MultiIndex when auto_adjust=True
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = _compute_indicators(raw)

    # Trim to requested range AFTER computing indicators — no look-ahead bias
    df = df[df.index >= pd.Timestamp(start_date)]

    if df.empty or len(df) < 45:
        return {"error": "Not enough data in requested range after indicator warmup."}

    trades, equity_curve = _simulate(df, atr_mult=atr_mult, max_hold_days=max_hold_days)
    metrics              = _compute_metrics(trades, equity_curve, df)

    # Buy-and-hold curve for chart comparison
    bh_curve = (10_000.0 * df["Close"] / df["Close"].iloc[0]).round(2)

    print(f"📊 Backtest done — {len(trades)} trades | "
          f"Return: {metrics.get('total_return_pct', 'N/A')}% | "
          f"B&H: {metrics.get('buy_hold_return_pct', 'N/A')}%")

    return {
        "ticker":       ticker,
        "start_date":   start_date,
        "end_date":     end_date,
        "metrics":      metrics,
        "trades":       trades,
        # Serialise as {date_str: value} for JSON
        "equity_curve": {
            d.strftime("%Y-%m-%d"): v
            for d, v in equity_curve.items()
        },
        "bh_curve": {
            d.strftime("%Y-%m-%d"): v
            for d, v in bh_curve.items()
        },
    }
