"""
Price Recommendation Agent — deterministic, no LLM.

Translates the synthesis recommendation (BUY / SELL / HOLD) into
actionable price levels using Fibonacci retracement, ATR-14, and MAs.

Logic overview
──────────────
BUY (no position)
  Entry zone : [nearest Fib support ≤ current price, current price + 0.5×ATR]
  Stop loss  : tighter of (entry_low - 1.5×ATR) or (next Fib level below entry_low)
  Target 1   : nearest Fib resistance above current price
  Target 2   : next Fib resistance above Target 1 (or 52-week high)

BUY / ADD (owns stock)
  Entry zone : same as above — user is adding at current price
  Stop loss  : protect cost basis — tighter of (buy_price - 1×ATR) or Fib below buy_price
  Targets    : same

SELL / EXIT (owns stock)
  No entry — recommend exiting at market (current price)
  Target     : current price (exit now)
  Stop       : N/A

HOLD (no position)
  No trade — all levels set to 0.0

HOLD (owns stock)
  Trailing stop suggestion only — 1.5×ATR below current price or nearest Fib below
"""

from core.state import FinancialState

# ATR multipliers
_STOP_ATR_MULT   = 1.5   # stop loss distance from entry
_ENTRY_ATR_HALF  = 0.5   # half-ATR band above entry to form the entry zone top


def _sorted_fib_prices(fib_levels: dict) -> list[float]:
    """Returns Fib price values sorted ascending (low → high)."""
    return sorted(fib_levels.values())


def _nearest_support(price: float, fib_prices: list[float]) -> float | None:
    """Highest Fib level that is at or below `price`."""
    candidates = [f for f in fib_prices if f <= price]
    return max(candidates) if candidates else None


def _nearest_resistance(price: float, fib_prices: list[float]) -> float | None:
    """Lowest Fib level that is strictly above `price`."""
    candidates = [f for f in fib_prices if f > price]
    return min(candidates) if candidates else None


def _second_resistance(price: float, fib_prices: list[float]) -> float | None:
    """Second-lowest Fib level above `price` (Target 2)."""
    candidates = sorted(f for f in fib_prices if f > price)
    return candidates[1] if len(candidates) >= 2 else (candidates[0] if candidates else None)


def _fib_below(price: float, fib_prices: list[float]) -> float | None:
    """Highest Fib level strictly below `price`."""
    candidates = [f for f in fib_prices if f < price]
    return max(candidates) if candidates else None


def _round2(v: float | None) -> float:
    return round(v, 2) if v is not None else 0.0


def price_recommendation_node(state: FinancialState) -> dict:
    """
    Post-synthesis agent that converts BUY/SELL/HOLD into price levels.
    All arithmetic — no LLM call.
    """
    rec         = state.get("recommendation", "HOLD")
    fib_levels  = state.get("fib_levels", {})
    technicals  = state.get("technical_indicators", {})
    owns_stock  = state.get("owns_stock",  False)
    buy_price   = state.get("buy_price",   0.0)

    current = technicals.get("current_price", 0.0)
    atr     = technicals.get("atr_14",        0.0)
    w52_high = technicals.get("week52_high",  0.0)

    ticker = state["ticker"]
    print(f"--- [Price Rec] {ticker} | Rec: {rec} | Owns: {owns_stock} | ATR: ${atr:.2f} ---")

    fib_prices = _sorted_fib_prices(fib_levels) if fib_levels else []

    # ── HOLD with no position — no trade ────────────────────────────────────
    if rec == "HOLD" and not owns_stock:
        return {
            "entry_low":      0.0,
            "entry_high":     0.0,
            "stop_loss":      0.0,
            "target_1":       0.0,
            "target_2":       0.0,
            "risk_reward":    0.0,
            "price_rec_note": "HOLD — no position. No entry recommended. Wait for clearer signal.",
        }

    # ── HOLD with existing position — trailing stop only ────────────────────
    if rec == "HOLD" and owns_stock:
        fib_stop  = _fib_below(current, fib_prices)
        atr_stop  = current - _STOP_ATR_MULT * atr
        stop      = _round2(max(fib_stop, atr_stop) if fib_stop else atr_stop)
        return {
            "entry_low":      0.0,
            "entry_high":     0.0,
            "stop_loss":      stop,
            "target_1":       0.0,
            "target_2":       0.0,
            "risk_reward":    0.0,
            "price_rec_note": (
                f"HOLD existing position. Suggested trailing stop: ${stop:.2f} "
                f"(tighter of 1.5×ATR below current or nearest Fib support)."
            ),
        }

    # ── SELL with existing position — exit at market ─────────────────────────
    if rec == "SELL" and owns_stock:
        pnl_pct = (current - buy_price) / buy_price * 100 if buy_price else 0.0
        return {
            "entry_low":      0.0,
            "entry_high":     0.0,
            "stop_loss":      0.0,
            "target_1":       _round2(current),
            "target_2":       0.0,
            "risk_reward":    0.0,
            "price_rec_note": (
                f"EXIT position at market (~${current:.2f}). "
                f"Unrealised P&L from cost basis ${buy_price:.2f}: {pnl_pct:+.1f}%."
            ),
        }

    # ── SELL with no position — short entry (advanced) ──────────────────────
    if rec == "SELL" and not owns_stock:
        entry_high = _round2(current)
        entry_low  = _round2(current - _ENTRY_ATR_HALF * atr)
        fib_stop_r = _nearest_resistance(current, fib_prices)
        atr_stop   = current + _STOP_ATR_MULT * atr
        stop       = _round2(min(fib_stop_r, atr_stop) if fib_stop_r else atr_stop)
        t1         = _round2(_nearest_support(current, fib_prices) or (current - 2 * atr))
        t2_fib     = _fib_below(t1, fib_prices) if t1 else None
        t2         = _round2(t2_fib or (t1 - 2 * atr))
        entry_mid  = (entry_low + entry_high) / 2
        rr         = _round2((entry_mid - t1) / (stop - entry_mid)) if stop > entry_mid else 0.0
        return {
            "entry_low":      entry_low,
            "entry_high":     entry_high,
            "stop_loss":      stop,
            "target_1":       t1,
            "target_2":       t2,
            "risk_reward":    rr,
            "price_rec_note": (
                f"SHORT entry zone ${entry_low:.2f}–${entry_high:.2f}. "
                f"Stop: ${stop:.2f} (1.5×ATR or next Fib resistance). "
                f"T1: ${t1:.2f} | T2: ${t2:.2f} | R/R: {rr:.1f}:1."
            ),
        }

    # ── BUY / ADD — long entry ───────────────────────────────────────────────
    # Entry zone anchored to nearest Fib support at or below current price
    fib_support = _nearest_support(current, fib_prices)
    entry_low   = _round2(fib_support if fib_support else (current - _ENTRY_ATR_HALF * atr))
    entry_high  = _round2(current + _ENTRY_ATR_HALF * atr)
    entry_mid   = (entry_low + entry_high) / 2

    # Stop: tighter of ATR-based or next Fib below entry_low
    fib_stop = _fib_below(entry_low, fib_prices)
    atr_stop = entry_low - _STOP_ATR_MULT * atr
    stop     = _round2(max(fib_stop, atr_stop) if fib_stop else atr_stop)

    # If user owns the stock, also consider protecting cost basis
    if owns_stock and buy_price > 0:
        basis_stop = buy_price - atr  # don't let a loss exceed 1×ATR below cost basis
        stop = _round2(max(stop, basis_stop))

    # Targets: next two Fib resistance levels above current price
    t1 = _round2(_nearest_resistance(current, fib_prices) or (current + 2 * atr))
    t2_fib = _second_resistance(current, fib_prices)
    t2 = _round2(t2_fib or w52_high or (t1 + 2 * atr))

    rr = _round2((t1 - entry_mid) / (entry_mid - stop)) if entry_mid > stop else 0.0

    action = "ADD to position" if owns_stock else "BUY"
    note = (
        f"{action} — entry zone ${entry_low:.2f}–${entry_high:.2f} "
        f"(anchored to Fib support). "
        f"Stop: ${stop:.2f} (1.5×ATR"
        + (" + cost basis protection" if owns_stock and buy_price > 0 else "")
        + f"). T1: ${t1:.2f} | T2: ${t2:.2f} | R/R: {rr:.1f}:1."
    )

    print(f"--- [Price Rec] Entry: ${entry_low:.2f}–${entry_high:.2f} | Stop: ${stop:.2f} | T1: ${t1:.2f} | T2: ${t2:.2f} | R/R: {rr:.1f}:1 ---")

    return {
        "entry_low":      entry_low,
        "entry_high":     entry_high,
        "stop_loss":      stop,
        "target_1":       t1,
        "target_2":       t2,
        "risk_reward":    rr,
        "price_rec_note": note,
    }
