"""
Gate engine — evaluates all enabled hard gates against the current state.
Returns the first gate that fires, or None if all gates pass.

Each gate returns a GateResult(triggered, gate_name, reason) or None.
The synthesis agent checks this before calling Gemini.
"""

from dataclasses import dataclass

from config.gates import GATE_CONFIG


@dataclass
class GateResult:
    triggered:  bool
    gate_name:  str
    reason:     str
    forced_rec: str   # "HOLD" or "SELL"


def _parse_revenue(value: str) -> float | None:
    """Converts '$1234.56M' strings from earnings_history back to a float."""
    try:
        return float(value.replace("$", "").replace("M", "").strip())
    except (ValueError, AttributeError):
        return None


def _check_earnings_proximity(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["earnings_proximity"]
    if not cfg["enabled"]:
        return None

    days = state.get("days_to_earnings", -1)
    threshold = cfg["threshold_days"]

    if 0 <= days <= threshold:
        return GateResult(
            triggered=True,
            gate_name="earnings_proximity",
            reason=f"Earnings report in {days} day(s) ({state.get('next_earnings_date', 'Unknown')}) — high volatility expected. Avoiding new positions.",
            forced_rec="HOLD",
        )
    return None


def _check_revenue_decline(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["revenue_decline"]
    if not cfg["enabled"]:
        return None

    history = state.get("earnings_history", [])
    n = cfg["consecutive_quarters"]

    if len(history) < n:
        return None

    # earnings_history is ordered most recent first
    recent = [_parse_revenue(q.get("sales")) for q in history[:n]]

    if any(v is None for v in recent):
        return None

    # history is most-recent-first, so recent[0] = latest, recent[1] = previous
    # Declining means each quarter is lower than the one before it in time:
    # recent[0] < recent[1] < recent[2] ...
    declining = all(recent[i] < recent[i + 1] for i in range(n - 1))

    if declining:
        return GateResult(
            triggered=True,
            gate_name="revenue_decline",
            reason=f"Revenue has declined for {n} consecutive quarters — fundamental deterioration. Capping at HOLD.",
            forced_rec="HOLD",
        )
    return None


def _check_extreme_drawdown(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["extreme_drawdown"]
    if not cfg["enabled"]:
        return None

    drawdown = state.get("risk_metrics", {}).get("current_drawdown", 0)
    threshold = cfg["threshold_pct"] / 100   # stored as decimal e.g. -0.30

    if drawdown <= threshold:
        return GateResult(
            triggered=True,
            gate_name="extreme_drawdown",
            reason=f"Current drawdown of {drawdown:.1%} exceeds threshold of {cfg['threshold_pct']}% — stock is in freefall.",
            forced_rec="HOLD",
        )
    return None


def _check_extreme_volatility(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["extreme_volatility"]
    if not cfg["enabled"]:
        return None

    volatility = state.get("risk_metrics", {}).get("volatility", 0) * 100
    threshold  = cfg["threshold_pct"]

    if volatility >= threshold:
        return GateResult(
            triggered=True,
            gate_name="extreme_volatility",
            reason=f"Annualised volatility of {volatility:.1f}% exceeds threshold of {threshold}% — signal reliability too low.",
            forced_rec="HOLD",
        )
    return None


def _check_negative_networth(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["negative_networth"]
    if not cfg["enabled"]:
        return None

    history = state.get("earnings_history", [])
    if not history:
        return None

    latest_networth = _parse_revenue(history[0].get("networth"))
    if latest_networth is not None and latest_networth < 0:
        return GateResult(
            triggered=True,
            gate_name="negative_networth",
            reason=f"Company has negative networth ({history[0].get('networth')}) — liabilities exceed assets.",
            forced_rec="HOLD",
        )
    return None


def _check_extreme_macro(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["extreme_macro"]
    if not cfg["enabled"]:
        return None

    fed_rate = state.get("fed_funds_rate")
    pe_ratio = state.get("fundamental_metrics", {}).get("pe_ratio")

    if fed_rate is None or pe_ratio is None:
        return None

    if fed_rate >= cfg["min_fed_rate"] and pe_ratio >= cfg["min_pe_ratio"]:
        return GateResult(
            triggered=True,
            gate_name="extreme_macro",
            reason=f"Fed rate {fed_rate}% + P/E {pe_ratio:.1f} — expensive stock in a high-rate environment.",
            forced_rec="HOLD",
        )
    return None


def _check_pipeline_failure(state: dict) -> GateResult | None:
    cfg = GATE_CONFIG["pipeline_failure"]
    if not cfg["enabled"]:
        return None

    # Critical fields that must be present for a reliable recommendation
    critical_fields = [
        ("historical_data",     state.get("historical_data") is not None),
        ("technical_indicators",state.get("technical_indicators") is not None),
        ("risk_metrics",        state.get("risk_metrics") is not None),
        ("fundamental_metrics", state.get("fundamental_metrics") is not None),
    ]

    failed = [name for name, ok in critical_fields if not ok]

    if failed:
        return GateResult(
            triggered=True,
            gate_name="pipeline_failure",
            reason=f"Critical data missing from agents: {', '.join(failed)}. Cannot make a reliable recommendation.",
            forced_rec="HOLD",
        )
    return None


# Ordered list of gate checks — evaluated top to bottom, first trigger wins
_GATE_CHECKS = [
    _check_pipeline_failure,      # always check data integrity first
    _check_earnings_proximity,
    _check_revenue_decline,
    _check_extreme_drawdown,
    _check_extreme_volatility,
    _check_negative_networth,
    _check_extreme_macro,
]


def evaluate_gates(state: dict) -> GateResult | None:
    """
    Runs all enabled gates in priority order.
    Returns the first GateResult that triggers, or None if all pass.
    """
    for check in _GATE_CHECKS:
        result = check(state)
        if result is not None:
            print(f"[Gate Engine] ⛔ Gate '{result.gate_name}' triggered → {result.forced_rec}: {result.reason}")
            return result

    print("[Gate Engine] ✅ All gates passed — proceeding to Gemini synthesis.")
    return None
