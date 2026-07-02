"""
Decision tree hard gates — non-negotiable rules that override Gemini's judgment.

To experiment:
  - Flip `enabled` True/False to activate/deactivate a gate
  - Adjust threshold values without touching any agent logic
  - Recommended starting config: earnings_proximity, revenue_decline, pipeline_failure ON
"""

GATE_CONFIG = {

    # Gate 1 — Earnings Proximity
    # Forces HOLD if an earnings report is imminent (binary event risk)
    "earnings_proximity": {
        "enabled":        True,
        "threshold_days": 7,       # days_to_earnings <= this → HOLD
    },

    # Gate 2 — Consecutive Revenue Decline
    # Caps recommendation at HOLD if revenue has fallen N quarters in a row
    # Prevents BUY signals on fundamentally deteriorating businesses
    "revenue_decline": {
        "enabled":               True,
        "consecutive_quarters":  3,    # how many declining quarters trigger the gate
    },

    # Gate 3 — Extreme Current Drawdown
    # Forces HOLD if stock is in freefall (catching a falling knife)
    "extreme_drawdown": {
        "enabled":       False,
        "threshold_pct": -30.0,   # current_drawdown <= this % → HOLD
    },

    # Gate 4 — Extreme Volatility
    # Forces HOLD if annualised volatility is too high to make a reliable call
    "extreme_volatility": {
        "enabled":       False,
        "threshold_pct": 80.0,    # annualised_volatility >= this % → HOLD
    },

    # Gate 5 — Negative Networth
    # Never BUY a company that owes more than it owns
    "negative_networth": {
        "enabled": False,
    },

    # Gate 6 — Extreme Macro Environment
    # Caps at HOLD when both rates AND valuation are dangerously high together
    "extreme_macro": {
        "enabled":      False,
        "min_fed_rate": 6.0,    # fed_funds_rate >= this %
        "min_pe_ratio": 40.0,   # AND pe_ratio >= this → HOLD
    },

    # Gate 7 — Pipeline Failure
    # Forces HOLD if any critical agent returned an error or missing data
    # Always keep this ON — never make a call on incomplete information
    "pipeline_failure": {
        "enabled": True,
    },
}
