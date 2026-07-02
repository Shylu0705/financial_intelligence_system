import os

from core.state import FinancialState

# Risk-free rate sourced from environment so it can be updated without code changes.
# Defaults to 4% (approximate 2024 US Treasury yield) if not set.
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", 0.04))


def risk_node(state: FinancialState) -> dict:
    """
    Agent 3 — Risk Management
    Computes annualised volatility, max drawdown (5-year), current drawdown,
    and Sharpe ratio from the historical price series.
    """
    ticker = state["ticker"]
    data = state["historical_data"]

    print(f"--- [Agent 3] Assessing risk for {ticker}... ---")

    df = data.copy()
    df["Returns"] = df["Close"].pct_change().dropna()

    # Max Drawdown (5-year window)
    cumulative = (1 + df["Returns"]).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Current Drawdown (most recent bar)
    current_drawdown = drawdown.iloc[-1]

    # Annualised Sharpe Ratio
    mean_return = df["Returns"].mean() * 252
    std_dev = df["Returns"].std() * (252 ** 0.5)
    sharpe_ratio = (mean_return - RISK_FREE_RATE) / std_dev if std_dev != 0 else 0

    risk_metrics = {
        "max_drawdown":     round(max_drawdown, 4),
        "current_drawdown": round(current_drawdown, 4),
        "volatility":       round(std_dev, 4),
        "sharpe_ratio":     round(sharpe_ratio, 2),
    }

    print(
        f"--- [Agent 3] Curr DD: {risk_metrics['current_drawdown']:.2%} | "
        f"Max DD: {risk_metrics['max_drawdown']:.2%} ---"
    )

    return {"risk_metrics": risk_metrics}
