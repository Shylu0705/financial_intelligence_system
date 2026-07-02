import os
import sys

# Ensure the project root is on sys.path when Streamlit runs this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from db.crud import add_stock, delete_stock, get_all_holdings
from db.database import SessionLocal

# Base URL of the running FastAPI server — override via environment variable
# so this works in local dev (localhost) and deployed environments equally.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Financial Intelligence System", layout="wide")
st.title("🚀 Multi-Agent Financial Intelligence")


def call_analyze_api(ticker: str, owns_stock: bool = False, buy_price: float = 0.0, shares_held: float = 0.0) -> dict:
    """
    Calls the FastAPI /analyze endpoint and returns the result dict.

    TODO: Implement this function (5-8 lines).

    Things to consider:
    - Use requests.post() to POST to f"{API_BASE_URL}/analyze" with JSON body {"ticker": ticker}
    - Set a timeout — the pipeline can take 10-20 seconds; what's a sensible ceiling?
    - Raise an informative exception on non-200 status codes (use response.raise_for_status()
      or check response.status_code manually for a friendlier message)
    - Return response.json() on success

    Trade-off: raise_for_status() gives a generic HTTPError; checking status_code manually
    lets you show the user the API's own error detail message from response.json()["detail"].
    """
    response = requests.post(
        f"{API_BASE_URL}/analyze",
        json={"ticker": ticker, "owns_stock": owns_stock, "buy_price": buy_price, "shares_held": shares_held},
        timeout=180,
    )

    if response.status_code != 200:
        detail = response.json().get("detail", "Unknown error from API.")
        raise RuntimeError(f"[{response.status_code}] {detail}")

    return response.json()


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Stock Analysis", "Compare Stocks", "Portfolio", "Backtest"])

# ── Tab 1: Single ticker analysis ───────────────────────────────────────────
with tab1:
    st.header("Analyze a Ticker")
    ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL):", key="single").upper()

    st.caption("If this ticker is in your Portfolio tab, position context is loaded automatically.")
    owns_stock = st.toggle("I own this stock (manual override)")
    buy_price   = 0.0
    shares_held = 0.0
    if owns_stock:
        pi1, pi2 = st.columns(2)
        buy_price   = pi1.number_input("My buy price ($ per share)", min_value=0.01, format="%.2f")
        shares_held = pi2.number_input("Shares I hold", min_value=0.01, format="%.4f")

    if st.button("Run Intelligence Pipeline"):
        with st.spinner(f"Agents are analysing {ticker}..."):
            try:
                result = call_analyze_api(ticker, owns_stock, buy_price, shares_held)
                if result.get("owns_stock"):
                    curr   = result["technical_indicators"]["current_price"]
                    bp     = result.get("buy_price", 0)
                    sh     = result.get("shares_held", 0)
                    source = result.get("portfolio_source", "manual")
                    pnl_pct = (curr - bp) / bp * 100 if bp else 0
                    pnl_val = (curr - bp) * sh if bp else 0
                    color   = "green" if pnl_pct >= 0 else "red"
                    label   = "Portfolio position (from Portfolio tab)" if source == "db" else "Portfolio position (manual)"
                    st.info(
                        f"**{label}:** {sh} shares @ ${bp:.2f}  ·  "
                        f"Current: ${curr:.2f}  ·  "
                        f"P&L: :{color}[{pnl_pct:+.1f}% (${pnl_val:+,.0f})]"
                    )

                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Price",            f"${result['technical_indicators']['current_price']:.2f}")
                col2.metric("Recommendation",   result["recommendation"])
                col3.metric("Risk Level",       result["risk_level"])
                col4.metric("Sentiment",        f"{result['sentiment_label']} ({result['sentiment_score']:+.2f})")
                col5.metric("Chart Bias",       result.get("chart_bias", "N/A"))
                col6.metric("Sector RS",        f"{result.get('sector_label', 'N/A')} ({result.get('relative_strength', 0):+.1f}%)")

                # Macro context
                st.subheader("Macro & Market Context")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Fed Funds Rate",  f"{result.get('fed_funds_rate', 'N/A')}%")
                mc2.metric("CPI",             f"{result.get('cpi', 'N/A')}")
                mc3.metric("10Y Treasury",    f"{result.get('treasury_10y', 'N/A')}%")

                bm1, bm2 = st.columns(2)
                bm1.metric("vs S&P 500",      f"{result.get('vs_spy_label', 'N/A')} ({result.get('vs_spy', 0):+.1f}%)")
                bm2.metric("vs NASDAQ 100",   f"{result.get('vs_qqq_label', 'N/A')} ({result.get('vs_qqq', 0):+.1f}%)")

                st.caption(result.get("rate_environment", ""))
                st.caption(result.get("inflation_environment", ""))

                st.subheader("News Sentiment")
                st.write(result["sentiment_summary"])

                st.subheader("Chart Pattern Analysis")
                ta = result.get("technical_indicators", {})
                adx_val    = ta.get("adx_14", "N/A")
                adx_regime = ta.get("adx_regime", "N/A")
                plus_di    = ta.get("plus_di", "N/A")
                minus_di   = ta.get("minus_di", "N/A")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Chart Confidence", result.get("chart_confidence", "N/A"))
                cc2.metric("ADX-14", f"{adx_val} ({adx_regime})")
                cc3.metric("+DI / -DI", f"{plus_di} / {minus_di}")

                if result.get("chart_patterns_detected"):
                    st.markdown("**Patterns detected:** " + " · ".join(result["chart_patterns_detected"]))
                st.write(result.get("chart_analysis", ""))

                fib = result.get("fib_levels", {})
                if fib:
                    with st.expander("📐 Fibonacci Retracement Levels (1-Year)", expanded=False):
                        fib_cols = st.columns(len(fib))
                        for col, (label, price) in zip(fib_cols, fib.items()):
                            col.metric(label, f"${price:.2f}")

                # Earnings warning banner
                warning = result.get("earnings_warning", "")
                if "⚠️" in warning:
                    st.warning(warning)
                elif warning:
                    st.info(warning)

                # Quarterly earnings table
                if result.get("earnings_history"):
                    st.subheader("Quarterly Financials")
                    st.dataframe(
                        pd.DataFrame(result["earnings_history"]).rename(columns={
                            "quarter":          "Quarter",
                            "sales":            "Sales",
                            "net_profit":       "Net Profit",
                            "dividend":         "Dividend",
                            "equity":           "Equity",
                            "reserves_surplus": "Reserves & Surplus",
                            "networth":         "Networth",
                            "debt":             "Debt",
                            "share_price":      "Share Price",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Decision tree path
                if result.get("tree_path"):
                    with st.expander("🌳 Decision Tree Path", expanded=False):
                        for step in result["tree_path"]:
                            st.markdown(f"- {step}")

                # Price recommendation
                st.subheader("Price Levels")
                entry_low  = result.get("entry_low",   0.0)
                entry_high = result.get("entry_high",  0.0)
                stop       = result.get("stop_loss",   0.0)
                t1         = result.get("target_1",    0.0)
                t2         = result.get("target_2",    0.0)
                rr         = result.get("risk_reward", 0.0)
                note       = result.get("price_rec_note", "")

                if entry_low or stop or t1:
                    pr1, pr2, pr3, pr4, pr5 = st.columns(5)
                    pr1.metric("Entry Zone",   f"${entry_low:.2f} – ${entry_high:.2f}" if entry_low else "—")
                    pr2.metric("Stop Loss",    f"${stop:.2f}" if stop else "—")
                    pr3.metric("Target 1",     f"${t1:.2f}"  if t1   else "—")
                    pr4.metric("Target 2",     f"${t2:.2f}"  if t2   else "—")
                    pr5.metric("Risk / Reward",f"{rr:.1f}:1" if rr   else "—")
                if note:
                    st.caption(note)

                st.subheader("AI Reasoning")
                st.write(result["final_report"])
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ── Tab 2: Side-by-side comparison ──────────────────────────────────────────
with tab2:
    st.header("Comparison Dashboard")
    col_a, col_b = st.columns(2)
    with col_a:
        t1 = st.text_input("Ticker 1", value="MSFT").upper()
    with col_b:
        t2 = st.text_input("Ticker 2", value="GOOGL").upper()

    if st.button("Compare"):
        with st.spinner("Running both pipelines..."):
            try:
                res1 = call_analyze_api(t1)
                res2 = call_analyze_api(t2)

                comparison_df = pd.DataFrame({
                    "Metric": ["Recommendation", "RSI", "Trend", "Sharpe Ratio", "Max Drawdown"],
                    t1: [
                        res1["recommendation"],
                        res1["technical_indicators"]["rsi_14"],
                        res1["technical_indicators"]["trend"],
                        res1["risk_metrics"]["sharpe_ratio"],
                        res1["risk_metrics"]["max_drawdown"],
                    ],
                    t2: [
                        res2["recommendation"],
                        res2["technical_indicators"]["rsi_14"],
                        res2["technical_indicators"]["trend"],
                        res2["risk_metrics"]["sharpe_ratio"],
                        res2["risk_metrics"]["max_drawdown"],
                    ],
                })
                st.table(comparison_df)
            except Exception as e:
                st.error(f"Comparison failed: {e}")

# ── Tab 3: Portfolio manager ─────────────────────────────────────────────────
with tab3:
    st.header("Portfolio Manager")

    with st.expander("➕ Add Shares", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_ticker = st.text_input("Ticker Symbol").upper().strip()
        with col2:
            new_shares = st.number_input("Number of Shares", min_value=0.0, step=0.1)

        if st.button("Add to Portfolio"):
            if not new_ticker or new_shares <= 0:
                st.error("Both ticker and number of shares are required.")
            else:
                db = SessionLocal()
                try:
                    add_stock(db, new_ticker, new_shares)
                    st.success(f"Added {new_ticker} to portfolio!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding stock: {e}")
                finally:
                    db.close()

    st.subheader("Your Holdings")
    db = SessionLocal()
    holdings = get_all_holdings(db)
    db.close()

    if not holdings:
        st.info("Your portfolio is currently empty.")
    else:
        header_cols = st.columns([1, 1, 1, 1, 1, 1])
        for col, label in zip(header_cols, ["Ticker", "Shares", "Date Added", "Buy Price", "Current Price", "Delete"]):
            col.markdown(f"**{label}**")

        for h in holdings:
            try:
                curr_price = yf.Ticker(h.ticker).history(period="1d")["Close"].iloc[-1]
            except Exception:
                curr_price = 0.0

            cols = st.columns([1, 1, 1, 1, 1, 1])
            cols[0].write(h.ticker)
            cols[1].write(h.shares)
            cols[2].write(h.date_added.strftime("%Y-%m-%d"))
            cols[3].write(f"${h.buy_price:.2f}")
            cols[4].write(f"${curr_price:.2f}")

            if cols[5].button("🗑️", key=f"del_{h.ticker}"):
                db = SessionLocal()
                delete_stock(db, h.ticker)
                db.close()
                st.rerun()

# ── Tab 4: Backtest ──────────────────────────────────────────────────────────
with tab4:
    st.header("Signal Backtest")
    st.caption(
        "Replays RSI/MACD/MA/ATR signals historically — no LLM calls. "
        "Signals checked weekly; stop loss enforced daily."
    )

    bt1, bt2, bt3 = st.columns(3)
    bt_ticker = bt1.text_input("Ticker", value="NVDA", key="bt_ticker").upper().strip()
    bt_start  = bt2.date_input("Start date", value=pd.to_datetime("2022-01-01"), key="bt_start")
    bt_end    = bt3.date_input("End date",   value=pd.to_datetime("2025-01-01"), key="bt_end")

    with st.expander("Advanced settings", expanded=False):
        adv1, adv2 = st.columns(2)
        max_hold  = adv1.number_input("Max holding period (days)", min_value=10, max_value=365, value=100, step=10)
        atr_mult  = adv2.number_input("Stop loss ATR multiplier",  min_value=0.5, max_value=5.0, value=1.5, step=0.5)

    if st.button("Run Backtest"):
        with st.spinner(f"Backtesting {bt_ticker}..."):
            try:
                bt_resp = requests.post(
                    f"{API_BASE_URL}/backtest",
                    json={
                        "ticker":        bt_ticker,
                        "start_date":    str(bt_start),
                        "end_date":      str(bt_end),
                        "max_hold_days": int(max_hold),
                        "atr_mult":      float(atr_mult),
                    },
                    timeout=120,
                )
                if bt_resp.status_code != 200:
                    st.error(bt_resp.json().get("detail", "Backtest failed."))
                else:
                    bt = bt_resp.json()
                    m  = bt["metrics"]

                    # ── Metrics tiles ────────────────────────────────────────
                    st.subheader("Performance Summary")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Strategy Return",  f"{m.get('total_return_pct', 0):+.1f}%")
                    m2.metric("Buy & Hold",        f"{m.get('buy_hold_return_pct', 0):+.1f}%")
                    m3.metric("Win Rate",          f"{m.get('win_rate_pct', 0):.1f}%")
                    m4.metric("Sharpe Ratio",      f"{m.get('sharpe_ratio', 0):.2f}")
                    m5.metric("Max Drawdown",      f"{m.get('max_drawdown_pct', 0):.1f}%")

                    m6, m7, m8, m9, m10 = st.columns(5)
                    m6.metric("Trades",            m.get("num_trades", 0))
                    m7.metric("Avg Hold (days)",   m.get("avg_hold_days", 0))
                    m8.metric("Avg Win",           f"{m.get('avg_win_pct', 0):+.1f}%")
                    m9.metric("Avg Loss",          f"{m.get('avg_loss_pct', 0):+.1f}%")
                    m10.metric("Profit Factor",    m.get("profit_factor", 0))

                    # ── Equity curve ─────────────────────────────────────────
                    st.subheader("Equity Curve")
                    eq_df = pd.DataFrame({
                        "Strategy":    bt["equity_curve"],
                        "Buy & Hold":  bt["bh_curve"],
                    })
                    eq_df.index = pd.to_datetime(eq_df.index)
                    eq_df = eq_df.sort_index()
                    st.line_chart(eq_df, height=350)

                    # ── Trade log ────────────────────────────────────────────
                    if bt.get("trades"):
                        st.subheader(f"Trade Log ({len(bt['trades'])} trades)")
                        trades_df = pd.DataFrame(bt["trades"]).rename(columns={
                            "entry_date":   "Entry Date",
                            "exit_date":    "Exit Date",
                            "entry_price":  "Entry $",
                            "exit_price":   "Exit $",
                            "exit_reason":  "Exit Reason",
                            "return_pct":   "Return %",
                            "holding_days": "Days Held",
                        })
                        # Colour-code returns
                        st.dataframe(
                            trades_df.style.applymap(
                                lambda v: "color: green" if isinstance(v, float) and v > 0
                                     else ("color: red" if isinstance(v, float) and v < 0 else ""),
                                subset=["Return %"],
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No trades were generated in this period.")

            except Exception as e:
                st.error(f"Backtest error: {e}")
