import yfinance as yf

from core.state import FinancialState


def data_ingestion_node(state: FinancialState) -> dict:
    """
    Agent 1 — Data Ingestion
    Fetches 5 years of price history for the requested ticker via yfinance.
    Passes the raw DataFrame downstream; does NOT perform any calculations.
    """
    ticker = state["ticker"]
    print(f"--- [Agent 1] Fetching data for {ticker}... ---")

    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y", auto_adjust=True)

    if hist.empty:
        print(f"Error: No data found for {ticker}")
        return {"historical_data": None}

    start_date = hist.index.min().strftime("%Y-%m-%d")
    end_date = hist.index.max().strftime("%Y-%m-%d")

    print(f"--- [Agent 1] Success! {len(hist)} rows from {start_date} to {end_date} ---")

    return {
        "historical_data": hist,
        "start_date": start_date,
        "end_date": end_date,
    }
