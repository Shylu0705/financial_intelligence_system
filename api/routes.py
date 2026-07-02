import asyncio

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.backtester import run_backtest
from core.workflow import run_financial_analysis
from db.crud import add_stock, delete_stock, get_all_holdings, get_holding
from db.database import get_db, init_db

app = FastAPI(
    title="Financial Intelligence System",
    description="Multi-Agent AI System for Stock Analysis",
    version="2.0",
)


@app.on_event("startup")
def on_startup():
    """Initialise DB tables when the API server starts."""
    init_db()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TickerRequest(BaseModel):
    ticker:      str
    owns_stock:  bool  = False
    buy_price:   float = 0.0
    shares_held: float = 0.0

class HoldingRequest(BaseModel):
    ticker: str
    shares: float

class BacktestRequest(BaseModel):
    ticker:        str
    start_date:    str
    end_date:      str
    max_hold_days: int   = 100
    atr_mult:      float = 1.5


# ---------------------------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", summary="Run the multi-agent pipeline for a ticker")
async def analyze_stock(request: TickerRequest, db: Session = Depends(get_db)):
    """
    Triggers the multi-agent workflow and returns structured analysis.

    Portfolio context resolution (priority order):
      1. DB lookup — if the ticker is in the portfolio, use stored buy_price + shares.
      2. Manual override — if the request body sets owns_stock=True with values, use those.
      3. Default — no position context.

    The pipeline is CPU/IO-bound, so it runs in a thread pool via
    asyncio.to_thread() to avoid blocking the FastAPI event loop.
    """
    ticker = request.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    # DB lookup takes priority over the manual toggle
    holding = get_holding(db, ticker)
    if holding:
        owns_stock  = True
        buy_price   = holding.buy_price
        shares_held = holding.shares
        print(f"[API] Portfolio position found for {ticker}: {shares_held} shares @ ${buy_price:.2f}")
    else:
        owns_stock  = request.owns_stock
        buy_price   = request.buy_price
        shares_held = request.shares_held

    try:
        result = await asyncio.to_thread(
            run_financial_analysis,
            ticker,
            owns_stock,
            buy_price,
            shares_held,
        )
        # Tell the UI whether the position came from DB or manual toggle
        result["portfolio_source"] = "db" if holding else ("manual" if owns_stock else "none")

        if result.get("data_range", {}).get("start") is None:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for ticker '{ticker}'.",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Unhandled error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Backtest endpoint
# ---------------------------------------------------------------------------

@app.post("/backtest", summary="Run a deterministic backtest for a ticker")
async def backtest_stock(request: BacktestRequest):
    """
    Replays RSI/MACD/MA/ATR signals historically without LLM calls.
    Returns metrics, a trade log, and daily equity + buy-and-hold curves.
    """
    ticker = request.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")

    try:
        result = await asyncio.to_thread(
            run_backtest,
            ticker,
            request.start_date,
            request.end_date,
            request.max_hold_days,
            request.atr_mult,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Portfolio endpoints
# ---------------------------------------------------------------------------

@app.get("/portfolio", summary="List all holdings")
def list_holdings(db: Session = Depends(get_db)):
    holdings = get_all_holdings(db)
    return [
        {
            "ticker":     h.ticker,
            "shares":     h.shares,
            "buy_price":  h.buy_price,
            "date_added": h.date_added.strftime("%Y-%m-%d"),
        }
        for h in holdings
    ]


@app.post("/portfolio", summary="Add a holding")
def add_holding(request: HoldingRequest, db: Session = Depends(get_db)):
    if not request.ticker or request.shares <= 0:
        raise HTTPException(status_code=400, detail="Valid ticker and shares > 0 required.")
    try:
        holding = add_stock(db, request.ticker, request.shares)
        return {"message": f"Added {holding.ticker}", "buy_price": holding.buy_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/portfolio/{ticker}", summary="Remove a holding")
def remove_holding(ticker: str, db: Session = Depends(get_db)):
    deleted = delete_stock(db, ticker)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"'{ticker}' not found in portfolio.")
    return {"message": f"Deleted {ticker.upper()}"}
