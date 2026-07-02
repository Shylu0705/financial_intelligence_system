from datetime import datetime

import yfinance as yf
from sqlalchemy.orm import Session

from db.models import Holding


def add_stock(db: Session, ticker: str, shares: float) -> Holding:
    """Fetches the current market price and persists the holding."""
    ticker = ticker.upper().strip()
    current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

    holding = Holding(
        ticker=ticker,
        shares=shares,
        buy_price=round(float(current_price), 2),
        date_added=datetime.now(),
    )
    try:
        db.add(holding)
        db.commit()
        db.refresh(holding)
        return holding
    except Exception:
        db.rollback()
        raise


def get_holding(db: Session, ticker: str) -> Holding | None:
    """Returns the holding for a single ticker, or None if not in portfolio."""
    return db.query(Holding).filter(Holding.ticker == ticker.upper()).first()


def get_all_holdings(db: Session) -> list[Holding]:
    return db.query(Holding).all()


def delete_stock(db: Session, ticker: str) -> bool:
    item = db.query(Holding).filter(Holding.ticker == ticker.upper()).first()
    if item:
        db.delete(item)
        db.commit()
        return True
    return False
