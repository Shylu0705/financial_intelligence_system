import yfinance as yf
from datetime import datetime
from sqlalchemy.orm import Session
from database import Holding

def add_stock(db: Session, ticker: str, shares: float):
    """
    Adds a stock to the database. Fetches the current market price 
    via yfinance and uses the current system time.
    """
    ticker = ticker.upper().strip()
    
    # Fetch current price from yfinance
    stock_info = yf.Ticker(ticker)
    # Using 'fast_info' for speed or 'history' for the last close
    current_price = stock_info.history(period="1d")["Close"].iloc[-1]
    
    db_holding = Holding(
        ticker=ticker,
        shares=shares,
        buy_price=round(float(current_price), 2),
        date_added=datetime.now()
    )
    
    try:
        db.add(db_holding)
        db.commit()
        db.refresh(db_holding)
        return db_holding
    except Exception as e:
        db.rollback()
        raise e

def get_all_holdings(db: Session):
    """Retrieves all stocks in the portfolio."""
    return db.query(Holding).all()

def delete_stock(db: Session, ticker: str):
    """Removes a stock from the portfolio by ticker symbol."""
    item = db.query(Holding).filter(Holding.ticker == ticker.upper()).first()
    if item:
        db.delete(item)
        db.commit()
        return True
    return False