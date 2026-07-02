from sqlalchemy import Column, DateTime, Float, Integer, String

from db.database import Base


class Holding(Base):
    __tablename__ = "holdings"

    id         = Column(Integer, primary_key=True, index=True)
    ticker     = Column(String,  unique=True, index=True, nullable=False)
    shares     = Column(Float,   nullable=False)
    buy_price  = Column(Float,   nullable=False)
    date_added = Column(DateTime, nullable=False)
