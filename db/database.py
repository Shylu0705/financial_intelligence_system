from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///portfolio_v2.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    FastAPI-compatible dependency that yields a database session and
    guarantees it is closed after the request, even on error.

    Usage in a route:
        @app.get("/...")
        def my_route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables defined under Base. Call once on startup."""
    from db import models  # noqa: F401 — import triggers table registration
    Base.metadata.create_all(bind=engine)
