"""
Entry point for running the Financial Intelligence System API server.

Start the server:
    python main.py
    -- or --
    uvicorn main:app --reload

Run the Streamlit UI (separate terminal, after the API is running):
    streamlit run ui/app.py

Run a quick CLI analysis:
    python test.py
"""

import uvicorn
from db.database import init_db
from api.routes import app  # noqa: F401 — re-export for `uvicorn main:app`

if __name__ == "__main__":
    init_db()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
