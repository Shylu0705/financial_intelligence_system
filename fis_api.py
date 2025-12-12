from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from main import run_financial_analysis

# 1. Initialize FastAPI
app = FastAPI(
    title="Financial Intelligence System API",
    description="Multi-Agent AI System for Stock Analysis",
    version="1.0"
)

# 2. Define Request Model
class TickerRequest(BaseModel):
    ticker: str

# 3. Define the Endpoint
@app.post("/analyze", summary="Analyze a specific stock ticker")
async def analyze_stock(request: TickerRequest):
    """
    Triggers the multi-agent workflow for the provided ticker.
    Returns structured financial analysis including recommendation, risk level, and key drivers.
    """
    ticker = request.ticker.upper().strip()
    
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty")
    
    try:
        # Call your existing main function
        result = run_financial_analysis(ticker)
        
        # Check if data was actually retrieved
        if result.get("data_range", {}).get("start") is None:
             raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")
             
        return result
    
    except Exception as e:
        # Log the error internally (print to console)
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. Entry point for debugging (Optional)
if __name__ == "__main__":
    uvicorn.run("fis_api:app", host="127.0.0.1", port=8000, reload=True)