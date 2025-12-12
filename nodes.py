import yfinance as yf
import pandas as pd
from datetime import datetime
from state import FinancialState
import numpy as np
import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()

def data_ingestion_node(state: FinancialState):
    """
    Agent 1: Data Ingestion
    Fetches 5 years of historical data using yfinance.
    """
    ticker = state["ticker"]
    print(f"--- [Agent 1] Fetching data for {ticker}... ---")
    
    # Fetch 5 years of history
    # We use 'auto_adjust=True' to handle stock splits/dividends automatically
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y", auto_adjust=True)
    
    # Basic validation
    if hist.empty:
        print(f"Error: No data found for {ticker}")
        return {"historical_data": None}
    
    # Store dates for reference
    start_date = hist.index.min().strftime('%Y-%m-%d')
    end_date = hist.index.max().strftime('%Y-%m-%d')
    
    print(f"--- [Agent 1] Success! Retrieved {len(hist)} rows from {start_date} to {end_date} ---")
    
    # Return the update to the state
    return {
        "historical_data": hist,
        "start_date": start_date,
        "end_date": end_date
    }


def analysis_node(state: FinancialState):
    """
    Agent 2: Technical & Fundamental Analysis
    Calculates RSI, MACD, and fetches basic fundamentals (P/E, Market Cap).
    """
    ticker = state["ticker"]
    data = state["historical_data"]
    
    print(f"--- [Agent 2] Analyzing {ticker}... ---")

    # 1. Fundamental Analysis (Basic)
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # diverse keys often appear in yfinance info; we use .get() for safety
    fundamentals = {
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "sector": info.get("sector", "Unknown")
    }
    
    # 2. Technical Analysis (Manual Calculation)
    # We work on a copy to avoid SettingWithCopy warnings
    df = data.copy()
    
    # --- RSI (Relative Strength Index) Calculation ---
    # Formula: 100 - (100 / (1 + RS))
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # --- MACD (Moving Average Convergence Divergence) ---
    # Formula: 12-day EMA - 26-day EMA
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    
    df["MACD_Line"] = ema_12 - ema_26
    df["Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    
    # Get the most recent values
    latest = df.iloc[-1]
    
    technicals = {
        "current_price": latest["Close"],
        "rsi_14": round(latest["RSI"], 2),
        "macd_line": round(latest["MACD_Line"], 2),
        "signal_line": round(latest["Signal_Line"], 2),
        "trend": "Bullish" if latest["MACD_Line"] > latest["Signal_Line"] else "Bearish"
    }

    print(f"--- [Agent 2] Analysis Complete. RSI: {technicals['rsi_14']} | Trend: {technicals['trend']} ---")

    return {
        "fundamental_metrics": fundamentals,
        "technical_indicators": technicals
    }

def risk_node(state: FinancialState):
    """
    Agent 3: Risk Management
    Calculates Volatility, Max Drawdown, Sharpe Ratio, AND Current Drawdown.
    """
    data = state["historical_data"]
    ticker = state["ticker"]
    
    print(f"--- [Agent 3] Assessing Risk for {ticker}... ---")
    
    # Calculate Daily Returns
    df = data.copy()
    df["Returns"] = df["Close"].pct_change().dropna()
    
    # 1. Max Drawdown (Historical - 5 Years)
    cumulative_returns = (1 + df["Returns"]).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 2. Current Drawdown (Right Now)
    # The last value in the drawdown series is the current drawdown
    current_drawdown = drawdown.iloc[-1]
    
    # 3. Sharpe Ratio
    risk_free_rate = 0.04
    mean_return = df["Returns"].mean() * 252
    std_dev = df["Returns"].std() * (252 ** 0.5)
    
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    
    risk_metrics = {
        "max_drawdown": round(max_drawdown, 4),      # Historical worst case
        "current_drawdown": round(current_drawdown, 4), # Relevant for today's decision
        "volatility": round(std_dev, 4),
        "sharpe_ratio": round(sharpe_ratio, 2)
    }
    
    print(f"--- [Agent 3] Risk Analysis Complete. Curr DD: {risk_metrics['current_drawdown']:.2%} | Max DD: {risk_metrics['max_drawdown']:.2%} ---")
    
    return {"risk_metrics": risk_metrics}


class FinancialReport(BaseModel):
    recommendation: str = Field(description="The final trading recommendation: 'BUY', 'SELL', or 'HOLD'")
    reasoning: str = Field(description="A concise explanation (under 100 words) justifying the recommendation")
    key_drivers: list[str] = Field(description="A list of 3-5 specific data points that influenced the decision (e.g. 'High RSI', 'Low Sharpe Ratio')")
    risk_level: str = Field(description="The assessed risk level: 'Low', 'Medium', 'High', or 'Extreme'")

# --- The Synthesis Node ---
def synthesis_node(state: FinancialState):
    """
    Agent 4: Insight Synthesis
    Uses Gemini to aggregate all data and generate a structured JSON report.
    """
    print(f"--- [Agent 4] Synthesizing Report... ---")
    
    # 1. Initialize Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 2. Setup the Parser
    parser = PydanticOutputParser(pydantic_object=FinancialReport)
    
    # 3. Prepare the Prompt with Format Instructions
    template = """You are a Senior Investment Strategist at a top-tier hedge fund.
    
    ### DATA INPUTS:
    1. Fundamental Data:
    {fundamentals}
    
    2. Technical Indicators:
    {technicals}
    
    3. Risk Metrics:
    {risk}
    
    ### DECISION FRAMEWORK:
    - **BUY**: Strong technicals (Bullish trend, RSI < 70) AND reasonable valuation OR strong growth momentum.
    - **HOLD**: Conflicting signals (e.g., Good technicals but High P/E).
    - **SELL**: Weak technicals (Bearish trend) AND (High P/E OR High Current Drawdown).
    
    ### IMPORTANT RISK CONTEXT:
    - "Max Drawdown" is historical (5-year window). Do not penalize a stock solely for a crash that happened years ago (e.g., 2022) if it has since recovered.
    - Focus on "Current Drawdown" and "Volatility" for immediate risk.
    - A high P/E ratio is acceptable for high-growth tech stocks if momentum is strong.
    
    ### OUTPUT FORMAT INSTRUCTIONS:
    You must strictly output a single valid JSON object. Do not include Markdown formatting. 
    Follow this exact schema:
    
    {{
        "recommendation": "BUY, SELL, or HOLD",
        "risk_level": "Low, Medium, High, or Extreme",
        "reasoning": "A concise explanation (under 100 words). Focus on the balance between current momentum and valuation.",
        "key_drivers": [
            "List 3-5 specific data points",
            "e.g. 'Current Drawdown of -5%'",
            "e.g. 'RSI of 65'"
        ]
    }}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Format the input
    messages = prompt.format_messages(
        ticker=state["ticker"],
        fundamentals=state["fundamental_metrics"],
        technicals=state["technical_indicators"],
        risk=state["risk_metrics"],
        format_instructions=parser.get_format_instructions()
    )
    
    # 5. Invoke and Parse
    try:
        response = llm.invoke(messages)
        # The parser automatically extracts the JSON and converts it to a python object
        parsed_report = parser.parse(response.content)
        
        # Convert back to dict for state storage
        report_dict = parsed_report.dict()
        
        final_report = (
            f"**Recommendation:** {report_dict['recommendation']}\n"
            f"**Risk Level:** {report_dict['risk_level']}\n\n"
            f"**Reasoning:**\n{report_dict['reasoning']}\n\n"
            f"**Key Drivers:**\n- " + "\n- ".join(report_dict['key_drivers'])
        )

        print(f"--- [Agent 4] Structured Report Generated! ---")
        
        report_dict["final_report"] = final_report

        return report_dict

    except Exception as e:
        print(f"Error calling LLM or Parsing: {e}")
        return {
            "recommendation": "ERROR",
            "final_report": "Error generating structured report."
        }