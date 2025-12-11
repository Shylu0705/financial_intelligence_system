# Financial Intelligence System

A multi-agent financial analysis system that provides real-time stock analysis and trading recommendations for NYSE-listed stocks. The system uses LangGraph to orchestrate a pipeline of specialized agents that perform data ingestion, technical analysis, risk assessment, and AI-powered synthesis.

## Features

- **Data Ingestion**: Fetches 5 years of historical stock data using yfinance
- **Technical Analysis**: Calculates RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence) indicators
- **Fundamental Analysis**: Retrieves key metrics including P/E ratio, market cap, and sector information
- **Risk Assessment**: Computes volatility, Sharpe ratio, maximum drawdown, and current drawdown
- **AI-Powered Recommendations**: Uses Google Gemini AI to synthesize analysis and generate BUY/SELL/HOLD recommendations with detailed reasoning

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key (for AI synthesis)

## Installation

1. Clone the repository:
```bash
cd financial_intellligence_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory:
```bash
# .env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Note**: You can obtain a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

Run the analysis for any NYSE ticker symbol:

```bash
python test.py
```

When prompted, enter a ticker symbol (e.g., `AMZN`, `GOOG`, `TSLA`, `AAPL`, `ORCL`, `MSFT`, `NVDA`).

The system will:
1. Fetch historical data for the ticker
2. Perform technical and fundamental analysis
3. Calculate risk metrics
4. Generate an AI-powered recommendation with reasoning

### Example Output

```
--- EXECUTIVE SUMMARY ---
**Recommendation:** BUY
**Risk Level:** Medium

**Reasoning:**
[AI-generated analysis based on all metrics]

**Key Drivers:**
- Current Drawdown of -2.5%
- RSI of 65
- Bullish MACD trend
- ...
```

## API Status

⚠️ **Note**: The API interface is not yet implemented. The `fis_api.py` file is reserved for future API development. Currently, the system can only be used via the command-line interface through `test.py`.
