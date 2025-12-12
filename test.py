from main import run_financial_analysis

if __name__ == "__main__":
    ticker_symbol = input("Enter a nyse ticker: ")  # Try AMZN, GOOG, TSLA, AAPL, ORCL, MSFT, etc
    print("\n")
    result = run_financial_analysis(ticker_symbol)
    
    print("\n--- EXECUTIVE SUMMARY ---")

    print(result['final_report'])