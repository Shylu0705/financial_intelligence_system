from main import run_financial_analysis

if __name__ == "__main__":
    ticker_symbol = input("Enter a nyse ticker: ")  # Try AMZN, GOOG, TSLA, AAPL, ORCL, MSFT, etc
    print("\n")
    result = run_financial_analysis(ticker_symbol)
    
    print("\n--- EXECUTIVE SUMMARY ---")

    formatted_text = (
        f"**Recommendation:** {result['recommendation']}\n"
        f"**Risk Level:** {result['risk_level']}\n\n"
        f"**Reasoning:**\n{result['reasoning']}\n\n"
        f"**Key Drivers:**\n- " + "\n- ".join(result['key_drivers'])
    )
    print(formatted_text)