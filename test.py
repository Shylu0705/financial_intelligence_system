from core.workflow import run_financial_analysis

if __name__ == "__main__":
    ticker_symbol = input("Enter a NYSE ticker: ").upper().strip()
    print()
    result = run_financial_analysis(ticker_symbol)

    print("\n--- EXECUTIVE SUMMARY ---")
    print(result["final_report"])
