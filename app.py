import streamlit as st
import pandas as pd
from main import run_financial_analysis
import yfinance as yf
from database import SessionLocal
from crud import add_stock, get_all_holdings, delete_stock

st.set_page_config(page_title="Financial Intelligence System", layout="wide")

st.title("🚀 Multi-Agent Financial Intelligence")

# Helper to get database session


# Use tabs for the Portfolio and Comparison features
tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Compare Stocks", "Portfolio"])

with tab1:
    st.header("Analyze a Ticker")
    ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL):", key="single").upper()
    
    if st.button("Run Intelligence Pipeline"):
        with st.spinner(f"Agents are analyzing {ticker}..."):
            # Calling your existing main.py function
            result = run_financial_analysis(ticker)
            
            # Displaying your existing node outputs
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${result['technical_indicators']['current_price']}")
            col2.metric("Recommendation", result['recommendation'])
            col3.metric("Risk Level", result['risk_level'])
            
            st.subheader("AI Reasoning")
            st.write(result['final_report'])

with tab2:
    st.header("Comparison Dashboard")
    col_a, col_b = st.columns(2)
    
    with col_a:
        t1 = st.text_input("Ticker 1", value="MSFT").upper()
    with col_b:
        t2 = st.text_input("Ticker 2", value="GOOGL").upper()
        
    if st.button("Compare"):
        res1 = run_financial_analysis(t1)
        res2 = run_financial_analysis(t2)
        
        # Side-by-side comparison table
        comparison_df = pd.DataFrame({
            "Metric": ["Recommendation", "RSI", "Trend", "Sharpe Ratio", "Max Drawdown"],
            t1: [res1['recommendation'], res1['technical_indicators']['rsi_14'], res1['technical_indicators']['trend'], res1['risk_metrics']['sharpe_ratio'], res1['risk_metrics']['max_drawdown']],
            t2: [res2['recommendation'], res2['technical_indicators']['rsi_14'], res2['technical_indicators']['trend'], res2['risk_metrics']['sharpe_ratio'], res2['risk_metrics']['max_drawdown']]
        })
        st.table(comparison_df)

with tab3:
    st.header("Portfolio Manager")
    
    # 1. Add Shares Panel
    with st.expander("➕ Add Shares", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_ticker = st.text_input("Ticker Symbol").upper().strip()
        with col2:
            new_shares = st.number_input("Number of Shares", min_value=0.0, step=0.1)
        
        if st.button("Add to Portfolio"):
            if not new_ticker or new_shares <= 0:
                st.error("Both Ticker and Number of Shares are required.")
            else:
                db = SessionLocal()
                try:
                    add_stock(db, new_ticker, new_shares)
                    st.success(f"Added {new_ticker} to portfolio!")
                    st.rerun() # Refresh the page to show new data
                except Exception as e:
                    st.error(f"Error adding stock: {e}")
                finally:
                    db.close()

    # 2. Portfolio Table
    st.subheader("Your Holdings")
    db = SessionLocal()
    holdings = get_all_holdings(db)
    db.close()

    if not holdings:
        st.info("Your portfolio is currently empty.")
    else:
        header_cols = st.columns([1, 1, 1, 1, 1, 1])
        header_cols[0].markdown("**Ticker**")
        header_cols[1].markdown("**Shares**")
        header_cols[2].markdown("**Date Added**")
        header_cols[3].markdown("**Buy Price**")
        header_cols[4].markdown("**Current Price**")
        header_cols[5].markdown("**Delete**")
        portfolio_data = []
        for h in holdings:
            # Fetch current price for each holding
            try:
                curr_price = yf.Ticker(h.ticker).history(period="1d")["Close"].iloc[-1]
            except:
                curr_price = 0.0

            portfolio_data.append({
                "Ticker": h.ticker,
                "Shares": h.shares,
                "Date Added": h.date_added.strftime("%Y-%m-%d"),
                "Buy Price": f"${h.buy_price:.2f}",
                "Current Price": f"${curr_price:.2f}",
                "Delete": h.ticker # Used for the button key
            })

        # Create the table display
        for row in portfolio_data:
            cols = st.columns([1, 1, 1, 1, 1, 1])
            cols[0].write(row["Ticker"])
            cols[1].write(row["Shares"])
            cols[2].write(row["Date Added"])
            cols[3].write(row["Buy Price"])
            cols[4].write(row["Current Price"])
            
            # Delete button logic
            if cols[5].button("🗑️", key=f"del_{row['Ticker']}"):
                db = SessionLocal()
                delete_stock(db, row["Ticker"])
                db.close()
                st.rerun()