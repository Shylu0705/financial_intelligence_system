import json
import numpy as np
import pandas as pd
from datetime import datetime
from langgraph.graph import StateGraph, END
from state import FinancialState
from nodes import data_ingestion_node, analysis_node, risk_node, synthesis_node

# --- Helper to make data JSON serializable ---
class FinancialEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super(FinancialEncoder, self).default(obj)

# --- Main Workflow Definition ---
def build_workflow():
    workflow = StateGraph(FinancialState)
    
    # Add Nodes
    workflow.add_node("data_agent", data_ingestion_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("risk_agent", risk_node)
    workflow.add_node("synthesis_agent", synthesis_node)
    
    # Define Edges
    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "risk_agent")
    workflow.add_edge("risk_agent", "synthesis_agent")
    workflow.add_edge("synthesis_agent", END)
    
    return workflow.compile()

# --- Execution Function ---
def run_financial_analysis(ticker: str):
    """
    Runs the multi-agent pipeline for a given ticker and saves the result to JSON.
    """
    print(f"\n🚀 Starting Analysis for: {ticker}")
    print("=" * 50)
    
    # 1. Compile and Run Graph
    app = build_workflow()
    final_state = app.invoke({"ticker": ticker})
    
    # 2. Extract Key Data
    output_data = {
        "ticker": final_state["ticker"],
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": {
            "start": final_state.get("start_date"),
            "end": final_state.get("end_date")
        },
        "fundamental_metrics": final_state.get("fundamental_metrics"),
        "technical_indicators": final_state.get("technical_indicators"),
        "risk_metrics": final_state.get("risk_metrics"),
        "final_report": final_state.get("final_report")
    }
      
    print(f"✅ Analysis Complete!")
    print("=" * 50)
    
    return output_data

# --- Entry Point ---
# if __name__ == "__main__":
#     # Example Usage
#     ticker_symbol = "NVDA"  # Try changing to any other NYSE ticker
#     result = run_financial_analysis(ticker_symbol)
    
#     # Print the report part to console for verification
#     print("\n--- EXECUTIVE SUMMARY ---")

#     # Create a clean string version for the final report text
#     formatted_text = (
#         f"**Recommendation:** {result['recommendation']}\n"
#         f"**Risk Level:** {result['risk_level']}\n\n"
#         f"**Reasoning:**\n{result['reasoning']}\n\n"
#         f"**Key Drivers:**\n- " + "\n- ".join(result['key_drivers'])
#     )
#     print(formatted_text)