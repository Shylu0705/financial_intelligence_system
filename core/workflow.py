from datetime import datetime

from langgraph.graph import END, StateGraph

from core.state import FinancialState
from agents.data_agent import data_ingestion_node
from agents.analysis_agent import analysis_node
from agents.risk_agent import risk_node
from agents.sector_agent import sector_agent_node
from agents.macro_agent import macro_agent_node
from agents.sentiment_agent import sentiment_agent_node
from agents.chart_vision_agent import chart_vision_agent_node
from agents.earnings_agent import earnings_agent_node
from agents.synthesis_agent import synthesis_node
from agents.price_recommendation_agent import price_recommendation_node


def build_workflow():
    """
    Assembles the four-agent LangGraph pipeline and returns a compiled graph.
    The pipeline is strictly linear:
        data_agent → analysis_agent → risk_agent → synthesis_agent → END
    """
    workflow = StateGraph(FinancialState)

    workflow.add_node("data_agent", data_ingestion_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("risk_agent", risk_node)
    workflow.add_node("sector_agent", sector_agent_node)
    workflow.add_node("macro_agent", macro_agent_node)
    workflow.add_node("sentiment_agent", sentiment_agent_node)
    workflow.add_node("chart_vision_agent", chart_vision_agent_node)
    workflow.add_node("earnings_agent", earnings_agent_node)
    workflow.add_node("synthesis_agent", synthesis_node)
    workflow.add_node("price_rec_agent", price_recommendation_node)

    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "risk_agent")
    workflow.add_edge("risk_agent", "sector_agent")
    workflow.add_edge("sector_agent", "macro_agent")
    workflow.add_edge("macro_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "chart_vision_agent")
    workflow.add_edge("chart_vision_agent", "earnings_agent")
    workflow.add_edge("earnings_agent", "synthesis_agent")
    workflow.add_edge("synthesis_agent", "price_rec_agent")
    workflow.add_edge("price_rec_agent", END)

    return workflow.compile()


def run_financial_analysis(
    ticker: str,
    owns_stock:  bool  = False,
    buy_price:   float = 0.0,
    shares_held: float = 0.0,
) -> dict:
    """
    Runs the full multi-agent pipeline for a given ticker.
    Portfolio context (owns_stock / buy_price / shares_held) is optional —
    when provided it is seeded into the initial state so the synthesis agent
    can frame its recommendation relative to the user's cost basis.
    Returns a plain dict safe for JSON serialisation.
    """
    print(f"\n🚀 Starting Analysis for: {ticker}")
    print("=" * 50)

    app = build_workflow()
    final_state = app.invoke({
        "ticker":      ticker,
        "owns_stock":  owns_stock,
        "buy_price":   buy_price,
        "shares_held": shares_held,
    })

    output = {
        "ticker":         final_state["ticker"],
        "analysis_date":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": {
            "start": final_state.get("start_date"),
            "end":   final_state.get("end_date"),
        },
        # Agent 2: Technical & Fundamental
        "fundamental_metrics":  final_state.get("fundamental_metrics"),
        "technical_indicators": final_state.get("technical_indicators"),
        # Agent 3: Risk
        "risk_metrics":         final_state.get("risk_metrics"),
        # Agent 4: Sector
        "sector_etf":           final_state.get("sector_etf"),
        "stock_3m_return":      final_state.get("stock_3m_return"),
        "sector_3m_return":     final_state.get("sector_3m_return"),
        "relative_strength":    final_state.get("relative_strength"),
        "sector_label":         final_state.get("sector_label"),
        # Agent 5: Macro
        "fed_funds_rate":       final_state.get("fed_funds_rate"),
        "cpi":                  final_state.get("cpi"),
        "treasury_10y":         final_state.get("treasury_10y"),
        "rate_environment":     final_state.get("rate_environment"),
        "inflation_environment":final_state.get("inflation_environment"),
        "spy_3m_return":        final_state.get("spy_3m_return"),
        "qqq_3m_return":        final_state.get("qqq_3m_return"),
        "vs_spy":               final_state.get("vs_spy"),
        "vs_qqq":               final_state.get("vs_qqq"),
        "vs_spy_label":         final_state.get("vs_spy_label"),
        "vs_qqq_label":         final_state.get("vs_qqq_label"),
        # Agent 6: Sentiment
        "news_headlines":       final_state.get("news_headlines"),
        "sentiment_score":      final_state.get("sentiment_score"),
        "sentiment_label":      final_state.get("sentiment_label"),
        "sentiment_summary":    final_state.get("sentiment_summary"),
        # Agent 7: Chart Vision
        "chart_bias":              final_state.get("chart_bias"),
        "chart_confidence":        final_state.get("chart_confidence"),
        "chart_patterns_detected": final_state.get("chart_patterns_detected"),
        "chart_analysis":          final_state.get("chart_analysis"),
        "fib_levels":              final_state.get("fib_levels"),
        # Agent 8: Earnings
        "earnings_history":     final_state.get("earnings_history"),
        "next_earnings_date":   final_state.get("next_earnings_date"),
        "days_to_earnings":     final_state.get("days_to_earnings"),
        "earnings_warning":     final_state.get("earnings_warning"),
        # Price Recommendation
        "entry_low":      final_state.get("entry_low",      0.0),
        "entry_high":     final_state.get("entry_high",     0.0),
        "stop_loss":      final_state.get("stop_loss",      0.0),
        "target_1":       final_state.get("target_1",       0.0),
        "target_2":       final_state.get("target_2",       0.0),
        "risk_reward":    final_state.get("risk_reward",    0.0),
        "price_rec_note": final_state.get("price_rec_note", ""),
        # Portfolio context (echoed back so UI can display it)
        "owns_stock":  final_state.get("owns_stock", False),
        "buy_price":   final_state.get("buy_price",  0.0),
        "shares_held": final_state.get("shares_held", 0.0),
        # Agent 9: Synthesis
        "recommendation":       final_state.get("recommendation"),
        "risk_level":           final_state.get("risk_level"),
        "reasoning":            final_state.get("reasoning"),
        "key_drivers":          final_state.get("key_drivers"),
        "tree_path":            final_state.get("tree_path"),
        "final_report":         final_state.get("final_report"),
    }

    print("✅ Analysis Complete!")
    print("=" * 50)
    return output
