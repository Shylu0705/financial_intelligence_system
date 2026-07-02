from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.gate_engine import evaluate_gates
from core.llm import get_llm
from core.state import FinancialState

_llm = get_llm()


# --- Structured output schema ---
class FinancialReport(BaseModel):
    recommendation: str = Field(
        description="Final trading recommendation: 'BUY', 'SELL', or 'HOLD'"
    )
    reasoning: str = Field(
        description="Concise explanation (under 150 words) justifying the recommendation"
    )
    key_drivers: list[str] = Field(
        description="3-5 specific data points that influenced the decision"
    )
    risk_level: str = Field(
        description="Assessed risk level: 'Low', 'Medium', 'High', or 'Extreme'"
    )
    tree_path: list[str] = Field(
        description="The decision tree steps you followed, in order, as short labels e.g. ['Chart: Bearish', 'Sentiment: confirms', 'Macro: headwind', 'Decision: HOLD']"
    )


_PROMPT_TEMPLATE = """You are a Senior Investment Strategist at a top-tier hedge fund.
All hard gate rules have already been checked and passed. Your job is to make the final
recommendation by working through the decision tree below IN ORDER.

═══════════════════════════════════════════════════════
DATA INPUTS
═══════════════════════════════════════════════════════

1. Fundamental Data:
{fundamentals}

2. Technical Indicators:
{technicals}

3. Risk Metrics:
{risk}

4. Macro & Market Context:
- Fed Funds Rate: {fed_funds_rate}% — {rate_environment}
- CPI YoY: {cpi}% — {inflation_environment}
- 10Y Treasury Yield: {treasury_10y}%
- vs S&P 500 (SPY): {vs_spy}% ({vs_spy_label})
- vs NASDAQ 100 (QQQ): {vs_qqq}% ({vs_qqq_label})

5. Sector Relative Strength (3-month):
- Sector ETF: {sector_etf}
- Stock Return: {stock_3m_return}%
- Sector Return: {sector_3m_return}%
- Relative Strength: {relative_strength}%
- Label: {sector_label}

6. News Sentiment (last 7 days):
- Label: {sentiment_label}
- Score: {sentiment_score} (scale: -1.0 bearish → +1.0 bullish)
- Summary: {sentiment_summary}

7. Visual Chart Analysis (HIGHEST WEIGHT — treat this as your primary signal):
- Chart Bias: {chart_bias}
- Confidence: {chart_confidence}
- Patterns Detected: {chart_patterns_detected}
- Visual Analysis: {chart_analysis}

8. Quarterly Earnings & Financials (most recent first):
{earnings_history}

9. Portfolio Context:
{portfolio_context}

═══════════════════════════════════════════════════════
DECISION TREE — work through each step in order
═══════════════════════════════════════════════════════

STEP 1 — EARNINGS TRUST (foundation of the decision)
Examine the quarterly earnings history. Is revenue and net profit growing,
stable, or declining? A company with consistently growing earnings and
strong networth earns a HIGH trust score. Erratic or mixed results = MEDIUM.
→ Record: trust score (HIGH / MEDIUM / LOW) and why.

STEP 2 — MACRO HEADWIND OR TAILWIND
Do the current interest rate and inflation environment help or hurt this
specific stock? High rates hurt high-P/E growth stocks most. Low rates
help them. Consider the stock's sector and valuation in this context.
→ Record: macro stance (TAILWIND / NEUTRAL / HEADWIND) and why.

STEP 3 — CHART SIGNAL (primary signal — highest weight)
What is the chart bias and how confident is the assessment?
- HIGH confidence Bullish  → strong lean toward BUY
- HIGH confidence Bearish  → strong lean toward SELL/HOLD
- MEDIUM confidence        → treat as a lean, not a decision
- LOW confidence           → treat as a weak signal only
The chart cannot force a decision alone — it must be weighed against
Steps 4 and 5. But it is the heaviest single input.
→ Record: chart signal strength and direction.

STEP 4 — CONFIRMATION CHECK
Do sector RS, sentiment, and technicals confirm or contradict the chart?
Count confirming vs contradicting signals:
- 2+ signals confirm chart → strengthen the chart's recommendation
- 2+ signals contradict    → moderate toward HOLD
- Mixed (1 confirm, 1 contra) → chart signal stands but with lower conviction
→ Record: confirmation count and net direction.

STEP 5 — FINAL SYNTHESIS
Combine all steps. The chart (Step 3) carries the most weight, modulated
by trust (Step 1), macro (Step 2), and confirmation (Step 4).

Guidelines (no portfolio position):
- BUY:  High trust + tailwind/neutral macro + bullish chart + 1+ confirming signal
- HOLD: Any two of: medium/low trust, headwind macro, mixed chart, contradicting signals
- SELL: Low/medium trust + bearish chart (medium+ confidence) + 1+ confirming signal
        OR high trust + bearish chart (HIGH confidence) + 2+ confirming signals

If the user ALREADY OWNS this stock (see Portfolio Context), adjust your framing:
- BUY  → interpret as "ADD to position" — only if strong conviction
- HOLD → interpret as "HOLD existing position" — consider whether to protect gains or average down
- SELL → interpret as "EXIT or REDUCE position"
- Factor in unrealised P&L: if the user is up significantly, a HOLD lean becomes more conservative;
  if down significantly, require stronger conviction before recommending ADD.
- Your reasoning MUST reference the user's cost basis and current P&L if known.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
Output a single valid JSON object — no Markdown, no extra text:
{{
    "recommendation": "BUY | SELL | HOLD",
    "risk_level": "Low | Medium | High | Extreme",
    "reasoning": "Under 150 words covering all 5 steps",
    "key_drivers": ["3-5 specific data points"],
    "tree_path": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ...", "Step 5: ..."]
}}
"""


def _build_gate_report(gate_result) -> dict:
    """Builds a state update when a hard gate fires — no LLM needed."""
    rec = gate_result.forced_rec
    reason = gate_result.reason

    final_report = (
        f"**Recommendation:** {rec}\n"
        f"**Risk Level:** High\n\n"
        f"**Gate Triggered:** {gate_result.gate_name}\n\n"
        f"**Reasoning:**\n{reason}\n\n"
        f"**Key Drivers:**\n- Hard gate override — no further analysis performed."
    )

    return {
        "recommendation": rec,
        "risk_level":     "High",
        "reasoning":      reason,
        "key_drivers":    [f"Gate '{gate_result.gate_name}' triggered: {reason}"],
        "tree_path":      [f"Gate override: {gate_result.gate_name}"],
        "final_report":   final_report,
    }


def synthesis_node(state: FinancialState) -> dict:
    """
    Final synthesis agent — hybrid decision tree.

    Layer 1 (deterministic): Hard gates evaluated first via gate_engine.
                             If any gate fires, returns immediately with
                             forced_rec — no LLM call made.

    Layer 2 (Gemini tree):   If all gates pass, Gemini works through a
                             structured 5-step decision tree with the chart
                             signal as the primary (highest weight) input.
    """
    ticker = state["ticker"]
    print(f"--- [Synthesis] Running decision tree for {ticker}... ---")

    # ── Layer 1: Hard gates ──────────────────────────────────────────────────
    gate_result = evaluate_gates(state)
    if gate_result and gate_result.triggered:
        print(f"--- [Synthesis] Gate override → {gate_result.forced_rec} ---")
        return _build_gate_report(gate_result)

    # ── Layer 2: Gemini decision tree ────────────────────────────────────────
    print("--- [Synthesis] All gates passed — invoking Gemini decision tree... ---")

    # Build portfolio context string for the prompt
    owns_stock  = state.get("owns_stock",  False)
    buy_price   = state.get("buy_price",   0.0)
    shares_held = state.get("shares_held", 0.0)

    if owns_stock and buy_price > 0:
        current_price = state.get("technical_indicators", {}).get("current_price", 0.0)
        pnl_pct = ((current_price - buy_price) / buy_price * 100) if buy_price else 0.0
        pnl_sign = "+" if pnl_pct >= 0 else ""
        portfolio_context = (
            f"User HOLDS this stock.\n"
            f"  Shares held:    {shares_held}\n"
            f"  Cost basis:     ${buy_price:.2f} per share\n"
            f"  Current price:  ${current_price:.2f} per share\n"
            f"  Unrealised P&L: {pnl_sign}{pnl_pct:.1f}%\n"
            f"Interpret BUY as 'Add to position', SELL as 'Exit/Reduce', HOLD as 'Hold existing'."
        )
        print(f"--- [Synthesis] Portfolio context: owns {shares_held} shares @ ${buy_price:.2f} ({pnl_sign}{pnl_pct:.1f}%) ---")
    else:
        portfolio_context = "User does NOT currently hold this stock. Standard entry recommendation applies."

    parser = PydanticOutputParser(pydantic_object=FinancialReport)
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)

    messages = prompt.format_messages(
        ticker=ticker,
        fundamentals=state["fundamental_metrics"],
        technicals=state["technical_indicators"],
        risk=state["risk_metrics"],
        fed_funds_rate=state.get("fed_funds_rate", "N/A"),
        cpi=state.get("cpi", "N/A"),
        treasury_10y=state.get("treasury_10y", "N/A"),
        rate_environment=state.get("rate_environment", "Unknown"),
        inflation_environment=state.get("inflation_environment", "Unknown"),
        vs_spy=state.get("vs_spy", "N/A"),
        vs_qqq=state.get("vs_qqq", "N/A"),
        vs_spy_label=state.get("vs_spy_label", "Unknown"),
        vs_qqq_label=state.get("vs_qqq_label", "Unknown"),
        sector_etf=state.get("sector_etf", "N/A"),
        stock_3m_return=state.get("stock_3m_return", "N/A"),
        sector_3m_return=state.get("sector_3m_return", "N/A"),
        relative_strength=state.get("relative_strength", "N/A"),
        sector_label=state.get("sector_label", "Unknown"),
        sentiment_label=state.get("sentiment_label", "Neutral"),
        sentiment_score=state.get("sentiment_score", 0.0),
        sentiment_summary=state.get("sentiment_summary", "No sentiment data available."),
        chart_bias=state.get("chart_bias", "Neutral"),
        chart_confidence=state.get("chart_confidence", "Unknown"),
        chart_patterns_detected=state.get("chart_patterns_detected", []),
        chart_analysis=state.get("chart_analysis", "No chart analysis available."),
        earnings_history=state.get("earnings_history", []),
        portfolio_context=portfolio_context,
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response   = _llm.invoke(messages)
        report     = parser.parse(response.content)
        report_dict = report.dict()

        tree_path_str = "\n".join(f"- {step}" for step in report_dict.get("tree_path", []))

        report_dict["final_report"] = (
            f"**Recommendation:** {report_dict['recommendation']}\n"
            f"**Risk Level:** {report_dict['risk_level']}\n\n"
            f"**Decision Tree:**\n{tree_path_str}\n\n"
            f"**Reasoning:**\n{report_dict['reasoning']}\n\n"
            f"**Key Drivers:**\n- " + "\n- ".join(report_dict["key_drivers"])
        )

        print(f"--- [Synthesis] {report_dict['recommendation']} | Risk: {report_dict['risk_level']} ---")
        return report_dict

    except Exception as e:
        print(f"[Synthesis] LLM/parse error: {e}")
        return {
            "recommendation": "ERROR",
            "risk_level":     "Unknown",
            "reasoning":      str(e),
            "key_drivers":    [],
            "tree_path":      [],
            "final_report":   "Error generating structured report.",
        }
