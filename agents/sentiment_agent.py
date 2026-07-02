import os
from datetime import datetime, timedelta

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from core.llm import get_llm
from core.state import FinancialState

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

# Headlines cap: 15 for normal days, 30 when a major event is detected
CAP_NORMAL = 15
CAP_MAJOR_EVENT = 30

# Keywords that signal a price-moving event
MAJOR_EVENT_KEYWORDS = [
    "earnings", "beat", "miss", "revenue", "guidance",
    "sec", "fraud", "lawsuit", "investigation",
    "merger", "acquisition", "buyout", "takeover",
    "bankruptcy", "recall", "fda", "layoff", "restructuring",
]

_llm = get_llm()


# --- Structured output schema ---
class SentimentReport(BaseModel):
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (very bearish) to +1.0 (very bullish)"
    )
    sentiment_label: str = Field(
        description="One of: 'Bullish', 'Neutral', or 'Bearish'"
    )
    sentiment_summary: str = Field(
        description="2-3 sentence summary of the dominant news themes and their likely market impact"
    )


_SENTIMENT_PROMPT = """You are a financial news analyst. Analyse the following headlines for {ticker}
and return a structured sentiment assessment.

### HEADLINES (last 7 days):
{headlines}

### INSTRUCTIONS:
- Score sentiment from -1.0 (extremely bearish) to +1.0 (extremely bullish), 0.0 = neutral
- Weight recent headlines more heavily than older ones
- Focus on price-moving events: earnings, guidance, legal issues, M&A, macro impacts
- Ignore generic industry news with no direct company impact

### OUTPUT FORMAT:
Return a single valid JSON object:
{{
    "sentiment_score": <float between -1.0 and 1.0>,
    "sentiment_label": "Bullish | Neutral | Bearish",
    "sentiment_summary": "<2-3 sentences on dominant themes and market impact>"
}}
"""


def _detect_major_event(headlines: list[str]) -> bool:
    """Returns True if any headline contains a high-signal keyword."""
    combined = " ".join(headlines).lower()
    return any(keyword in combined for keyword in MAJOR_EVENT_KEYWORDS)


def sentiment_agent_node(state: FinancialState) -> dict:
    """
    Agent 4 — Sentiment Analysis
    Fetches the last 7 days of company news from Finnhub, applies keyword-based
    event detection to set the headline cap (15 normal / 30 major event),
    then passes headlines to Gemini for structured sentiment scoring.
    """
    ticker = state["ticker"]
    print(f"--- [Agent 4] Fetching news sentiment for {ticker}... ---")

    if not FINNHUB_API_KEY:
        print("[Agent 4] Warning: FINNHUB_API_KEY not set. Skipping sentiment.")
        return {
            "news_headlines":   [],
            "sentiment_score":  0.0,
            "sentiment_label":  "Neutral",
            "sentiment_summary": "Sentiment unavailable — no Finnhub API key configured.",
        }

    # --- 1. Fetch headlines from Finnhub ---
    to_date   = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        response = requests.get(
            FINNHUB_NEWS_URL,
            params={"symbol": ticker, "from": from_date, "to": to_date, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        articles = response.json()
    except Exception as e:
        print(f"[Agent 4] Finnhub fetch error: {e}")
        return {
            "news_headlines":   [],
            "sentiment_score":  0.0,
            "sentiment_label":  "Neutral",
            "sentiment_summary": f"Sentiment unavailable — news fetch failed: {e}",
        }

    all_headlines = [a["headline"] for a in articles if a.get("headline")]

    if not all_headlines:
        print(f"[Agent 4] No headlines found for {ticker}.")
        return {
            "news_headlines":   [],
            "sentiment_score":  0.0,
            "sentiment_label":  "Neutral",
            "sentiment_summary": "No recent news found for this ticker.",
        }

    # --- 2. Keyword-based event detection → dynamic cap ---
    major_event = _detect_major_event(all_headlines)
    cap = CAP_MAJOR_EVENT if major_event else CAP_NORMAL
    selected_headlines = all_headlines[:cap]

    print(
        f"--- [Agent 4] {len(all_headlines)} headlines found. "
        f"Major event: {major_event} → using top {len(selected_headlines)} ---"
    )

    # --- 3. Gemini sentiment scoring ---
    parser = PydanticOutputParser(pydantic_object=SentimentReport)
    prompt = ChatPromptTemplate.from_template(_SENTIMENT_PROMPT)
    messages = prompt.format_messages(
        ticker=ticker,
        headlines="\n".join(f"- {h}" for h in selected_headlines),
        format_instructions=parser.get_format_instructions(),
    )

    try:
        llm_response = _llm.invoke(messages)
        report = parser.parse(llm_response.content)

        print(
            f"--- [Agent 4] Sentiment: {report.sentiment_label} "
            f"(score: {report.sentiment_score:.2f}) ---"
        )

        return {
            "news_headlines":   selected_headlines,
            "sentiment_score":  round(report.sentiment_score, 2),
            "sentiment_label":  report.sentiment_label,
            "sentiment_summary": report.sentiment_summary,
        }

    except Exception as e:
        print(f"[Agent 4] Gemini sentiment error: {e}")
        return {
            "news_headlines":   selected_headlines,
            "sentiment_score":  0.0,
            "sentiment_label":  "Neutral",
            "sentiment_summary": f"Sentiment scoring failed: {e}",
        }
