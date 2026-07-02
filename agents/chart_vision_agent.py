import base64
import os
import tempfile

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from core.llm import get_llm
from core.state import FinancialState

_llm = get_llm()

# Chart periods
BIGPICTURE_DAYS = 252   # ~12 months of trading days (also the Fibonacci window)
CLOSEUP_DAYS    = 30    # ~6 weeks of trading days

# MA colors — explicitly referenced in the Gemini prompt
MA_COLORS = {"MA9": "orange", "MA45": "blue", "MA180": "gray"}

# Fibonacci level colors — dashed, distinct from MA solid lines
FIB_COLORS = {
    "0%":    "#9e9e9e",   # gray   (swing low)
    "23.6%": "#ab47bc",   # purple
    "38.2%": "#ffa726",   # amber
    "50%":   "#ef5350",   # red
    "61.8%": "#42a5f5",   # light blue (golden ratio — most important)
    "100%":  "#9e9e9e",   # gray   (swing high)
}

# Candle market colors
_MARKET_COLORS = mpf.make_marketcolors(
    up="green", down="red",
    wick={"up": "green", "down": "red"},
    edge={"up": "green", "down": "red"},
)
_CHART_STYLE = mpf.make_mpf_style(
    marketcolors=_MARKET_COLORS,
    gridstyle="--",
    gridcolor="#e0e0e0",
    facecolor="white",
)


# --- Structured output schema ---
class ChartReport(BaseModel):
    chart_bias: str = Field(
        description="Overall technical bias from the chart: 'Bullish', 'Bearish', or 'Neutral'"
    )
    chart_patterns_detected: list[str] = Field(
        description="List of specific patterns or signals observed (e.g. 'Convergence forming', '9MA crossing 45MA', 'Inside Day', 'Double Top', 'Price at 61.8% Fib support')"
    )
    chart_analysis: str = Field(
        description="3-5 sentence narrative covering big-picture trend, MA structure, Fibonacci levels, and recent candlestick patterns"
    )
    confidence: str = Field(
        description="Confidence in the visual analysis: 'Low', 'Medium', or 'High'"
    )


_CHART_PROMPT = """You are a technical analyst reviewing two candlestick charts for {ticker}.

CHART CONVENTIONS (apply to both charts):
- Green candles = price closed HIGHER than it opened (bullish day)
- Red candles = price closed LOWER than it opened (bearish day)
- Orange line  = 9-day moving average (most reactive, shortest-term)
- Blue line    = 45-day moving average (intermediate trend)
- Gray line    = 180-day moving average (slowest, primary trend)

CHART 1 ONLY — Fibonacci Retracement Levels (dashed horizontal lines):
The Fibonacci levels are drawn on Chart 1 from the 12-month swing low to swing high.
- Gray dashed    = 0% (12-month swing LOW — floor of the retracement range)
- Purple dashed  = 23.6% retracement level
- Amber dashed   = 38.2% retracement level
- Red dashed     = 50% retracement level
- Light blue dashed = 61.8% retracement level (golden ratio — strongest support/resistance)
- Gray dashed    = 100% (12-month swing HIGH — ceiling of the retracement range)

Fibonacci context passed as numbers: {fib_context}
Use these numbers to anchor your visual observations to exact price levels.

ADX Trend Regime (from numerical calculation, for context):
- ADX: {adx_14} — Regime: {adx_regime}
- +DI: {plus_di} / -DI: {minus_di}
(ADX > 25 = trending market where MA signals are reliable; ADX < 20 = ranging market where MA crossovers are less meaningful)

You will see TWO charts:
  CHART 1 = 12-month big picture view (with Fibonacci levels overlaid)
  CHART 2 = 6-week close-up view

Analyse them in this exact order:

STEP 1 — BIG PICTURE (Chart 1 only):
- Is the overall trend bullish (higher highs + higher lows) or bearish (lower highs + lower lows)?
- What is the current ordering of the three MA lines from top to bottom?
- Is each MA rising, flat, or declining?
- Are the MAs spreading apart (trend strengthening) or converging toward each other (potential breakout/reversal)?
- Where is the current price relative to the Fibonacci retracement levels?
  (Above 61.8% = strong recovery; Between 38.2-61.8% = consolidation zone; Below 38.2% = weak)
- Is price bouncing off or being rejected by any Fibonacci level?
- Are there any visible double tops, triple tops, or major horizontal support/resistance levels?

STEP 2 — CLOSE-UP (Chart 2 only):
- Has the 9-day MA recently crossed the 45-day MA? In which direction?
- Is price currently above or below all three MAs?
- Are the MAs tightening into convergence right now?
- Identify any notable candlestick patterns in the last 10 trading days:
  (inside day, outside day, hammer, shooting star, engulfing, doji)
- Is price testing or bouncing off any of the MA lines?

STEP 3 — SYNTHESIS (use both charts + ADX regime):
- Does the ADX regime (trending vs ranging) affect how you weight the MA signals?
  If ADX < 20 (ranging), MA crossovers are less reliable — note this in your analysis.
- What is the overall technical bias: Bullish, Bearish, or Neutral?
- List the 3-5 most significant patterns or signals you observed (include any Fibonacci confluences)
- Write a 3-5 sentence narrative combining both views, Fibonacci position, and ADX regime
- Rate your confidence: Low (patterns unclear), Medium (some clear signals), High (multiple confirming signals)

If you are uncertain about any specific pattern, say so explicitly rather than guessing.

{format_instructions}
"""


def _compute_mas(df):
    """Compute all three MAs on the full dataset to avoid NaN on sliced charts."""
    df = df.copy()
    df["MA9"]   = df["Close"].rolling(9).mean()
    df["MA45"]  = df["Close"].rolling(45).mean()
    df["MA180"] = df["Close"].rolling(180).mean()
    return df


def _compute_fib_levels(df, n_days: int) -> dict[str, float]:
    """
    Computes Fibonacci retracement levels from the swing high and low
    within the last n_days window. Levels anchored to the same window
    Gemini will see in the chart, so they correspond to visible price action.
    """
    window = df.tail(n_days)
    swing_low  = float(window["Low"].min())
    swing_high = float(window["High"].max())
    diff = swing_high - swing_low

    return {
        "0%":    round(swing_low, 2),
        "23.6%": round(swing_high - 0.236 * diff, 2),
        "38.2%": round(swing_high - 0.382 * diff, 2),
        "50%":   round(swing_high - 0.500 * diff, 2),
        "61.8%": round(swing_high - 0.618 * diff, 2),
        "100%":  round(swing_high, 2),
    }


def _generate_chart(df_with_mas, n_days: int, title: str, fib_levels: dict | None = None) -> str:
    """
    Generates a candlestick chart for the last n_days of data.
    MAs are pre-computed on the full dataset so no NaN appears in the slice.
    If fib_levels is provided, overlays dashed Fibonacci retracement lines.
    Returns the path to a saved PNG temp file.
    """
    data = df_with_mas.tail(n_days).copy()
    ohlcv = data[["Open", "High", "Low", "Close", "Volume"]]

    addplots = [
        mpf.make_addplot(data["MA9"],   color=MA_COLORS["MA9"],   width=1.2),
        mpf.make_addplot(data["MA45"],  color=MA_COLORS["MA45"],  width=1.5),
        mpf.make_addplot(data["MA180"], color=MA_COLORS["MA180"], width=2.0),
    ]

    # Fibonacci lines: constant-value series spanning the full window
    if fib_levels:
        for label, price in fib_levels.items():
            fib_series = pd.Series([price] * len(data), index=data.index)
            addplots.append(
                mpf.make_addplot(
                    fib_series,
                    color=FIB_COLORS[label],
                    width=0.8,
                    linestyle="--",
                )
            )

    fig, axes = mpf.plot(
        ohlcv,
        type="candle",
        addplot=addplots,
        style=_CHART_STYLE,
        title=title,
        figsize=(14, 7),
        returnfig=True,
        tight_layout=True,
    )

    # MA legend entries
    legend_handles = [
        mpatches.Patch(color="orange", label="9-day MA"),
        mpatches.Patch(color="blue",   label="45-day MA"),
        mpatches.Patch(color="gray",   label="180-day MA"),
        mpatches.Patch(color="green",  label="Bullish candle"),
        mpatches.Patch(color="red",    label="Bearish candle"),
    ]

    # Fibonacci legend entries (dashed lines shown as patches for simplicity)
    if fib_levels:
        for label, price in fib_levels.items():
            legend_handles.append(
                mpatches.Patch(
                    color=FIB_COLORS[label],
                    label=f"Fib {label} (${price:.2f})",
                    linestyle="--",
                )
            )

    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.85)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return tmp.name


def _encode_image(path: str) -> str:
    """Base64-encode a PNG file for Gemini Vision."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chart_vision_agent_node(state: FinancialState) -> dict:
    """
    Agent 5 — Visual Chart Analysis

    Chart 1 (1-year daily): 9/45/180-day MAs + Fibonacci retracement overlay
    Chart 2 (6-week close-up): 9/45/180-day MAs only (Fibonacci too cluttered at this scale)

    ADX regime is passed as text context alongside both charts so Gemini
    can weight MA signals appropriately (ADX < 20 → ranging → crossovers less reliable).
    """
    ticker = state["ticker"]
    data   = state["historical_data"]

    print(f"--- [Agent 5] Generating charts for {ticker}... ---")

    bigpicture_path = None
    closeup_path    = None

    try:
        df = _compute_mas(data)

        # Fibonacci levels from the 1-year window (same window as Chart 1)
        fib_levels = _compute_fib_levels(df, BIGPICTURE_DAYS)
        fib_context = "  |  ".join(f"{k}: ${v:.2f}" for k, v in fib_levels.items())
        print(f"--- [Agent 5] Fibonacci levels: {fib_context} ---")

        # Chart 1: 1-year daily with Fibonacci overlay
        bigpicture_path = _generate_chart(
            df, BIGPICTURE_DAYS,
            f"{ticker} — 12 Month View (with Fibonacci)",
            fib_levels=fib_levels,
        )

        # Chart 2: 6-week close-up, no Fibonacci (too cluttered at this zoom)
        closeup_path = _generate_chart(
            df, CLOSEUP_DAYS,
            f"{ticker} — 6 Week Close-Up",
        )

        print(f"--- [Agent 5] Charts generated. Sending to Gemini Vision... ---")

        # Pull ADX from state (computed in analysis_agent)
        technicals = state.get("technical_indicators", {})
        adx_14    = technicals.get("adx_14",    "N/A")
        adx_regime = technicals.get("adx_regime", "Unknown")
        plus_di   = technicals.get("plus_di",   "N/A")
        minus_di  = technicals.get("minus_di",  "N/A")

        # Encode both images
        img1_b64 = _encode_image(bigpicture_path)
        img2_b64 = _encode_image(closeup_path)

        # Build the vision prompt with format instructions
        parser = PydanticOutputParser(pydantic_object=ChartReport)
        prompt_text = _CHART_PROMPT.format(
            ticker=ticker,
            fib_context=fib_context,
            adx_14=adx_14,
            adx_regime=adx_regime,
            plus_di=plus_di,
            minus_di=minus_di,
            format_instructions=parser.get_format_instructions(),
        )

        # Single Gemini call with both images + prompt
        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
            {"type": "text", "text": "The image above is CHART 1 (12-month big picture with Fibonacci levels)."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
            {"type": "text", "text": "The image above is CHART 2 (6-week close-up, no Fibonacci). Now provide your structured analysis."},
        ])

        response = _llm.invoke([message])
        report   = parser.parse(response.content)

        print(f"--- [Agent 5] Chart bias: {report.chart_bias} | Confidence: {report.confidence} ---")

        return {
            "chart_image_path":        closeup_path,
            "chart_bias":              report.chart_bias,
            "chart_confidence":        report.confidence,
            "chart_patterns_detected": report.chart_patterns_detected,
            "chart_analysis":          report.chart_analysis,
            "fib_levels":              fib_levels,
        }

    except Exception as e:
        print(f"[Agent 5] Chart vision error: {e}")
        return {
            "chart_image_path":        None,
            "chart_bias":              "Neutral",
            "chart_confidence":        "Low",
            "chart_patterns_detected": [],
            "chart_analysis":          f"Chart analysis unavailable: {e}",
            "fib_levels":              {},
        }

    finally:
        # Clean up temp files (Option B — each agent deletes its own)
        for path in [bigpicture_path, closeup_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
