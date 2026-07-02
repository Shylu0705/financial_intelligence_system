# FinLens: Multi-Agent Financial Intelligence System

## Problem Statement
Traditional retail trading platforms provide raw data but lack **automated synthesis**. Independent traders spend hours manually cross-referencing technical indicators, news sentiment, chart patterns, and macroeconomic context. This manual process is slow and prone to information overload. **FinLens** solves this by automating the analyst role — reducing a 30-minute research session into a structured, evidence-based report with actionable price levels.

## Solution Overview
FinLens is a layered AI system built around a 9-agent LangGraph pipeline. A **FastAPI backend** orchestrates the agentic logic and exposes a REST API; a **Streamlit frontend** provides user interaction by calling that API.

- **AI's Core Role:** Gemini Vision reads candlestick charts with Fibonacci overlays as the primary signal; Gemini synthesises all numerical and visual inputs into a structured recommendation via a 5-step decision tree.
- **The AI Advantage:** Unlike a rule-based screener, the hybrid gate + LLM architecture weighs contextual trade-offs, respects hard non-negotiable rules (earnings proximity, revenue decline), and outputs justified, structured recommendations rather than raw scores.

---

## Architecture

```
financial_intelligence_system/
│
├── core/
│   ├── state.py          ← FinancialState TypedDict (shared contract across all agents)
│   ├── workflow.py       ← LangGraph DAG assembly + run_financial_analysis()
│   ├── llm.py            ← Shared LLM factory with retry logic (429/503 auto-retry)
│   └── backtester.py     ← Deterministic signal backtest engine (no LLM)
│
├── agents/
│   ├── data_agent.py               ← Agent 1: 5yr OHLCV history via yfinance
│   ├── analysis_agent.py           ← Agent 2: RSI-14, MACD, ADX-14, ATR-14, MAs, 52W range
│   ├── risk_agent.py               ← Agent 3: Sharpe ratio, max drawdown, volatility
│   ├── sector_agent.py             ← Agent 4: Sector ETF relative strength (63-day)
│   ├── macro_agent.py              ← Agent 5: Fed rate, CPI YoY, 10Y Treasury via FRED; SPY/QQQ RS
│   ├── sentiment_agent.py          ← Agent 6: Finnhub headlines → Gemini sentiment score
│   ├── chart_vision_agent.py       ← Agent 7: mplfinance charts + Fibonacci → Gemini Vision
│   ├── earnings_agent.py           ← Agent 8: Quarterly financials + next earnings date
│   ├── synthesis_agent.py          ← Agent 9: Hard gates → Gemini 5-step decision tree
│   └── price_recommendation_agent.py ← Post-synthesis: deterministic entry/stop/target levels
│
├── config/
│   ├── gates.py          ← Hard gate switches and thresholds (enable/disable per gate)
│   └── gate_engine.py    ← Gate evaluation engine (pipeline failure, earnings, revenue, etc.)
│
├── api/
│   └── routes.py         ← FastAPI endpoints (/analyze, /backtest, /portfolio CRUD)
│
├── db/
│   ├── database.py       ← SQLAlchemy engine + session dependency
│   ├── models.py         ← Holding ORM model (SQLite)
│   └── crud.py           ← add_stock, get_holding, get_all_holdings, delete_stock
│
├── ui/
│   └── app.py            ← Streamlit frontend — 4 tabs (never imports agents directly)
│
├── main.py               ← Uvicorn entry point
└── test.py               ← CLI runner for quick terminal analysis
```

### Pipeline (Linear DAG — 10 nodes)
```
data → analysis → risk → sector → macro → sentiment → chart_vision → earnings → synthesis → price_rec → END
```
Each agent reads from the shared `FinancialState` TypedDict and writes only its own output fields — no agent can modify another's output.

### Key Design Decisions
| Decision | Rationale |
|---|---|
| Layered folder structure | Enforces one-way dependency: UI → API → Core → Agents → DB |
| Streamlit calls FastAPI (not Python directly) | Single execution path; UI gets the same error handling as any other client |
| `asyncio.to_thread()` in FastAPI | Pipeline is blocking I/O (yfinance + LLM); offloads to thread pool so event loop stays responsive |
| Shared `core/llm.py` factory | One place to change the model; built-in 429/503 retry with Gemini's suggested delay |
| Hybrid gate + Gemini synthesis | Deterministic gates enforce non-negotiable rules; Gemini handles nuanced multi-factor judgment |
| Fibonacci on 1-year chart only | Levels anchored to the same window Gemini sees — spatial alignment matters for vision models |
| ADX passed as text context | Non-directional trend strength is exact arithmetic; vision models shouldn't estimate it |
| `PydanticOutputParser` on all LLM outputs | Guarantees typed schema on every LLM response; prevents hallucinated field names reaching the UI |
| Price rec agent is deterministic | Entry/stop/target are pure arithmetic from Fibonacci + ATR; no LLM needed post-synthesis |
| DB lookup in API layer | Portfolio context resolved before pipeline starts; agents only see plain values in state |

---

## Agent Details

### Agent 2 — Technical Analysis
- **RSI-14:** Exponential-weighted gain/loss ratio
- **MACD:** 12/26/9 EMA crossover
- **ADX-14:** Trend strength with +DI/-DI; regime label (Trending / Weak Trend / Ranging)
- **ATR-14:** Average True Range in dollars — used for stop loss sizing
- **MAs:** 9/45/180-day current dollar values
- **52-week high/low**

### Agent 4 — Sector Relative Strength
Compares the stock's 63-trading-day return against its sector ETF (XLK, XLV, XLF, etc.). Tiers: Strong Outperformer / Outperformer / In-line / Underperformer / Strong Underperformer.

### Agent 5 — Macro Context
Fetches Fed Funds Rate, CPI YoY (computed from FRED CPIAUCSL index), and 10Y Treasury yield via FRED API. Also computes SPY/QQQ 3-month returns and the stock's relative performance vs each.

### Agent 6 — News Sentiment
Fetches up to 7 days of Finnhub headlines. Keyword detection caps at 15 headlines normally and 30 when major events (earnings, SEC, merger, bankruptcy, FDA) are detected. Gemini scores sentiment -1.0 → +1.0 with a label and 2-3 sentence summary.

### Agent 7 — Chart Vision
Generates two mplfinance charts (9/45/180-day MAs, colour-coded candles):
- **Chart 1 (1-year daily):** Fibonacci retracement levels overlaid as dashed lines (0%, 23.6%, 38.2%, 50%, 61.8%, 100%) anchored to the 1-year swing high/low
- **Chart 2 (6-week close-up):** Clean view for candlestick pattern detection

Both charts sent to Gemini Vision in a single call. ADX regime passed as text context so Gemini can weight MA signals appropriately (ADX < 20 → ranging → crossovers less reliable).

### Agent 9 — Synthesis (Hybrid)
**Layer 1 — Hard gates** (deterministic, no LLM):
| Gate | Default | Trigger |
|---|---|---|
| Pipeline failure | ON | Any critical data missing |
| Earnings proximity | ON | Earnings ≤ 7 days away |
| Revenue decline | ON | 3 consecutive quarters of declining sales |
| Extreme drawdown | OFF | Current drawdown ≤ -30% |
| Extreme volatility | OFF | Annualised vol ≥ 80% |
| Negative networth | OFF | Latest networth < 0 |
| Extreme macro | OFF | Fed rate ≥ 6% AND P/E ≥ 40 |

All gate thresholds are configurable in `config/gates.py` without touching agent logic.

**Layer 2 — Gemini 5-step decision tree:**
1. Earnings trust (HIGH / MEDIUM / LOW)
2. Macro headwind/tailwind
3. Chart signal (primary, highest weight)
4. Confirmation check (sector RS + sentiment + technicals)
5. Final synthesis → BUY / SELL / HOLD

When portfolio context is present (user owns the stock), Step 5 reframes as ADD / HOLD existing / EXIT and factors in unrealised P&L.

### Price Recommendation Agent (deterministic)
| Scenario | Output |
|---|---|
| BUY / ADD | Entry zone anchored to nearest Fib support; stop = tighter of 1.5×ATR or next Fib below; T1/T2 = next Fib resistance levels |
| SELL + owns stock | Exit at market — shows P&L from cost basis |
| SELL + no position | Short entry zone, stop above, targets below |
| HOLD + owns stock | Trailing stop suggestion (1.5×ATR or nearest Fib below) |
| HOLD + no position | No levels — wait for signal |

---

## Backtest Engine
Deterministic replay of RSI/MACD/MA/ATR signals — no LLM calls.

**Signal rules:**
- BUY: MACD > Signal AND Close > MA180 AND RSI < 70
- SELL: MACD < Signal AND Close < MA45 AND RSI > 30

**Trade management:** Signals checked weekly (Friday). Stop loss enforced daily via `row["Low"]`. Configurable max holding period and ATR multiplier.

**Metrics:** Total return vs buy-and-hold, win rate, avg win/loss, profit factor, max drawdown, Sharpe ratio, number of trades, avg holding days.

**Warmup:** Fetches 300 days before `start_date` for indicator warmup (MA180 requires 180 prior days) then trims to the requested range — no look-ahead bias.

---

## Getting Started

### Prerequisites
- Python 3.10+
- A **Google Gemini API key** (free tier: 20 req/day for gemini-2.5-flash; pay-as-you-go for unlimited)
- Optionally: **Finnhub API key** (sentiment degrades gracefully without it) and **FRED API key** (macro data)
- Two terminal windows (API + UI run separately)

### Installation
```bash
git clone https://github.com/Shylu0705/financial_intellligence_system.git
cd financial_intellligence_system

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env
```

### Running the System

**Terminal 1 — API server:**
```bash
python main.py
# http://127.0.0.1:8000  |  Swagger docs: http://127.0.0.1:8000/docs
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run ui/app.py
# http://localhost:8501
```

**CLI (no UI):**
```bash
python test.py
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key |
| `FINNHUB_API_KEY` | *(optional)* | News headlines for sentiment agent |
| `FRED_API_KEY` | *(optional)* | Fed rate, CPI, Treasury yield |
| `RISK_FREE_RATE` | `0.04` | Annual risk-free rate for Sharpe ratio |
| `API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL (override for deployment) |

### Switching Models
All three LLM agents share a single factory in `core/llm.py`. To change the model:
```python
# core/llm.py
_MODEL = "gemini-2.5-flash"   # or "gemini-2.0-flash" with billing enabled
```

---

## UI — 4 Tabs

| Tab | Description |
|---|---|
| **Stock Analysis** | Full pipeline — metrics, chart analysis, Fibonacci levels, ADX regime, earnings, decision tree path, price levels |
| **Compare Stocks** | Side-by-side RSI, trend, Sharpe, max drawdown for two tickers |
| **Portfolio** | Add holdings (buy price auto-fetched), view current P&L, delete |
| **Backtest** | Date range picker, equity curve vs buy-and-hold, trade log with colour-coded returns |

Portfolio context is **auto-detected** — if a ticker being analysed is in the Portfolio tab, `owns_stock=True` is set automatically and the synthesis + price levels adjust accordingly. A manual override toggle is also available.

---

## API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/analyze` | `{ticker, owns_stock?, buy_price?, shares_held?}` | Full 9-agent pipeline |
| `POST` | `/backtest` | `{ticker, start_date, end_date, max_hold_days?, atr_mult?}` | Deterministic signal backtest |
| `GET` | `/portfolio` | — | List all holdings |
| `POST` | `/portfolio` | `{ticker, shares}` | Add holding (price auto-fetched) |
| `DELETE` | `/portfolio/{ticker}` | — | Remove holding |

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Error Handling
- Invalid tickers return `404` with a descriptive message
- LLM parse failures fall back to an `ERROR` recommendation state — pipeline does not crash
- 429/503 from Gemini are auto-retried using the API's suggested delay (up to 3 attempts)
- Hard gates short-circuit synthesis on non-negotiable conditions — no LLM call made
- All API errors surface the `detail` field directly in the Streamlit UI
