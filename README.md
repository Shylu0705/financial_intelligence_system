# FinLens: Multi-Agent Financial Intelligence System

## Problem Statement
Traditional retail trading platforms provide raw data but lack **automated synthesis**. Independent traders, such as my parents, spend hours manually cross-referencing technical indicators and news sentiment. This manual process is slow and prone to information overload. **FinLens** solves this by automating the "analyst" role. Success is defined as reducing a 30-minute research session into a 15-second "peer-reviewed" report, providing traders with structured, evidence-based recommendations.

## Solution Overview
FinLens is a decoupled AI system that utilizes a multi-agent workflow to evaluate market assets. It consists of a **FastAPI backend** managing agentic logic and a **Streamlit frontend** for user interaction.
* **AI's Core Role:** AI is central to the functionality, performing non-linear data interpretation that non-AI systems cannot achieve. 
* **The AI Advantage:** Unlike a static dashboard, the AI component cross-references contradictory signals between agents (e.g., bullish technicals vs. bearish sentiment) to provide a reasoned final synthesis.

## AI Integration
* **Agentic Patterns:** Built using **LangGraph** to manage state across a directed acyclic graph (DAG). This allows specialized nodes—Technical, Sentiment, and Portfolio—to work collaboratively.
* **Models & APIs:** Utilizes **Gemini 1.5 Pro** for its high-reasoning capabilities and large context window.
* **Tradeoffs:** I prioritized **accuracy and reliability** over latency by implementing **Pydantic** for structured tool use. While schema validation adds slight overhead, it ensures the LLM produces production-ready data rather than hallucinations.

## Architecture / Design Decisions
* **Decoupled Stack:** The system is split into an **Agent API (FastAPI)** and a **UI (Streamlit)** to allow for independent scaling and modularity.
* **State Management:** I used a centralized `AgentState` in LangGraph to ensure data consistency as the "Lead Analyst" coordinates between nodes.
* **Assumption:** The design assumes real-time internet access for the `yfinance` tool to provide "ground truth" data to the agents.

## What did AI help you do faster, and where did it get in your way?
* **Acceleration:** I used **Cursor and Claude 3.5 Sonnet** to rapidly prototype the complex LangGraph state transitions and FastAPI endpoints. This shifted my focus from syntax debugging to high-level system design.
* **Limitations:** The AI coding tools occasionally struggled with the specific library edge cases of `yfinance`. I had to manually implement robust error handling for market-closed scenarios where the LLM’s training data was insufficient.

---

## Getting Started / Setup Instructions

### Prerequisites
* Python 3.10+
* A **Google Gemini API Key** (for LLM reasoning)
* Two terminal windows (one for the API, one for the UI)

### Installation & Environment
```bash
# Clone the repository
git clone https://github.com/Shylu0705/financial_intellligence_system.git
cd financial_intellligence_system

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and add your GOOGLE_API_KEY
```

### Execution Flow
To run the system, you must start the backend API **before** the frontend.

**Step 1: Start the FIS Agent API**
In your first terminal, launch the FastAPI server:
```bash
python fis_api.py
```

**Step 2: Start the Streamlit UI**
In your second terminal, launch the frontend:
```bash
streamlit run app.py
```

## Usage
1.  Navigate to the local Streamlit URL (usually `localhost:8501`).
2.  **Market Analysis:** Enter a ticker (e.g., `AAPL`) to trigger the agentic workflow.
3.  **Comparison Tab:** Add multiple tickers to see a side-by-side agentic evaluation.
4.  **Portfolio:** Track your holdings and view AI-generated risk assessments based on current market state.

---

## Testing / Error Handling
* **Failure Modes:** Evaluated how the system handles invalid tickers or empty news feeds. 
* **Graceful Degradation:** If one agent fails (e.g., Sentiment API timeout), the Lead Analyst is programmed to proceed with a "Limited Data" warning rather than crashing the entire pipeline.

## Future Improvements / Stretch Goals
* **Multimodal Vision:** Feeding actual technical chart images into the model to mimic a human trader's visual pattern recognition.
* **Backtesting Integration:** Building a module to test the AI's "Buy/Sell" signals against historical performance data.