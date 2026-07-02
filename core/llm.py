"""
Shared LLM factory with automatic retry on transient Gemini errors.

All agents import get_llm() from here instead of instantiating
ChatGoogleGenerativeAI directly. This centralises:
  - Model selection (change once, applies everywhere)
  - Retry logic for 429 / 503 responses
  - load_dotenv() ordering (called before any LLM is created)
"""

import re
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Change the model here to affect the entire pipeline at once.
_MODEL = "gemini-2.5-flash"

# Retry settings for transient errors (429 rate-limit, 503 overload).
_MAX_RETRIES   = 3
_BASE_DELAY_S  = 5.0   # minimum wait before first retry
_RETRY_CODES   = {429, 503}


def _extract_retry_delay(error_message: str) -> float | None:
    """
    Pulls the suggested retry delay from the Gemini error message, e.g.
    'Please retry in 22.96s' → 23.0 seconds.
    Returns None if the pattern isn't found.
    """
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(error_message), re.IGNORECASE)
    return float(match.group(1)) if match else None


class _RetryLLM:
    """
    Thin wrapper around ChatGoogleGenerativeAI that retries on 429/503.
    Exposes only .invoke() — the single method all agents use.
    """

    def __init__(self, model: str):
        self._llm = ChatGoogleGenerativeAI(model=model)

    def invoke(self, messages):
        last_error = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._llm.invoke(messages)
            except Exception as e:
                err_str = str(e)
                # Check for a retryable HTTP status code in the error text
                is_retryable = any(str(code) in err_str for code in _RETRY_CODES)
                if not is_retryable or attempt == _MAX_RETRIES:
                    raise

                suggested = _extract_retry_delay(err_str)
                wait = suggested if suggested else _BASE_DELAY_S * attempt
                print(
                    f"[LLM] Transient error (attempt {attempt}/{_MAX_RETRIES}). "
                    f"Retrying in {wait:.0f}s... [{e}]"
                )
                time.sleep(wait)
                last_error = e

        raise last_error  # unreachable, but satisfies type checkers


def get_llm() -> _RetryLLM:
    """Returns a retry-enabled LLM instance. Call once at module level."""
    return _RetryLLM(_MODEL)
