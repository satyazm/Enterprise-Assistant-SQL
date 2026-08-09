"""
app/llm.py

Shared LLM dispatcher, used by both the RAG pipeline and the SQL agent
(and, from Phase 3 onward, by every LangGraph node). Kept in one place
so provider swaps only happen here.
"""

import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 8

# Matches the suggested wait time Gemini includes in 429 error bodies,
# e.g. "Please retry in 46.6s" or "retryDelay': '46s'" — prefer the
# provider's own estimate over a blind exponential guess when it's given one.
_RETRY_DELAY_RE = re.compile(r"retry(?:Delay)?['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _is_retryable(error: Exception) -> bool:
    message = str(error)

    # A per-day quota error won't resolve within this process's lifetime —
    # retrying just burns the backoff window for nothing. Per-minute limits
    # and transient 5xx errors are worth retrying; per-day ones aren't.
    if "GenerateRequestsPerDay" in message:
        return False

    return any(marker in message for marker in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500"))


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    match = _RETRY_DELAY_RE.search(str(error))
    if match:
        return float(match.group(1)) + 1  # small buffer past the provider's own estimate
    return BASE_BACKOFF_SECONDS * (2 ** attempt)


def _invoke_with_retry(model, prompt: str):
    """Retries transient provider errors (rate limits, 5xx) with backoff.
    Anything else (bad API key, invalid request) fails immediately —
    retrying those would just waste time on a guaranteed repeat failure."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return model.invoke(prompt)
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(_retry_delay_seconds(e, attempt))
    raise last_error


def call_llm(prompt: str) -> str:
    """Uses LangChain chat model wrappers so this plugs directly into
    LangGraph nodes later without a rewrite."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env")

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
        )
        response = _invoke_with_retry(model, prompt)
        return response.content

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")

        model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
        )
        response = _invoke_with_retry(model, prompt)
        return response.content

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
