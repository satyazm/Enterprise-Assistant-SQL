"""
app/llm.py

Shared LLM dispatcher, used by both the RAG pipeline and the SQL agent
(and, from Phase 3 onward, by every LangGraph node). Kept in one place
so provider swaps only happen here.

LLM_PROVIDER supports "gemini" (default, what the deployed app actually
runs on), "openai", and "groq". The "groq" option exists mainly so
eval/run_eval.py can point the *whole* graph — not just its judge — at
Groq when Gemini's free-tier quota is the bottleneck, at the cost of no
longer evaluating the exact model the app is deployed on. See that
module's docstring for the tradeoff.
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


_NON_RETRYABLE_MARKERS = (
    "GenerateRequestsPerDay",  # Gemini: requests/day quota
    "tokens per day",  # Groq: TPD (tokens per day) quota
    "(TPD)",
)


def _is_retryable(error: Exception) -> bool:
    message = str(error)

    # A per-day quota error won't resolve within this process's lifetime —
    # retrying just burns the backoff window sleeping through a cap that
    # won't lift for hours. Per-minute limits and transient 5xx errors are
    # worth retrying; per-day ones aren't, regardless of which provider's
    # wording they show up in.
    if any(marker in message for marker in _NON_RETRYABLE_MARKERS):
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

    elif LLM_PROVIDER == "groq":
        return call_llm_groq(prompt)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def call_llm_groq(prompt: str) -> str:
    """Separate from call_llm() on purpose: this isn't a provider option
    behind LLM_PROVIDER, it's a dedicated path for eval/run_eval.py's
    LLM-as-judge specifically, so grading traffic doesn't compete with the
    system-under-test's own Gemini quota — and, as a side benefit, judging
    from a different model family than the one being judged avoids a model
    being biased toward favoring its own family's outputs.

    GROQ_MODEL defaults to a current Groq-hosted model but is overridable
    in .env, since which models Groq hosts changes over time — check
    https://console.groq.com/docs/models if the default starts 404ing."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    model = ChatGroq(model=model_name, groq_api_key=api_key)
    response = _invoke_with_retry(model, prompt)
    return response.content
