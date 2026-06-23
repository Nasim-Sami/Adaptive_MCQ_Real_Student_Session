"""
llm_client.py — Thin wrapper over the Groq cloud API (OpenAI-compatible chat
completions) for grounded MCQ generation, explanations, and learning paths.

Why Groq: a free, fast (sub-second to ~1-2s typical) cloud inference API,
replacing the local qwen2.5:3b/Ollama setup this previously used. Embeddings
(PDF-chunk retrieval, nomic-embed-text) stay on Ollama — see retriever.py —
Groq has no embeddings endpoint, so this module only covers generation.

Setup
-----
    export GROQ_API_KEY="gsk_..."     # https://console.groq.com -> API Keys

Features
--------
* JSON-mode generation with automatic JSON extraction from the response.
* Per-call timeout and retry with exponential back-off.
* A cached availability probe (checks the API key + a live ping).

Public API (unchanged from the old Ollama version - callers need no changes)
----------
    from llm_client import generate_json, generate_text, is_available, LLMUnavailable

    ok = is_available()          # True if GROQ_API_KEY is set and Groq is reachable
    result = generate_json(      # raises LLMUnavailable if Groq is unreachable
        prompt="...",
        system="...",
        temperature=0.7,
        max_tokens=512,
    )
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _load_dotenv_local() -> None:
    """Auto-load ``.env.local`` (simple ``export KEY="value"`` lines) from
    this file's directory, if present, without overriding any var already
    set in the real shell environment. Means GROQ_API_KEY doesn't need to be
    manually `source`d every time a new terminal / VS Code window opens -
    just keep .env.local in the project folder (never commit it)."""
    env_path = Path(__file__).resolve().parent / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_local()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
# llama-3.1-8b-instant: a real tutoring session makes several small calls per
# question (explanation, occasional generation, one learning-path call at the
# end) in quick succession - the 8b model's free-tier rate limit is much more
# generous than the 70b model's, so it holds up far better under that burst
# pattern. Set GROQ_MODEL=llama-3.3-70b-versatile for higher quality at the
# cost of a much lower free-tier request rate (expect more 429s / fallbacks).
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_DELAY = 2.0          # seconds, doubled on each retry


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LLMUnavailable(RuntimeError):
    """Raised when Groq / the model cannot be reached after all retries."""


# ---------------------------------------------------------------------------
# Availability probe (cached per process)
# ---------------------------------------------------------------------------
_available: bool | None = None


def _api_key() -> str | None:
    return os.environ.get(GROQ_API_KEY_ENV)


def is_available(force_check: bool = False) -> bool:
    """Return True if GROQ_API_KEY is set and the Groq API is reachable."""
    global _available
    if _available is not None and not force_check:
        return _available

    key = _api_key()
    if not key:
        _available = False
        return _available

    try:
        import requests
        resp = requests.get(
            f"{GROQ_BASE}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        _available = resp.status_code == 200
    except Exception:
        _available = False
    return _available


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Any:
    """
    Try to extract a JSON object or array from arbitrary LLM text.
    Handles markdown code fences and bare JSON.
    """
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, stripped)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    raise ValueError(f"No valid JSON found in LLM output:\n{text[:500]}")


def _chat(messages: list[dict[str, str]], temperature: float, max_tokens: int,
         json_mode: bool) -> str:
    if not is_available():
        raise LLMUnavailable(
            f"Groq is not available (set {GROQ_API_KEY_ENV} and check connectivity)."
        )

    import requests

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }

    delay = RETRY_DELAY
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{GROQ_BASE}/chat/completions",
                json=payload,
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                # rate-limited - respect Retry-After if Groq sends one,
                # otherwise fall back to exponential backoff
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
        except (KeyError, IndexError) as exc:
            # Unexpected response shape - don't retry
            raise ValueError(f"Unexpected Groq response shape: {exc}") from None

    raise LLMUnavailable(f"Groq unreachable after {MAX_RETRIES + 1} attempts: {last_exc}")


def generate_json(
    prompt: str,
    system: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    """
    Call the Groq model and return parsed JSON.

    Parameters
    ----------
    prompt      : The user prompt (already includes context / instructions).
    system      : Optional system message. Must mention "JSON" somewhere (the
                  system+prompt combined) - required by Groq/OpenAI JSON mode.
    temperature : Sampling temperature (0 = deterministic).
    max_tokens  : Maximum tokens in the response.

    Returns
    -------
    Parsed Python object (dict or list).

    Raises
    ------
    LLMUnavailable : If Groq cannot be reached after retries.
    ValueError     : If the response cannot be parsed as JSON (caller should
                     catch and fall back to bank).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    content = _chat(messages, temperature, max_tokens, json_mode=True)
    return _extract_json(content)


def generate_text(
    prompt: str,
    system: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Same as generate_json but returns raw text (no JSON parsing).
    Useful for free-form narrative (learning path, explanations).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    content = _chat(messages, temperature, max_tokens, json_mode=False)
    return content.strip()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Groq available: {is_available(force_check=True)}")
    if is_available():
        result = generate_json(
            prompt='Return {"status": "ok", "model": "groq"}',
            system="You are a helpful assistant. Always respond with valid JSON.",
        )
        print("Response:", result)
    else:
        print(f"Set {GROQ_API_KEY_ENV} to enable (get a free key at https://console.groq.com).")
