"""
config.py

Centralized configuration for the Multi-Agent AI system.

All LLM inference happens on FREE remote providers (Groq, Google Gemini,
OpenRouter, Cerebras), so the project needs no GPU and almost no CPU/RAM.
Every provider speaks the OpenAI-compatible chat API, so a single client
library covers all of them.

Secrets come from environment variables (or a local .env file, or
Streamlit secrets when deployed) -- never from source code.
"""

import os
from dataclasses import dataclass
from typing import Optional

try:  # .env support for local development; optional in production
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    key_env: Optional[str]  # None => no API key needed (e.g. local Ollama)
    default_model: str
    model_env: str          # env var that overrides the default model

    @property
    def api_key(self) -> Optional[str]:
        if self.key_env is None:
            return "not-needed"
        value = os.environ.get(self.key_env, "").strip()
        return value or None

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env, "").strip() or self.default_model

    @property
    def configured(self) -> bool:
        return self.api_key is not None


# NOTE: free-tier model IDs change over time. Override any of them with the
# *_MODEL env vars below instead of editing code.
PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        model_env="GROQ_MODEL",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model="gemini-flash-latest",
        model_env="GEMINI_MODEL",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        model_env="OPENROUTER_MODEL",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY",
        default_model="llama-3.3-70b",
        model_env="CEREBRAS_MODEL",
    ),
    # Optional: the original local backend. Only used if you list it
    # explicitly in GENERATOR_PROVIDERS / REVIEWER_PROVIDERS.
    "ollama": ProviderSpec(
        name="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        key_env=None,
        default_model="llama3.2:3b",
        model_env="OLLAMA_MODEL",
    ),
}


def _chain(env_name: str, default: str) -> list[str]:
    raw = os.environ.get(env_name, default)
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


# Provider order per role. The writer and the reviewer prefer DIFFERENT
# model families: a reviewer from another family is much better at
# catching the writer's blind spots than a model grading itself.
# Each chain is also the failover order if a provider is rate-limited.
GENERATOR_PROVIDERS: list[str] = _chain(
    "GENERATOR_PROVIDERS", "gemini,groq,openrouter,cerebras"
)
REVIEWER_PROVIDERS: list[str] = _chain(
    "REVIEWER_PROVIDERS", "groq,gemini,openrouter,cerebras"
)

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
GENERATOR_TEMPERATURE: float = _env_float("GENERATOR_TEMPERATURE", 0.4)
REVIEWER_TEMPERATURE: float = _env_float("REVIEWER_TEMPERATURE", 0.1)  # judge = deterministic
MAX_TOKENS: int = _env_int("MAX_TOKENS", 4096)  # reasoning models spend tokens on thinking too

# ---------------------------------------------------------------------------
# Reliability (free tiers are rate-limited, so we plan for it)
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT: float = _env_float("REQUEST_TIMEOUT", 60.0)
RETRIES_PER_PROVIDER: int = _env_int("RETRIES_PER_PROVIDER", 2)
RETRY_BACKOFF_SECONDS: float = _env_float("RETRY_BACKOFF_SECONDS", 1.5)
PROVIDER_COOLDOWN_SECONDS: float = _env_float("PROVIDER_COOLDOWN_SECONDS", 30.0)

# ---------------------------------------------------------------------------
# Solver <-> Reviewer loop
# ---------------------------------------------------------------------------
# Was sys.maxsize ("unlimited"). On free tiers an unbounded loop can burn the
# daily quota on one question; 3 rounds gives nearly all of the quality gain.
MAX_ITERATIONS: int = _env_int("MAX_ITERATIONS", 3)

# The Reviewer must both say APPROVED and give at least this score (1-10).
MIN_APPROVAL_SCORE: int = _env_int("MIN_APPROVAL_SCORE", 8)

# ---------------------------------------------------------------------------
# Web search grounding (Tavily, free: 1,000 credits/month, no credit card)
# ---------------------------------------------------------------------------
# Without a key the agent still works, but for time-sensitive questions it is
# instructed NOT to state specific dates/prices/fixtures as fact.
SEARCH_ENABLED: bool = os.environ.get("SEARCH_ENABLED", "true").strip().lower() not in ("0", "false", "no")
SEARCH_MAX_RESULTS: int = _env_int("SEARCH_MAX_RESULTS", 5)
SEARCH_TIMEOUT: float = _env_float("SEARCH_TIMEOUT", 20.0)
MAX_SEARCH_QUERIES: int = _env_int("MAX_SEARCH_QUERIES", 2)


def tavily_key() -> str:
    return os.environ.get("TAVILY_API_KEY", "").strip()


# Backwards-compatible aliases used by older code/UI.
TEMPERATURE: float = GENERATOR_TEMPERATURE


def configured_providers() -> list[str]:
    """Names of providers that currently have credentials available."""
    return [name for name, spec in PROVIDERS.items() if spec.configured and name != "ollama"]
