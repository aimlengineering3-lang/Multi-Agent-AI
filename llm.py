"""
llm.py

The ONLY module allowed to talk to LLM providers.
Agents depend on LLMService, never on a provider SDK directly.

Design:
- Every supported provider (Groq, Gemini, OpenRouter, Cerebras, optional
  local Ollama) exposes an OpenAI-compatible chat endpoint, so one client
  class handles all of them. No GPU or local model is required.
- A request walks a provider *chain*. If a provider is rate-limited, times
  out, errors, or has no key, we back off briefly and fail over to the next
  one. Free tiers are rate-limited by design, so failover is what makes the
  system feel reliable.
- Providers that return 429 are put on a short cooldown so we stop hammering
  them while they recover.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import config
from config import PROVIDERS, ProviderSpec


class LLMError(RuntimeError):
    """Raised when no provider in the chain could produce a response."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    attempts: list[str] = field(default_factory=list)  # failures before success


# Error classification ------------------------------------------------------
# Done by status code / class name so this module (and its tests) don't need
# to import the provider SDK's exception types.
_PERMANENT_STATUS = {400, 401, 403, 404}   # bad key / bad model id: skip provider
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _is_permanent(exc: Exception) -> bool:
    return _status_code(exc) in _PERMANENT_STATUS


def _is_rate_limit(exc: Exception) -> bool:
    return _status_code(exc) == 429 or "ratelimit" in type(exc).__name__.lower()


def _retry_after(exc: Exception) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean(text: str) -> str:
    """Strip reasoning-model <think> blocks so they never leak into answers."""
    return _THINK_BLOCK.sub("", text).strip()


def _default_client_factory(spec: ProviderSpec) -> Any:
    from openai import OpenAI  # imported lazily: keeps unit tests SDK-free

    return OpenAI(
        base_url=spec.base_url,
        api_key=spec.api_key,
        timeout=config.REQUEST_TIMEOUT,
        max_retries=0,  # we do our own retry + failover
    )


class LLMService:
    """Sends chat requests to remote LLM providers with automatic failover."""

    # Shared across instances so a rate-limited provider cools down for everyone.
    _cooldown_until: dict[str, float] = {}

    def __init__(
        self,
        chain: Optional[list[str]] = None,
        temperature: float = config.GENERATOR_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
        client_factory: Callable[[ProviderSpec], Any] = _default_client_factory,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.chain = chain if chain is not None else list(config.GENERATOR_PROVIDERS)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client_factory = client_factory
        self._sleep = sleep
        self._clients: dict[str, Any] = {}
        self.last_response: Optional[LLMResponse] = None

    # -- public API ---------------------------------------------------------
    def active_providers(self) -> list[ProviderSpec]:
        """Providers in this chain that have credentials, in priority order."""
        specs = []
        for name in self.chain:
            spec = PROVIDERS.get(name)
            if spec is not None and spec.configured:
                specs.append(spec)
        return specs

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Run a chat completion, failing over across providers as needed."""
        specs = self.active_providers()
        if not specs:
            raise LLMError(
                "No LLM provider is configured. Set at least one free API key "
                "(GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or CEREBRAS_API_KEY). "
                "See .env.example."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        failures: list[str] = []
        now = time.monotonic()
        # Providers on cooldown go to the back of the line, not off the list:
        # if everything is cooling down we still try them rather than give up.
        ready = [s for s in specs if self._cooldown_until.get(s.name, 0) <= now]
        cooling = [s for s in specs if s not in ready]

        for spec in ready + cooling:
            response = self._try_provider(spec, messages, failures)
            if response is not None:
                response.attempts = failures
                self.last_response = response
                return response

        raise LLMError("All LLM providers failed: " + " | ".join(failures))

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Convenience wrapper: return just the text."""
        return self.complete(prompt, system).text

    # -- internals ----------------------------------------------------------
    def _client(self, spec: ProviderSpec) -> Any:
        if spec.name not in self._clients:
            self._clients[spec.name] = self._client_factory(spec)
        return self._clients[spec.name]

    def _try_provider(
        self, spec: ProviderSpec, messages: list[dict], failures: list[str]
    ) -> Optional[LLMResponse]:
        attempts = max(1, config.RETRIES_PER_PROVIDER)
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                raw = self._client(spec).chat.completions.create(
                    model=spec.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = _clean(raw.choices[0].message.content or "")
                if not text:
                    raise ValueError("empty response")
                usage = getattr(raw, "usage", None)
                return LLMResponse(
                    text=text,
                    provider=spec.name,
                    model=spec.model,
                    latency_s=round(time.monotonic() - started, 2),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
            except Exception as exc:  # noqa: BLE001 - classify, then decide
                failures.append(f"{spec.name}: {type(exc).__name__}: {str(exc)[:120]}")

                if _is_permanent(exc):
                    return None  # bad key / model id: retrying won't help

                if _is_rate_limit(exc):
                    wait = _retry_after(exc) or config.PROVIDER_COOLDOWN_SECONDS
                    self._cooldown_until[spec.name] = time.monotonic() + wait
                    return None  # fail over immediately instead of waiting

                if attempt < attempts:
                    self._sleep(config.RETRY_BACKOFF_SECONDS * attempt)
        return None
