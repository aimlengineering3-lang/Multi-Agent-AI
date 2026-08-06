"""
llm.py

The ONLY module allowed to talk directly to Ollama.
Agents depend on LLMService, never on `ollama` directly.
"""

from ollama import Client
from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS

# WHY explicit host with 127.0.0.1 (not "localhost"):
# On this system, "localhost" resolves to IPv6 (::1), but the Ollama
# server only listens on IPv4 (127.0.0.1). Using the default client
# (which targets "localhost") caused every request to fail with
# ConnectionError, even though the server itself was running fine
# (confirmed working via curl and a direct 127.0.0.1 request).
_client = Client(host="http://127.0.0.1:11434")


class LLMService:
    """Wraps all communication with the Ollama LLM backend."""

    def __init__(self, model_name: str = MODEL_NAME, temperature: float = TEMPERATURE) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return its text response."""
        try:
            response = _client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "num_predict": MAX_TOKENS,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e