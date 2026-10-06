"""Shared fakes so tests run offline with no API keys and no SDK installed."""

import os
from types import SimpleNamespace

# Pretend two providers are configured. Set before config is read by tests.
os.environ.setdefault("GROQ_API_KEY", "test-groq")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini")


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, msg: str = "boom", retry_after=None):
        super().__init__(msg)
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers={"retry-after": str(retry_after)} if retry_after is not None else {},
        )


def make_completion(text: str):
    msg = SimpleNamespace(content=text)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create with a scripted queue."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return make_completion(item)


class FakeLLM:
    """Stand-in for LLMService used by agent/orchestrator tests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.systems = []
        self.last_response = SimpleNamespace(provider="fake", model="fake-1", latency_s=0.0)

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeSearch:
    """Stand-in for WebSearch."""

    def __init__(self, sources=None, available=True, error=None):
        from search import Source

        self._sources = [Source(t, u, c) for t, u, c in (sources or [])]
        self.available = available
        self.error = error
        self.queries = []

    def search_many(self, queries):
        from search import SearchError

        self.queries.append(list(queries))
        if self.error:
            raise SearchError(self.error)
        return list(self._sources)
