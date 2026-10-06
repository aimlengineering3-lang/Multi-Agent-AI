"""
search.py

Web search grounding via the Tavily API (free tier: 1,000 credits/month).

Why this exists: language models do not know today's schedules, prices or
fixtures, and a Reviewer without sources cannot catch invented specifics.
Retrieved sources give the Solver real evidence to cite and give the
Reviewer something to verify the draft against.

Only the standard library is used (urllib), so there is no extra dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import config

TAVILY_URL = "https://api.tavily.com/search"


class SearchError(RuntimeError):
    """Raised when a web search could not be completed."""


@dataclass
class Source:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _default_post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SearchError(f"Tavily HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SearchError(f"Tavily request failed: {exc}") from exc


class WebSearch:
    """Thin Tavily client returning a de-duplicated list of Sources."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        http_post: Callable[[str, dict, dict, float], dict] = _default_post,
    ) -> None:
        self._api_key = api_key
        self.max_results = max_results or config.SEARCH_MAX_RESULTS
        self.timeout = timeout or config.SEARCH_TIMEOUT
        self._http_post = http_post

    @property
    def api_key(self) -> str:
        return self._api_key if self._api_key is not None else config.tavily_key()

    @property
    def available(self) -> bool:
        return config.SEARCH_ENABLED and bool(self.api_key)

    def search(self, query: str) -> list[Source]:
        if not self.available:
            raise SearchError("web search is not configured (set TAVILY_API_KEY)")
        data = self._http_post(
            TAVILY_URL,
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            {"query": query, "max_results": self.max_results, "search_depth": "basic"},
            self.timeout,
        )
        sources = []
        for item in data.get("results", []) or []:
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            if url and content:
                sources.append(Source(str(item.get("title", "")).strip() or url, url, content))
        return sources

    def search_many(self, queries: list[str]) -> list[Source]:
        """Run several queries; merge results, dropping duplicate URLs."""
        seen: set[str] = set()
        merged: list[Source] = []
        errors: list[str] = []
        for query in queries:
            try:
                for src in self.search(query):
                    if src.url not in seen:
                        seen.add(src.url)
                        merged.append(src)
            except SearchError as exc:
                errors.append(str(exc))
        if not merged and errors:
            raise SearchError("; ".join(errors))
        return merged


def format_sources(sources: list[Source], snippet_chars: int = 700) -> str:
    """Render sources as a numbered block the Solver can cite as [1], [2]..."""
    lines = []
    for i, src in enumerate(sources, start=1):
        snippet = src.snippet if len(src.snippet) <= snippet_chars else src.snippet[:snippet_chars].rstrip() + "..."
        lines.append(f"[{i}] {src.title}\nURL: {src.url}\n{snippet}")
    return "\n\n".join(lines)
