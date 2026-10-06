"""
Agents/planner.py

PlannerAgent: decides whether a question needs fresh web evidence and, if
so, which search queries to run. It never answers the question itself.

If the LLM fails or replies with something unparseable, a conservative
keyword heuristic takes over, so planning can never crash a request.
"""

import json
import re
from typing import Optional

from Agents.base_agent import BaseAgent

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Words/patterns that usually mean "this answer can go stale".
_TIME_SENSITIVE = re.compile(
    r"\b(20[2-9]\d)\b|latest|current(ly)?|today|tonight|tomorrow|upcoming|recent(ly)?|"
    r"news|price|prices|cost|fare|ticket|schedule|fixture|fixtures|group|standings|"
    r"score|results?|weather|visa|rate|exchange|who is the|this (week|month|year)|next (week|month|year)",
    re.IGNORECASE,
)


class PlannerAgent(BaseAgent):
    """Routes a question: search the web first, or answer from knowledge?"""

    def run(self, user_input: str, today: str = "") -> str:
        task = (f"Today's date: {today}\n\n" if today else "") + f"User request:\n{user_input}"
        return self.llm_service.generate(task, system=self.system_prompt)

    @staticmethod
    def parse_plan(raw: str, user_input: str, max_queries: int = 2) -> dict:
        """Parse the planner's JSON; fall back to the heuristic on any problem."""
        match = _JSON_OBJECT.search(raw or "")
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "needs_search" in data:
                    needs = data["needs_search"] is True or str(data["needs_search"]).strip().lower() == "true"
                    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
                    if needs and not queries:
                        queries = [user_input.strip()[:200]]
                    return {
                        "needs_search": needs,
                        "queries": queries[:max_queries] if needs else [],
                        "reason": str(data.get("reason", "")).strip(),
                    }
            except (json.JSONDecodeError, ValueError):
                pass
        return PlannerAgent.heuristic_plan(user_input)

    @staticmethod
    def heuristic_plan(user_input: str) -> dict:
        needs = bool(_TIME_SENSITIVE.search(user_input))
        return {
            "needs_search": needs,
            "queries": [user_input.strip()[:200]] if needs else [],
            "reason": "heuristic: question looks time-sensitive" if needs else "heuristic: looks timeless",
        }
