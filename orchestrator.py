"""
orchestrator.py

Coordinates the full Solver <-> Reviewer iterative loop: generate, review,
and if not approved, revise and review again, up to MAX_ITERATIONS times.

WHY THE LOOP LIVES HERE:
Neither agent should know the other exists, and neither should know about
"iteration limits" or "approval logic" -- that is coordination logic, which
is exactly the Orchestrator's job.

Improvements over the first version:
- Solver and Reviewer use separate LLMService instances (different provider
  order => ideally different model families => real cross-checking).
- Returns the BEST-scoring draft if the limit is reached, not just the last.
- Stops early if a revision doesn't change anything (stagnation).
- If the Reviewer's providers are all down, returns the draft with a clear
  flag instead of crashing the whole request.
- Emits progress events so the UI can show live status.
- GROUNDING: a Planner decides if the question needs fresh facts. If so, web
  sources are retrieved (Tavily) and given to BOTH the Solver (to cite) and
  the Reviewer (to verify every specific claim). If search is unavailable the
  Solver is told not to state unverifiable specifics as fact, and the result
  is flagged "unverified" in the UI.
"""

from datetime import date
from pathlib import Path
from typing import Callable, Optional

import config
from Agents.generator import GeneratorAgent
from Agents.planner import PlannerAgent
from Agents.reviewer import ReviewerAgent
from llm import LLMError, LLMService
from search import SearchError, WebSearch, format_sources

_PROMPTS = Path(__file__).resolve().parent / "prompts"

EventCallback = Callable[[str, dict], None]


class Orchestrator:
    """Runs the iterative Solver -> Reviewer refinement workflow."""

    def __init__(
        self,
        generator_llm: Optional[LLMService] = None,
        reviewer_llm: Optional[LLMService] = None,
        max_iterations: Optional[int] = None,
        planner_llm: Optional[LLMService] = None,
        search: Optional[WebSearch] = None,
        today: Optional[str] = None,
    ) -> None:
        self.generator_llm = generator_llm or LLMService(
            chain=config.GENERATOR_PROVIDERS,
            temperature=config.GENERATOR_TEMPERATURE,
        )
        self.reviewer_llm = reviewer_llm or LLMService(
            chain=config.REVIEWER_PROVIDERS,
            temperature=config.REVIEWER_TEMPERATURE,
        )
        # The planner is a cheap, deterministic routing call; reuse the reviewer
        # provider order (Groq first by default) to spare the Solver's quota.
        self.planner_llm = planner_llm or LLMService(
            chain=config.REVIEWER_PROVIDERS, temperature=0.0, max_tokens=1024
        )
        self.search = search or WebSearch()
        self._today = today
        self.max_iterations = max_iterations or config.MAX_ITERATIONS

        self.generator = GeneratorAgent(
            llm_service=self.generator_llm,
            prompt_path=str(_PROMPTS / "generator.txt"),
        )
        self.reviewer = ReviewerAgent(
            llm_service=self.reviewer_llm,
            prompt_path=str(_PROMPTS / "reviewer.txt"),
        )
        self.planner = PlannerAgent(
            llm_service=self.planner_llm,
            prompt_path=str(_PROMPTS / "planner.txt"),
        )

    @staticmethod
    def _meta(llm: LLMService) -> dict:
        r = llm.last_response
        if r is None:
            return {}
        return {"provider": r.provider, "model": r.model, "latency_s": r.latency_s}

    def _plan(self, user_input: str, today: str) -> dict:
        """Ask the Planner; never raises (falls back to a keyword heuristic)."""
        try:
            raw = self.planner.run(user_input, today)
            return PlannerAgent.parse_plan(raw, user_input, config.MAX_SEARCH_QUERIES)
        except LLMError:
            return PlannerAgent.heuristic_plan(user_input)

    @staticmethod
    def _build_context(today: str, sources: list, mode: str) -> str:
        """Text block shared by Solver and Reviewer: date + evidence (or a caution)."""
        parts = [f"Today's date: {today}."]
        if mode == "grounded":
            parts.append("Web sources retrieved just now (cite as [1], [2], ...):\n" + format_sources(sources))
        elif mode == "unverified":
            parts.append(
                "NOTE: This question depends on current facts, but live web search was unavailable "
                "or returned nothing. No sources are available."
            )
        return "\n\n".join(parts)

    def run(self, user_input: str, on_event: Optional[EventCallback] = None) -> dict:
        """
        Execute the full iterative refinement workflow.

        Returns:
            {
                "final_answer": str,      # what the user should see
                "approved": bool,         # did the Reviewer approve it?
                "iterations_used": int,   # review rounds performed
                "best_score": int | None, # Reviewer's score for final_answer
                "review_unavailable": bool,
                "mode": str,              # grounded / unverified / general
                "sources": list[dict],    # web sources used (title, url, snippet)
                "history": list[dict],    # full trace (for the "inspect" panel)
            }
        """
        emit = on_event or (lambda stage, detail: None)
        today = self._today or date.today().isoformat()

        # --- 1. Plan: does this question need fresh facts? -----------------
        emit("planning", {})
        plan = self._plan(user_input, today)

        # --- 2. Retrieve evidence if needed --------------------------------
        sources: list = []
        search_error: Optional[str] = None
        if plan["needs_search"]:
            if self.search.available:
                emit("searching", {"queries": plan["queries"]})
                try:
                    sources = self.search.search_many(plan["queries"])
                except SearchError as exc:
                    search_error = str(exc)
                if not sources and not search_error:
                    search_error = "search returned no results"
            else:
                search_error = "web search not configured (no TAVILY_API_KEY)"

        mode = "grounded" if sources else ("unverified" if plan["needs_search"] else "general")
        context = self._build_context(today, sources, mode)

        # --- 3. Draft ---------------------------------------------------------
        emit("drafting", {"iteration": 1})
        draft = self.generator.run(user_input, context=context)  # LLMError propagates: nothing to show yet
        draft_meta = self._meta(self.generator_llm)

        history: list[dict] = []
        approved = False
        review_unavailable = False
        best = {"draft": draft, "score": -1}

        for iteration in range(1, self.max_iterations + 1):
            emit("reviewing", {"iteration": iteration})
            try:
                raw_review = self.reviewer.run(draft, user_input, context)
            except LLMError as exc:
                review_unavailable = True
                history.append({
                    "iteration": iteration, "draft": draft, "raw_review": f"(review unavailable: {exc})",
                    "decision": "REVIEW_UNAVAILABLE", "score": None, "feedback": "", "issues": [],
                    "solver": draft_meta, "reviewer": {},
                })
                break

            parsed = ReviewerAgent.parse_decision(raw_review)
            decision, feedback, score = parsed["decision"], parsed["feedback"], parsed["score"]

            history.append({
                "iteration": iteration,
                "draft": draft,
                "raw_review": raw_review,
                "decision": decision,
                "score": score,
                "feedback": feedback,
                "issues": parsed["issues"],
                "solver": draft_meta,
                "reviewer": self._meta(self.reviewer_llm),
            })

            # Track the best draft seen so far (unscored drafts rank below scored ones).
            effective = score if score is not None else (10 if decision == "APPROVED" else 0)
            if effective > best["score"]:
                best = {"draft": draft, "score": effective}

            # Stop 1: Reviewer approved.
            if decision == "APPROVED":
                approved = True
                break

            # Stop 2: last allowed round -- exit with best effort.
            if iteration == self.max_iterations:
                break

            # Otherwise revise and loop again.
            emit("revising", {"iteration": iteration + 1, "feedback": feedback})
            try:
                revised = self.generator.run(
                    user_input, previous_draft=draft, feedback=feedback, context=context
                )
            except LLMError:
                break  # keep the best draft we already have
            draft_meta = self._meta(self.generator_llm)

            # Stop 3: stagnation -- the model returned the same text.
            if revised.strip() == draft.strip():
                break
            draft = revised

        final_answer = draft if approved or review_unavailable else best["draft"]
        final_score = next(
            (h["score"] for h in history if h["draft"] == final_answer and h["score"] is not None), None
        )
        emit("done", {"approved": approved})

        return {
            "final_answer": final_answer,
            "approved": approved,
            "iterations_used": len(history),
            "best_score": final_score,
            "review_unavailable": review_unavailable,
            "mode": mode,  # "grounded" | "unverified" | "general"
            "sources": [src.as_dict() for src in sources],
            "queries": plan["queries"],
            "search_error": search_error,
            "history": history,
        }
