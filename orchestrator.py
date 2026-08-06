"""
orchestrator.py

Coordinates the full Solver <-> Reviewer iterative loop, per the
Solver-Reviewer Iterative Framework: generate, review, and if not
approved, revise and review again, up to MAX_ITERATIONS times.

WHY THE LOOP LIVES HERE:
Neither agent should know the other exists, and neither should know
about "iteration limits" or "approval logic" — that's coordination
logic, which is exactly Orchestrator's job.
"""

from config import MAX_ITERATIONS
from llm import LLMService
from Agents.generator import GeneratorAgent
from Agents.reviewer import ReviewerAgent


class Orchestrator:
    """
    Runs the iterative Solver -> Reviewer refinement workflow.
    """

    def __init__(self) -> None:
        llm_service = LLMService()

        self.generator = GeneratorAgent(
            llm_service=llm_service,
            prompt_path="prompts/generator.txt",
        )
        self.reviewer = ReviewerAgent(
            llm_service=llm_service,
            prompt_path="prompts/reviewer.txt",
        )

    def run(self, user_input: str) -> dict:
        """
        Execute the full iterative refinement workflow.

        Args:
            user_input: The user's original request.

        Returns:
            {
                "final_answer": str,      # what the user should see
                "approved": bool,         # did Reviewer approve it?
                "iterations_used": int,   # how many rounds happened
                "history": list[dict],    # full internal trace, for debugging/logging only — NOT for end users
            }
        """
        draft = self.generator.run(user_input)
        history = []
        decision = "NEEDS_REVISION"
        feedback = ""

        for iteration in range(1, MAX_ITERATIONS + 1):
            raw_review = self.reviewer.run(draft)
            parsed = ReviewerAgent.parse_decision(raw_review)
            decision = parsed["decision"]
            feedback = parsed["feedback"]

            history.append({
                "iteration": iteration,
                "draft": draft,
                "raw_review": raw_review,
                "decision": decision,
            })

            # Stopping condition 1: Reviewer approved it.
            if decision == "APPROVED":
                break

            # Stopping condition 2: we've used our last allowed round —
            # don't bother revising again, just exit with best effort.
            if iteration == MAX_ITERATIONS:
                break

            # Otherwise: revise and loop again.
            draft = self.generator.run(
                user_input,
                previous_draft=draft,
                feedback=feedback,
            )

        return {
            "final_answer": draft,
            "approved": decision == "APPROVED",
            "iterations_used": len(history),
            "history": history,
        }