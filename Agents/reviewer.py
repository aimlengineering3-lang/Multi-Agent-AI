"""
Agents/reviewer.py

ReviewerAgent: evaluates a draft and returns a structured decision
(APPROVED / NEEDS_REVISION) plus feedback, so the Orchestrator can
programmatically decide whether to loop again.
"""

from Agents.base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    """
    Reviews a draft and returns a parsed decision + feedback.
    """

    def run(self, draft: str) -> str:
        """
        Send the draft to the LLM and return its RAW text response.

        Kept separate from parsing (see parse_decision below) so that
        `run()` still matches BaseAgent's simple "string in, string
        out" contract — parsing is a distinct responsibility.
        """
        full_prompt = f"{self.system_prompt}\n\nDraft to review:\n{draft}"
        return self.llm_service.generate(full_prompt)

    @staticmethod
    def parse_decision(raw_response: str) -> dict:
        """
        Parse the Reviewer's structured text into a usable dict.

        WHY THIS IS SEPARATE FROM run():
        run() is the "talk to the LLM" responsibility. Parsing text
        into structured data is a DIFFERENT responsibility (Single
        Responsibility Principle again). Keeping them separate also
        makes parse_decision independently testable — we can feed it
        fake strings without ever calling the LLM.

        WHY THIS IS LENIENT:
        Small local models don't always follow formatting instructions
        perfectly. Defaulting to NEEDS_REVISION on any ambiguity is a
        deliberate safety choice — if we can't confidently detect an
        approval, we should NOT falsely treat it as approved and stop
        the loop early.

        Returns:
            {"decision": "APPROVED" | "NEEDS_REVISION", "feedback": str}
        """
        decision = "NEEDS_REVISION"
        feedback = raw_response.strip()

        for line in raw_response.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("DECISION:"):
                value = stripped.split(":", 1)[1].strip().upper()
                if "APPROVED" in value:
                    decision = "APPROVED"
            elif stripped.upper().startswith("FEEDBACK:"):
                feedback = stripped.split(":", 1)[1].strip()

        return {"decision": decision, "feedback": feedback}