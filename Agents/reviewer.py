"""
Agents/reviewer.py

ReviewerAgent: evaluates a draft against the ORIGINAL request and returns
a structured decision (APPROVED / NEEDS_REVISION), a 1-10 score, a list of
issues and feedback, so the Orchestrator can decide programmatically whether
to loop again.
"""

import json
import re
from typing import Optional

from Agents.base_agent import BaseAgent
from config import MIN_APPROVAL_SCORE

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ReviewerAgent(BaseAgent):
    """Reviews a draft and returns raw text plus a parsed decision."""

    def run(self, draft: str, user_input: str = "", context: Optional[str] = None) -> str:
        """
        Send the draft (and the original request) to the LLM; return RAW text.

        The original request is included because "relevance and completeness"
        cannot be judged from the draft alone (the previous version of this
        agent only saw the draft).

        Kept separate from parsing so run() still matches BaseAgent's
        "string in, string out" contract.
        """
        parts = []
        if context:
            parts.append(f"CONTEXT:\n{context}")
        if user_input:
            parts.append(f"Original user request:\n{user_input}")
        parts.append(f"Draft answer to review:\n{draft}")
        task = "\n\n".join(parts)
        return self.llm_service.generate(task, system=self.system_prompt)

    @staticmethod
    def parse_decision(raw_response: str) -> dict:
        """
        Parse the Reviewer's reply into a usable dict.

        Accepts the preferred JSON format and falls back to the legacy
        "DECISION: ... / FEEDBACK: ..." line format, because even strong
        models occasionally ignore formatting instructions.

        Safety rule: any ambiguity means NEEDS_REVISION. We never treat an
        unparseable reply as an approval. An APPROVED decision is also
        downgraded if the score is present and below MIN_APPROVAL_SCORE, so
        a reviewer cannot say "approved" while listing serious problems.

        Returns:
            {"decision": "APPROVED" | "NEEDS_REVISION",
             "feedback": str, "score": Optional[int], "issues": list[str]}
        """
        parsed = ReviewerAgent._parse_json(raw_response)
        if parsed is None:
            parsed = ReviewerAgent._parse_legacy(raw_response)

        score: Optional[int] = parsed["score"]
        if (
            parsed["decision"] == "APPROVED"
            and score is not None
            and score < MIN_APPROVAL_SCORE
        ):
            parsed["decision"] = "NEEDS_REVISION"
        return parsed

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        match = _JSON_OBJECT.search(raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or "decision" not in data:
            return None

        decision = (
            "APPROVED"
            if str(data.get("decision", "")).strip().upper() == "APPROVED"
            else "NEEDS_REVISION"
        )
        try:
            score = int(data["score"]) if data.get("score") is not None else None
        except (TypeError, ValueError):
            score = None
        if score is not None:
            score = max(1, min(10, score))

        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        issues = [str(i).strip() for i in issues if str(i).strip()]

        feedback = str(data.get("feedback", "")).strip()
        if issues and decision == "NEEDS_REVISION":
            feedback = (feedback + "\nSpecific issues:\n" + "\n".join(f"- {i}" for i in issues)).strip()
        return {"decision": decision, "feedback": feedback or raw.strip(), "score": score, "issues": issues}

    @staticmethod
    def _parse_legacy(raw: str) -> dict:
        decision = "NEEDS_REVISION"
        feedback = raw.strip()
        for line in raw.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("DECISION:"):
                value = stripped.split(":", 1)[1].strip().upper()
                # NEEDS_REVISION wins if both words somehow appear.
                if "APPROVED" in value and "NEEDS" not in value:
                    decision = "APPROVED"
            elif upper.startswith("FEEDBACK:"):
                feedback = stripped.split(":", 1)[1].strip()
        return {"decision": decision, "feedback": feedback, "score": None, "issues": []}
