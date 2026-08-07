"""
Agents/generator.py

GeneratorAgent (a.k.a. "Solver" per the architecture spec): produces
the initial solution, and revises it when given prior feedback.
"""

from typing import Optional
from Agents.base_agent import BaseAgent


class GeneratorAgent(BaseAgent):
    """
    Generates a first-pass response, and can revise a previous
    response based on Reviewer feedback.
    """

    def run(
        self,
        user_input: str,
        previous_draft: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Produce a solution — either fresh, or a revision.

        Args:
            user_input: The original user request (always needed,
                even during revision, so the Solver never loses
                sight of the original goal).
            previous_draft: The Solver's own last answer, if this
                is a revision round.
            feedback: The Reviewer's feedback on that previous
                draft, if this is a revision round.

        Returns:
            The generated (or revised) solution as a string.

        WHY previous_draft AND feedback are both optional and paired:
        A fresh first-pass call has neither. A revision call has both.
        We never want "feedback with no draft" or vice versa — that
        would be an incomplete/nonsensical prompt to the LLM.
        """
        if previous_draft and feedback:
            full_prompt = (
                f"{self.system_prompt}\n\n"
                f"Original request: {user_input}\n\n"
                f"Your previous answer:\n{previous_draft}\n\n"
                f"Reviewer feedback on that answer:\n{feedback}\n\n"
                f"Revise your answer to fully address this feedback."
            )
        else:
            full_prompt = f"{self.system_prompt}\n\nUser request: {user_input}"

        return self.llm_service.generate(full_prompt)