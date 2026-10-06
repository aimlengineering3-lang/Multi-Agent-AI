"""
Agents/generator.py

GeneratorAgent (a.k.a. "Solver"): produces the initial solution, and
revises it when given prior Reviewer feedback.
"""

from typing import Optional

from Agents.base_agent import BaseAgent


class GeneratorAgent(BaseAgent):
    """Generates a first-pass response, and can revise a previous one."""

    def run(
        self,
        user_input: str,
        previous_draft: Optional[str] = None,
        feedback: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Produce a solution -- either fresh, or a revision.

        previous_draft and feedback are paired: a fresh call has neither,
        a revision call has both. The role instructions go in the proper
        `system` message; the task goes in the `user` message. Chat models
        follow instructions much better this way than when everything is
        concatenated into one user prompt.
        """
        context_block = f"CONTEXT:\n{context}\n\n" if context else ""
        if previous_draft and feedback:
            task = (
                f"{context_block}"
                f"Original request:\n{user_input}\n\n"
                f"Your previous answer:\n{previous_draft}\n\n"
                f"Reviewer feedback to address:\n{feedback}\n\n"
                "Write the improved answer now."
            )
        else:
            task = f"{context_block}User request:\n{user_input}"

        return self.llm_service.generate(task, system=self.system_prompt)
