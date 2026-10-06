import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""
test_reviewer.py

Manual smoke test for ReviewerAgent — run from PROJECT ROOT only.
Simulates the Orchestrator's job: takes GeneratorAgent's output
and feeds it into ReviewerAgent.
"""

from llm import LLMService
from Agents.generator import GeneratorAgent
from Agents.reviewer import ReviewerAgent


def main() -> None:
    llm_service = LLMService()

    generator = GeneratorAgent(
        llm_service=llm_service,
        prompt_path="prompts/generator.txt",
    )
    reviewer = ReviewerAgent(
        llm_service=llm_service,
        prompt_path="prompts/reviewer.txt",
    )

    user_input = "Explain what a REST API is."
    print(f"User input: {user_input!r}\n")

    draft = generator.run(user_input)
    print(f"Generator output:\n{draft}\n")

    feedback = reviewer.run(draft)
    print(f"Reviewer feedback:\n{feedback}")


if __name__ == "__main__":
    main()