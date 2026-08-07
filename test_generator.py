"""
test_generator.py

Manual smoke test for GeneratorAgent — run from PROJECT ROOT only.
This is what correctly exercises BaseAgent + GeneratorAgent + LLMService
together, without ever running base_agent.py directly.
"""

from llm import LLMService
from Agents.generator import GeneratorAgent


def main() -> None:
    llm_service = LLMService()
    generator = GeneratorAgent(
        llm_service=llm_service,
        prompt_path="prompts/generator.txt",
    )

    user_input = "Explain what a REST API is."
    print(f"User input: {user_input!r}")

    output = generator.run(user_input)
    print(f"Generator output:\n{output}")


if __name__ == "__main__":
    main()