"""
test_orchestrator.py

Manual smoke test for Orchestrator — run from PROJECT ROOT only.
"""

from orchestrator import Orchestrator


def main() -> None:
    orchestrator = Orchestrator()

    user_input = "Explain what a REST API is."
    print(f"User input: {user_input!r}\n")

    result = orchestrator.run(user_input)

    print(f"Final answer:\n{result['final_answer']}\n")
    print(f"Approved: {result['approved']}")
    print(f"Iterations used: {result['iterations_used']}")


if __name__ == "__main__":
    main()