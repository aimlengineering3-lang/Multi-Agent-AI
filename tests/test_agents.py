from tests.helpers import FakeLLM
from Agents.generator import GeneratorAgent


def test_generator_uses_system_prompt_and_task_separately():
    llm = FakeLLM(["answer"])
    g = GeneratorAgent(llm, "prompts/generator.txt")
    assert g.run("Explain REST") == "answer"
    assert "Solver Agent" in llm.systems[0]
    assert "Explain REST" in llm.prompts[0] and "Solver Agent" not in llm.prompts[0]


def test_generator_revision_prompt_contains_draft_and_feedback():
    llm = FakeLLM(["better"])
    GeneratorAgent(llm, "prompts/generator.txt").run("q", previous_draft="old", feedback="fix X")
    assert "old" in llm.prompts[0] and "fix X" in llm.prompts[0]
