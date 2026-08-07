"""
agents/base_agent.py

Defines the shared contract and shared behavior for all agents.

WHY THIS EXISTS:
GeneratorAgent and ReviewerAgent both need to (1) hold an LLMService,
and (2) load a role-specific system prompt from a text file. Instead
of duplicating that logic in every agent, we put it here once.

BaseAgent does NOT know about other agents, and does NOT talk to
Ollama directly — it only talks to LLMService, which enforces the
architecture rule that Ollama access is centralized in one place.
"""

from abc import ABC, abstractmethod
from llm import LLMService


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Using Python's ABC (Abstract Base Class) here enforces, at the
    language level, that every subclass MUST implement `run()`.
    If a subclass forgets to implement it, Python raises a
    TypeError at instantiation time — catching that mistake early,
    instead of failing silently later when the Orchestrator calls
    a method that doesn't exist.
    """

    def __init__(self, llm_service: LLMService, prompt_path: str) -> None:
        """
        Args:
            llm_service: A shared LLMService instance, injected from
                the Orchestrator. BaseAgent does not create its own —
                this keeps LLM configuration centralized and testable.
            prompt_path: File path to this agent's system prompt
                (e.g. "prompts/generator.txt").
        """
        self.llm_service = llm_service
        self.system_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: str) -> str:
        """
        Load a system prompt from a text file.

        Why a separate method (not inlined in __init__): keeping this
        as its own method means subclasses could override HOW a prompt
        is loaded (e.g. from a database instead of a file) later,
        without changing the constructor's logic.
        """
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @abstractmethod
    def run(self, input_text: str) -> str:
        """
        Execute this agent's task on the given input and return output.

        Every subclass MUST implement this. This is the one method
        the Orchestrator calls — it doesn't need to know whether it's
        talking to a GeneratorAgent or ReviewerAgent, only that
        `.run(text)` exists and returns a string. This is what makes
        the Orchestrator's coordination logic simple and agent-agnostic.
        """
        raise NotImplementedError