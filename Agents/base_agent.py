"""
Agents/base_agent.py

Defines the shared contract and shared behavior for all agents.

WHY THIS EXISTS:
GeneratorAgent and ReviewerAgent both need to (1) hold an LLMService,
and (2) load a role-specific system prompt from a text file. Instead
of duplicating that logic in every agent, we put it here once.

BaseAgent does NOT know about other agents, and does NOT talk to any
LLM provider directly -- it only talks to LLMService, which keeps all
provider access (keys, failover, rate limits) in one place.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from llm import LLMService


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, llm_service: LLMService, prompt_path: str) -> None:
        """
        Args:
            llm_service: LLMService injected by the Orchestrator. Agents never
                create their own, which keeps LLM configuration centralized
                and makes agents trivially testable with a fake service.
            prompt_path: Path to this agent's system prompt file.
        """
        self.llm_service = llm_service
        self.system_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: str) -> str:
        """Load a system prompt from a UTF-8 text file."""
        return Path(prompt_path).read_text(encoding="utf-8").strip()

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Execute this agent's task and return its output as a string."""
        raise NotImplementedError
