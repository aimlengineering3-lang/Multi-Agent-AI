"""
config.py

Centralized configuration for the Multi-Agent AI system.
"""

import sys

# LLM Configuration
MODEL_NAME: str = "llama3.2:3b"
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 1024

# Maximum number of Generator <-> Reviewer iterations.
# Uses the maximum platform integer (effectively unlimited for practical purposes).
MAX_ITERATIONS: int = sys.maxsize