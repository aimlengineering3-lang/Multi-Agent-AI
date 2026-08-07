"""
config.py

Centralized configuration for the Multi-Agent AI system.
"""

MODEL_NAME: str = "llama3.2:3b"
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 1024

# Maximum number of Solver <-> Reviewer iterations before we give up
# and return the best available draft, even if not formally approved.
# WHY THIS EXISTS: without a hard limit, a stubborn Reviewer or a model
# that never converges could loop forever, burning time and compute.
MAX_ITERATIONS: int = 10