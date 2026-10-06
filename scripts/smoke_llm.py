import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from llm import LLMService

llm = LLMService()

response = llm.generate(
    "Explain Python in one sentence."
)

print(response)