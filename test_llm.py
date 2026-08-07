from llm import LLMService

llm = LLMService()

response = llm.generate(
    "Explain Python in one sentence."
)

print(response)