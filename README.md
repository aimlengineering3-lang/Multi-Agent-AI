# Multi-Agent AI System: Planner, Solver, Reviewer

An AI assistant that does not trust its own first answer. A **Planner** decides if a question needs fresh facts, a **Solver** writes a cited answer, and a **Reviewer** from a different model family fact-checks it and sends it back if it is not good enough.

**Live demo:** https://multi-agent-ai-reviewer-solver.streamlit.app

> A learning project built to understand how multi-agent LLM systems work end to end. It is not a production product (see [Limitations](#limitations)).

<!-- Add a screenshot after uploading it to docs/screenshot.png:
![Dashboard](docs/screenshot.png)
-->

## Why I built it

My first version asked a single model about the 2027 Cricket World Cup. It replied with a confident fixture and ticket prices that it had made up, and the Reviewer approved it with 9/10 because neither agent had any source to check against.

That failure shaped the design: **give the system evidence, make the Solver cite it, and make the Reviewer verify claims against it.** If no evidence is available, the system says what it cannot verify instead of guessing.

## How it works

```
                      ┌─▶ Planner ──▶ Tavily web search   (only if the facts can go stale)
User → Streamlit UI → Orchestrator
                      ├─▶ Solver ───┐
                      └─▶ Reviewer ─┴─▶ LLMService ─▶ Gemini │ Groq │ OpenRouter │ Cerebras
                                           (retry · cooldown · failover)
```

| Component | What it does |
|---|---|
| **Planner** | Decides whether the question needs web search and writes the queries. Falls back to a keyword heuristic if the model fails or returns bad JSON. |
| **Solver** | Writes the answer. When sources are available, it may only state time-sensitive facts found in them and cites them as [1], [2]. |
| **Reviewer** | Returns JSON: score (1-10), decision, issues, feedback. It is instructed to flag any specific claim the sources do not support. |
| **Orchestrator** | Runs the loop: plan, search, draft, review, revise. Stops on approval, on the iteration limit, or when a revision changes nothing. Returns the best-scored draft. |
| **LLMService** | The only module that talks to providers. Handles retries with backoff, a cooldown after rate limits (429), failover to the next provider, and removal of `<think>` blocks from reasoning models. |

By default the Solver runs on Gemini and the Reviewer on GPT-OSS 120B via Groq, so the second opinion comes from a different model family.

## Engineering decisions

- **Approval is not just a word.** A review counts as approved only if the model says APPROVED *and* the score is at least `MIN_APPROVAL_SCORE` (default 8). A reply that cannot be parsed is never treated as approval.
- **Bounded loop.** The first version had an unlimited loop. On free tiers that can burn a day's quota on one question, so it is now 3 rounds by default.
- **Graceful degradation.** If search fails, the answer is labeled "unverified". If the Reviewer's providers are all down, the draft is returned with a warning instead of an error.
- **One provider boundary.** Agents never import a provider SDK. Every provider speaks the OpenAI-compatible API, so adding or swapping one is configuration.
- **Offline tests.** 46 unit tests use fake providers and fake search, so they run without API keys.

## Tech stack

Python · Streamlit · OpenAI-compatible APIs (Gemini, Groq, OpenRouter, Cerebras) · Tavily search · pytest

## Project structure

```
app.py             Streamlit UI (progress, sources, internal Solver/Reviewer trace)
orchestrator.py    The plan → search → draft → review loop
llm.py             Provider client: retry, cooldown, failover
search.py          Tavily client and source formatting
config.py          Settings from environment variables
Agents/            base_agent.py, planner.py, generator.py (Solver), reviewer.py
prompts/           System prompts for each agent
tests/             Offline unit tests
```

## Run it locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # add at least one free API key
streamlit run app.py
```

Keys are free and need no credit card: [Groq](https://console.groq.com/keys), [Gemini](https://aistudio.google.com/apikey), [Tavily](https://app.tavily.com) (web search). Two LLM keys enable cross-model review and failover. Run the tests with `pip install -r requirements-dev.txt` and `pytest`.

To deploy on Streamlit Community Cloud, add the same keys under *Secrets*. Models, provider order and limits can be changed with environment variables listed in `.env.example`.

## Limitations

- **No evaluation yet.** I have not measured whether the Solver/Reviewer loop beats a single model call. That is the next thing to add.
- The Reviewer can only verify what the retrieved sources show, and the sources can be wrong.
- Web pages are passed to the model, so prompt injection from a malicious page is possible. There is no defence for it yet.
- Calls are synchronous, and the rate-limit cooldown lives in one process's memory.
- There is no login, logging, monitoring or cost tracking.
- Free tiers have strict limits and may use prompts to improve their models. Do not send sensitive data.

## Roadmap

- Evaluation: compare single call vs the loop on a fixed question set
- FastAPI service around the Orchestrator, with streaming responses
- Async provider calls and a shared cooldown store
- Defence against prompt injection in retrieved content
