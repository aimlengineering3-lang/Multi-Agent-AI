# Multi-Agent AI System (Solver ↔ Reviewer)

A multi-agent system in Python. A **Solver** agent drafts an answer, a **Reviewer** agent critiques it against the original request, and the loop repeats until the Reviewer approves or the iteration limit is reached.

**Runs with zero local hardware requirements.** All inference goes to free hosted LLM APIs (Groq, Google Gemini, OpenRouter, Cerebras) with automatic failover. No GPU, no model downloads, no Ollama. It deploys to a public URL on Streamlit Community Cloud for free.

```
User → Streamlit UI → Orchestrator ─┬─▶ Planner ─▶ Tavily web search (only if facts can go stale)
                                    ├─▶ Solver Agent   ─┐
                                    │                   ├─▶ LLMService ─▶ Groq │ Gemini │ OpenRouter │ Cerebras
                                    └─▶ Reviewer Agent ─┘        (retry · cooldown · failover)
```

## What changed from the local-Ollama version

| Area | Before | Now |
|---|---|---|
| Model | `llama3.2:3b` on local CPU/GPU | Gemini Flash (Solver) + GPT-OSS 120B on Groq (Reviewer), hosted, free |
| Hardware | Needs a machine that can run Ollama | None (pure API calls) |
| Reviewer | Saw only the draft | Sees the **original request** + draft |
| Reviewer output | Free text, 2-line format | JSON: score 1-10, decision, issues, feedback (legacy format still parsed) |
| Approval rule | Model says "APPROVED" | Must say APPROVED **and** score ≥ `MIN_APPROVAL_SCORE` |
| Self-grading | Same model wrote and judged | Writer and reviewer prefer **different model families** |
| Prompts | One concatenated string | Proper `system` / `user` messages |
| Loop limit | `sys.maxsize` (unbounded) | 3 rounds; returns the **best-scored** draft; stops on stagnation |
| Failures | Crash on any error | Retry → cooldown → provider failover; reviewer outage degrades gracefully |
| Output length | 1024 tokens (truncated answers) | 2048 |
| Facts | Model guessed dates/prices | Planner + Tavily web search; sources cited by Solver, verified by Reviewer; honest "unverified" mode without search |
| Tests | Manual scripts needing a live model | 46 offline unit tests (fake providers) |

## Quick start (local)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then paste at least one free key
streamlit run app.py
```

Web search key (optional but recommended): [Tavily](https://app.tavily.com), 1,000 free credits/month.
Free LLM keys (no credit card): [Groq](https://console.groq.com/keys) · [Gemini](https://aistudio.google.com/apikey) · [OpenRouter](https://openrouter.ai/keys) · [Cerebras](https://cloud.cerebras.ai).
One key is enough. **Two or more** enables cross-model review and failover.

## Deploy for free (public URL)

1. Push this repo to GitHub (`.env` and `.streamlit/secrets.toml` are git-ignored).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo, main file `app.py`.
3. In *Advanced settings → Secrets*, add:
   ```toml
   GROQ_API_KEY = "..."
   GEMINI_API_KEY = "..."
   TAVILY_API_KEY = "..."
   ```

## Configuration

All settings are environment variables (see `.env.example`): `GENERATOR_PROVIDERS`, `REVIEWER_PROVIDERS`, `GROQ_MODEL`, `GEMINI_MODEL`, `MAX_ITERATIONS`, `MIN_APPROVAL_SCORE`, `MAX_TOKENS`. Free-tier model IDs change over time, so override them here instead of editing code.

To run the original local model instead, set `GENERATOR_PROVIDERS=ollama` and `REVIEWER_PROVIDERS=ollama` (uses Ollama's OpenAI-compatible endpoint).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use scripted fake providers, so they run offline with no keys. Covered: reviewer parsing edge cases, rate-limit failover and cooldown, retry behavior, orchestrator stopping conditions, best-draft selection, and outage handling.

## Grounded answers (no more invented facts)

Time-sensitive questions (schedules, fixtures, prices, news) are routed through a **Planner** that triggers a Tavily search. The Solver may only state specifics found in the numbered sources (and cites them as [1], [2]); the Reviewer rejects any specific claim the sources do not support. If search is unavailable, the Solver is instructed to say what it cannot verify and where to check, and the UI shows an "unverified" warning.

## Design rules

- Agents never call each other; only the Orchestrator coordinates them.
- Only `LLMService` talks to providers (keys, retries, failover in one place).
- An unparseable review is never treated as approval.
- Secrets live in the environment, never in source.

## Limitations

- Free tiers have per-minute/day limits and may use prompts to improve their models; don't send sensitive data.
- Model quality is that of the hosted open/free models; the Reviewer improves reliability but cannot guarantee factual accuracy.
