"""
app.py

Streamlit UI for the Multi-Agent AI system.

RESPONSIBILITY OF THIS FILE:
This file does NOT decide how the Solver/Reviewer loop works, does
NOT talk to Ollama, and does NOT know how agents communicate. All of
that lives in Orchestrator and below. This file's ONLY job is:
    1. Collect user input
    2. Call Orchestrator.run()
    3. Display the result nicely

WHY THE UI LOOKS THE WAY IT DOES:
Streamlit's default theme has a known contrast issue with st.chat_input
placeholder text (light gray text on a light gray background). We fix
this -- and polish the rest of the interface -- by injecting custom CSS
via st.markdown(). Streamlit doesn't expose a clean theming API for
every element, so injecting raw CSS with unsafe_allow_html=True is the
standard, accepted way to fully control appearance in a Streamlit app.
"""

import streamlit as st
from orchestrator import Orchestrator
from config import MODEL_NAME, MAX_ITERATIONS


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# WHY THIS MUST BE THE FIRST STREAMLIT COMMAND:
# st.set_page_config() configures the browser tab title/icon and
# page layout. Streamlit requires this to be the very first Streamlit
# command executed -- calling anything else first raises a runtime error.
st.set_page_config(
    page_title="Multi-Agent AI System",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================
def inject_custom_css() -> None:
    """
    Inject custom CSS to override Streamlit's default styling.

    WHY A SEPARATE FUNCTION:
    Keeping CSS in its own function keeps main() focused on layout/
    flow logic, not styling details -- same Single Responsibility
    idea we've applied throughout the backend, applied to UI code too.
    """
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #0f1117 0%, #14161f 100%);
        }

        .app-header {
            padding: 1.75rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #1a1d29 0%, #21253a 100%);
            border: 1px solid #2a2f45;
            margin-bottom: 1.5rem;
        }
        .app-header h1 {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            color: #f4f5f7;
        }
        .app-header p {
            color: #9aa0b4;
            font-size: 0.95rem;
            margin: 0;
        }
        .badge-row {
            margin-top: 0.9rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .badge {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: #262b41;
            color: #a6b4ff;
            border: 1px solid #363c58;
        }

        /* ---- Fix: chat_input contrast (the original bug) ---- */
        [data-testid="stChatInput"] textarea {
            background-color: #1c1f2b !important;
            color: #f4f5f7 !important;
            border: 1.5px solid #363c58 !important;
            border-radius: 12px !important;
            font-size: 0.98rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #7d84a0 !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] {
            background-color: transparent !important;
        }

        [data-testid="stChatMessage"] {
            background-color: #171a24;
            border: 1px solid #262b3d;
            border-radius: 14px;
            padding: 0.5rem 0.25rem;
            margin-bottom: 0.6rem;
        }

        .status-approved {
            color: #4ade80;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .status-warning {
            color: #fbbf24;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .ai-disclaimer {
            font-size: 0.78rem;
            color: #6b7290;
            margin-top: 0.4rem;
            font-style: italic;
        }

        .empty-state {
            text-align: center;
            padding: 3rem 1.5rem;
            color: #6b7290;
        }
        .empty-state h3 {
            color: #9aa0b4;
            font-weight: 600;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background-color: #14161f !important;
            border-right: 1px solid #262b3d;
        }
        [data-testid="stSidebar"] * {
            color: #e4e6ee !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #262b3d !important;
        }

        [data-testid="stExpander"] {
            background-color: #12141c;
            border: 1px solid #262b3d;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKEND WIRING
# ============================================================
@st.cache_resource
def get_orchestrator() -> Orchestrator:
    """
    Create the Orchestrator once and reuse it across reruns.

    WHY @st.cache_resource:
    Streamlit reruns the ENTIRE script on every interaction. Without
    caching, we'd create a brand new Orchestrator (and LLMService)
    on every rerun, wasting setup cost. This is similar to a
    singleton pattern.
    """
    return Orchestrator()


def init_session_state() -> None:
    """
    Initialize persistent state for this browser session.

    WHY st.session_state:
    Because Streamlit reruns the whole script every time, a normal
    variable like `chat_history = []` would reset every rerun.
    session_state is preserved across reruns for the same session.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ============================================================
# UI: SIDEBAR
# ============================================================
def render_sidebar() -> None:
    """Render the sidebar: system info, settings, and controls."""
    with st.sidebar:
        st.markdown("### ⚙️ System Info")
        st.markdown(f"**Model:** `{MODEL_NAME}`")
        st.markdown(f"**Max iterations:** `{MAX_ITERATIONS}`")
        st.markdown("**Backend:** Local Ollama (no cloud API calls)")

        st.divider()

        st.markdown("### 🔁 How it works")
        st.markdown(
            "1. **Solver Agent** drafts an answer.\n"
            "2. **Reviewer Agent** checks it for correctness, "
            "completeness, and clarity.\n"
            "3. If revision is needed, the Solver improves the draft "
            "and the Reviewer checks again.\n"
            "4. This repeats until the Reviewer approves, or the "
            f"iteration limit ({MAX_ITERATIONS}) is reached.\n"
            "5. You only see the **final** answer -- the back-and-forth "
            "stays behind the scenes, but you can inspect it below "
            "each answer."
        )

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption(
            "⚠️ This is a local, experimental multi-agent system built "
            "from scratch in pure Python (no LangGraph/CrewAI/AutoGen). "
            "Responses come from a small local model and may contain "
            "mistakes -- always verify important information."
        )


# ============================================================
# UI: A SINGLE CHAT TURN
# ============================================================
def render_assistant_turn(final_answer: str, approved: bool, iterations: int, history: list) -> None:
    """
    Render one assistant response block: answer, status badge,
    disclaimer, and a collapsible internal process trace.

    WHY THIS IS A SEPARATE FUNCTION:
    The exact same rendering logic is needed twice -- once for saved
    history replay, once for a brand-new response. Extracting it
    avoids duplicating this UI block (DRY principle applied to UI).
    """
    st.write(final_answer)

    if approved:
        st.markdown(
            f'<span class="status-approved">✅ Approved by Reviewer</span> '
            f'&nbsp;·&nbsp; {iterations} iteration(s)',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-warning">⚠️ Best effort -- max iterations reached</span> '
            f'&nbsp;·&nbsp; {iterations} iteration(s)',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="ai-disclaimer">AI can make mistakes. Please verify important information.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Show internal Solver ↔ Reviewer process"):
        for step in history:
            st.markdown(f"**Iteration {step['iteration']}** -- Decision: `{step['decision']}`")
            st.markdown("**Solver draft:**")
            st.info(step["draft"])
            st.markdown("**Reviewer response:**")
            st.warning(step["raw_review"])
            st.divider()


# ============================================================
# UI: CHAT TAB
# ============================================================
def render_chat_tab(orchestrator: Orchestrator) -> None:
    """Render the main chat interface: history + input box."""

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <h3>👋 Ask me anything</h3>
                <p>Your question goes through a Solver Agent and a Reviewer Agent
                that iterate until the answer meets quality standards.<br>
                Try: <i>"Explain what a REST API is, like I'm a 10 year old."</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for turn in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑"):
            st.write(turn["question"])
        with st.chat_message("assistant", avatar="🤖"):
            render_assistant_turn(
                turn["answer"], turn["approved"], turn["iterations"], turn["history"]
            )

    user_input = st.chat_input("Ask something...")

    if user_input:
        with st.chat_message("user", avatar="🧑"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Solver is drafting an answer, Reviewer is checking it..."):
                result = orchestrator.run(user_input)

            render_assistant_turn(
                result["final_answer"],
                result["approved"],
                result["iterations_used"],
                result["history"],
            )

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": result["final_answer"],
            "approved": result["approved"],
            "iterations": result["iterations_used"],
            "history": result["history"],
        })


# ============================================================
# UI: ABOUT TAB
# ============================================================
def render_about_tab() -> None:
    """Render a short explanation of the system architecture."""
    st.markdown("### 🏗️ Architecture")
    st.markdown(
        """
This system is built entirely in **pure Python** -- no LangGraph, no CrewAI,
no AutoGen -- to make the internal mechanics of a multi-agent AI workflow
fully transparent.

```
User
  │
  ▼
Streamlit UI  (app.py)
  │
  ▼
Orchestrator  (orchestrator.py)
  │
  ├──▶ Solver Agent   ──┐
  │                     ├──▶  LLMService  ──▶  Ollama (llama3.2:3b)
  └──▶ Reviewer Agent ──┘
```
"""
    )

    st.markdown("### 📐 Design rules this project follows")
    st.markdown(
        """
- **Solver and Reviewer never call each other directly.** Only the
  Orchestrator coordinates them.
- **Only `LLMService` talks to Ollama.** Agents never import the
  `ollama` library directly -- this keeps the LLM backend swappable.
- **The loop has a hard iteration limit**, so a stubborn Reviewer or a
  non-converging model can't loop forever.
- **The user only ever sees the final answer.** All intermediate
  drafts and reviewer feedback are available on request, but hidden
  by default.
"""
    )

    st.info(
        "⚠️ **Disclaimer:** This is an educational/experimental project "
        "running a small local model. It can produce incorrect, incomplete, "
        "or outdated information. Do not rely on it for medical, legal, "
        "financial, or safety-critical decisions."
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main() -> None:
    inject_custom_css()
    init_session_state()

    st.markdown(
        """
        <div class="app-header">
            <h1>🤖 Multi-Agent AI System</h1>
            <p>A Solver Agent and a Reviewer Agent iterate together until the answer
            is approved -- running fully locally on Ollama.</p>
            <div class="badge-row">
                <span class="badge">🧩 Pure Python -- no agent frameworks</span>
                <span class="badge">🔒 Runs 100% locally</span>
                <span class="badge">🔁 Iterative self-review</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()
    orchestrator = get_orchestrator()

    tab_chat, tab_about = st.tabs(["💬 Chat", "ℹ️ About This System"])

    with tab_chat:
        render_chat_tab(orchestrator)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()