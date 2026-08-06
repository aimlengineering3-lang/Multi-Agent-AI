"""
app.py

Streamlit UI for the Multi-Agent AI system.
Shows only final, approved answers to the user — Solver/Reviewer
internal back-and-forth is hidden by default.
"""

import streamlit as st
from orchestrator import Orchestrator


@st.cache_resource
def get_orchestrator() -> Orchestrator:
    """Create the Orchestrator once and reuse it across reruns."""
    return Orchestrator()


def init_chat_history() -> None:
    """
    Initialize conversation history in Streamlit's session state.

    WHY session_state:
    Streamlit reruns the whole script on every interaction. Normal
    Python variables would reset every time. session_state persists
    across reruns for the same browser session — it's how we keep
    a running chat history instead of only showing the latest answer.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main() -> None:
    st.set_page_config(page_title="Multi-Agent AI", page_icon="🤖", layout="centered")

    st.title("🤖 Multi-Agent AI System")
    st.caption("Solver + Reviewer agents iterating until the answer is approved — powered by local Ollama (llama3.2:3b)")
    st.divider()

    init_chat_history()
    orchestrator = get_orchestrator()

    # Render existing conversation history first.
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(
                f"{'✅ Approved' if turn['approved'] else '⚠️ Max iterations reached'} "
                f"· {turn['iterations']} iteration(s)"
            )
            with st.expander("Show internal Solver/Reviewer process"):
                for step in turn["history"]:
                    st.markdown(f"**Iteration {step['iteration']}** — `{step['decision']}`")
                    st.markdown(f"*Draft:*\n\n{step['draft']}")
                    st.markdown(f"*Reviewer response:*\n\n{step['raw_review']}")
                    st.divider()

    # New input at the bottom, chat-app style.
    user_input = st.chat_input("Ask something...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Solver and Reviewer are working on your answer..."):
                result = orchestrator.run(user_input)

            st.write(result["final_answer"])
            st.caption(
                f"{'✅ Approved' if result['approved'] else '⚠️ Max iterations reached'} "
                f"· {result['iterations_used']} iteration(s)"
            )
            with st.expander("Show internal Solver/Reviewer process"):
                for step in result["history"]:
                    st.markdown(f"**Iteration {step['iteration']}** — `{step['decision']}`")
                    st.markdown(f"*Draft:*\n\n{step['draft']}")
                    st.markdown(f"*Reviewer response:*\n\n{step['raw_review']}")
                    st.divider()

        # Save this turn so it persists across reruns.
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": result["final_answer"],
            "approved": result["approved"],
            "iterations": result["iterations_used"],
            "history": result["history"],
        })


if __name__ == "__main__":
    main()