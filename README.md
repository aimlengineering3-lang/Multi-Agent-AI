# 🤖 Multi-Agent Autonomous AI Software Engineering Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph Orchestration](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit Interface](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)
[![Ollama Local LLM](https://img.shields.io/badge/llm-Ollama%20Local-orange.svg)](https://ollama.com/)

---

## 📌 Executive Summary

The **Multi-Agent Autonomous AI Software Engineering Platform** is an enterprise-grade local multi-agent software development framework. It deploys specialized AI agents operating in an autonomous **Developer ⟷ QA Lead** revision loop to generate, compile, test, debug, and deliver production-ready software applications directly to disk.

Unlike single-prompt code generators, this platform features an **Autonomous QA Execution Sandbox (`qa_runner.py`)** that performs real-time Python AST compilation, module import validation, and automated unit test suite execution (`unittest discover`) in isolated sandboxes. Stack traces and compiler errors are fed directly into the QA Lead Agent to guarantee bug-free software delivery.

---

## 🏗️ System Architecture & Workflow

                           User Software Requirement / Prompt
                                           │
                                           ▼
                                👨‍💻 Developer Agent
                              (Qwen2.5-Coder / 30b)
                                           │
                                  Generates Code Draft
                                           │
                                           ▼
                          🧪 Real QA Execution Sandbox (`qa_runner.py`)
                          ├── Isolated Workspace File Generation
                          ├── Python AST Syntax & PyCompile Validation
                          ├── Automated Test Discovery & Execution (`unittest`)
                          └── Real Execution Stack Trace Generation
                                           │
                                           ▼
                                 🧪 QA Lead Agent
                                  (DeepSeek-R1)
                                           │
                                 Evaluates Code + Stack Trace
                                           │
                                           ▼
                                  QA Approved?
                                     │    │
                                    No   Yes
                                     │    │
                                     ▼    ▼
                             QA Defect    Final Software Approved
                             Report &        │
                            Stack Trace      ▼
                               │      Saved to disk folder:
                               │      generated_projects/<timestamp>-<app>/
                      Developer Fixes
                        & Merges Files
                               │
                               └──────────► Repeat Loop (Up to 25 Rounds)

---

## 🔥 Key Technical Highlights

1. **Autonomous Real Code Execution & Test Sandbox (`qa_runner.py`):**
   - Automatically writes project files into isolated temporary sandboxes.
   - Runs `ast.parse` and `py_compile` checks to detect syntax errors, indentation flaws, and invalid imports before delivery.
   - Discovers and executes unit test suites (`unittest discover`) with strict timeout protections, feeding actual execution stack traces back to the QA Lead Agent.

2. **Multi-Round File Accumulation & Zero-Loss Code Retention:**
   - Dynamically tracks `accumulated_files` across revision rounds. Frontend assets (`index.html`, `styles.css`, `app.js`) and backend modules are preserved and merged when backend fixes occur in later rounds.

3. **Live Token Streaming & Auto-Expanding Context Window:**
   - Real-time status reporting (`⏳ model generating response...`) updated every 80 tokens.
   - Automatic `exceed_context_size_error` detection with dynamic context expansion (16k ➔ 32k ➔ 64k) and seamless retry.

4. **Robust Orchestration:**
   - Dual support for **LangGraph StateGraph Engine** and **Native High-Performance Workflow Loop**.
   - Fully customizable max iteration limit (up to 25 rounds).

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Core Framework** | Python 3.10+ |
| **Agent Orchestration** | LangGraph & Custom State Machines |
| **LLM Runtime Engine** | Ollama (Local Server) |
| **Developer Model** | `qwen2.5-coder:7b` / `qwen3-coder:30b` |
| **QA Lead Model** | `deepseek-r1` |
| **User Interface** | Streamlit Web UI |
| **Testing Sandbox** | Python `ast`, `py_compile`, `unittest`, `subprocess` |

---

## 🚀 Quickstart Guide

### 1. Launch via One-Click Batch Script
Simply double click `run_app.bat` or run:
```powershell
.\run_app.bat
cd Multi-Agent-AI-main
streamlit run app.py
🧪 Automated System Verification
Run the comprehensive unit test suite covering state accumulation, execution testing, and context retries:

powershell
python -m unittest discover -s Multi-Agent-AI-main/tests
