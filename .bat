@echo off
title Launch Multi-Agent AI Software Team
echo =======================================================
echo   Starting Multi-Agent AI Developer & QA Team (Streamlit)
echo =======================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
cd Multi-Agent-AI-main
streamlit run app.py
pause
