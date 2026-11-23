@echo off
rem Downloads the agent's LLM weights (~21 GB) into llm\models\. Resumable: just run again.
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
"%~dp0python\python.exe" scripts\get_llm_models.py %*
