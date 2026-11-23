@echo off
rem Starts the instacc agent (API + dashboard + Telegram bot + llama-server management).
rem Extra args are passed through, e.g.:  start-agent.bat --host 0.0.0.0
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "HF_HOME=%~dp0cache\hf"
set "PYTHONUTF8=1"
if not exist "%~dp0python\python.exe" (
  echo Portable Python not found in python\ - run setup.bat first.
  exit /b 1
)
"%~dp0python\python.exe" -m agent %*
