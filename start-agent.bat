@echo off
rem Starts the instacc agent (API + dashboard + Telegram bot + llama-server management).
rem Shows a QR code for your phone and opens the app in this PC's browser.
rem Extra args are passed through, e.g.:  start-agent.bat --host 127.0.0.1   (PC only)
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "HF_HOME=%~dp0cache\hf"
set "PYTHONUTF8=1"
if not exist "%~dp0python\python.exe" (
  echo Portable Python not found in python\ - run setup.bat first.
  exit /b 1
)
"%~dp0python\python.exe" -m agent %*
