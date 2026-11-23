@echo off
rem Installs the agent's Python packages into the portable python\ (nothing system-wide).
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "PYTHONUTF8=1"
"%~dp0python\python.exe" -m pip install --no-warn-script-location -r requirements-agent.txt
