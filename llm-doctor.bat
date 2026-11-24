@echo off
rem Diagnoses slow or nonsensical AI output. Run it while start-agent.bat is running.
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
"%~dp0python\python.exe" scripts\llm_doctor.py
echo.
pause
