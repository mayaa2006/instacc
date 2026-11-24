@echo off
rem Installs the free Piper voice engine (portable) and downloads a voice (~63 MB).
rem Usage: get-voice.bat [voice-name]   e.g. get-voice.bat en_GB-jenny_dioco-medium
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "PYTHONUTF8=1"
"%~dp0python\python.exe" scripts\get_voice.py %*
echo.
pause
