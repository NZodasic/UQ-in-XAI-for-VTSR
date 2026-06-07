@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Starting TSR Experiment on Windows...
set PYTHONUNBUFFERED=1
"%PYTHON_EXE%" -u main.py --config configs/config.yaml %*
exit /b %ERRORLEVEL%
