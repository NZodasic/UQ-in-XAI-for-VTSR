@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set "BENCHMARK_DEFAULT_ARGS=--epochs 10 --patience 1 --batch-size 192 --num-workers 2 --uncertainty-samples 10 --xai-mc-samples 5 --xai-samples 5 --ig-steps 24 --checkpoint-every 1 --skip-xai-metrics"
if not "%BENCHMARK_FAST%"=="" set "BENCHMARK_DEFAULT_ARGS=%BENCHMARK_FAST%"

set PYTHONUNBUFFERED=1
"%PYTHON_EXE%" -u experiment_tools\run_full_benchmark.py --config configs/config.yaml %BENCHMARK_DEFAULT_ARGS% %*
exit /b %ERRORLEVEL%
