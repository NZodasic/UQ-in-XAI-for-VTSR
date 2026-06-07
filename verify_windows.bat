@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -c "import sys, torch, torchvision; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
exit /b %ERRORLEVEL%
