@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM CommandLLM - Local CPU Environment Setup Script (Windows Command Prompt)
REM ============================================================================

echo ==========================================================
echo     CommandLLM: Initializing Local Development Env
echo ==========================================================

REM 1. Detect Python 3.10 - 3.12 (preferring 3.12/3.11 for PyTorch prebuilt wheel compatibility)
set "PYTHON_EXE="

REM Try specific stable Python versions via py launcher first
where py >nul 2>nul
if %errorlevel% equ 0 (
    for %%v in (3.12 3.11 3.10 3.13 3) do (
        if not defined PYTHON_EXE (
            for /f "tokens=*" %%i in ('py -%%v -c "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')" 2^>nul') do (
                if not "%%i"=="" set "PYTHON_EXE=%%i"
            )
        )
    )
)

REM Fallback to python on PATH
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')" 2^>nul') do (
            if not "%%i"=="" set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python 3.10 or higher was not detected.
    echo Please install Python 3.10+ from python.org or Microsoft Store and ensure it is on PATH.
    exit /b 1
)

for /f "tokens=*" %%v in ('"%PYTHON_EXE%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do (
    set "PYTHON_VERSION=%%v"
)
echo [OK] Found Python %PYTHON_VERSION% at: %PYTHON_EXE%

REM 2. Create virtual environment if missing
set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment in '%VENV_DIR%'...
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b %errorlevel%
    )
) else (
    echo [INFO] Virtual environment '%VENV_DIR%' already exists.
)

REM 3. Activate virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM 4. Upgrade pip tooling and install requirements
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet

echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    exit /b %errorlevel%
)

REM 5. Diagnostics Verification
echo.
echo ==========================================================
echo            Environment Verification Diagnostics           
echo ==========================================================
python -c "import sys, torch, numpy, fastapi, tiktoken; print(f' Python Executable : {sys.executable}'); print(f' Python Version    : {sys.version.split()[0]}'); print(f' PyTorch Version   : {torch.__version__}'); print(f' NumPy Version     : {numpy.__version__}'); print(f' FastAPI Version   : {fastapi.__version__}'); print(f' TikToken Version  : {tiktoken.__version__}'); is_cuda = torch.cuda.is_available(); dev_msg = f'CUDA ({torch.cuda.get_device_name(0)})' if is_cuda else 'CPU (Standard local inference target)'; print(f' Target Device     : {dev_msg}'); t = torch.randn(2, 2); print(f' Tensor Check      : Successfully allocated {tuple(t.shape)} tensor on {t.device}.')"

echo.
echo ==========================================================
echo  [SUCCESS] CommandLLM environment is configured and ready! 
echo  Activate manually with: .venv\Scripts\activate.bat
echo ==========================================================
