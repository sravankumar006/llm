<#
.SYNOPSIS
    CommandLLM - Local CPU Environment Setup Script (Windows PowerShell)
.DESCRIPTION
    Checks Python 3.10+, provisions .venv, installs CPU PyTorch dependencies,
    and runs diagnostic verification.
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "    CommandLLM: Initializing Local Development Env" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Detect Python 3.10+ (preferring 3.12/3.11/3.10 for PyTorch binary wheel compatibility)
$pythonCmd = $null
$pythonArgs = @()

# First test py launcher with specific stable versions
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($ver in @("-3.12", "-3.11", "-3.10", "-3.13", "-3")) {
        $check = & py $ver -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>$null
        if ($check -eq "1") {
            $pythonCmd = "py"
            $pythonArgs = @($ver)
            break
        }
    }
}

# Fallback to direct executables
if (-not $pythonCmd) {
    $candidates = @("python3.12", "python3.11", "python3.10", "python3", "python")
    foreach ($cmd in $candidates) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            $check = & $cmd -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>$null
            if ($check -eq "1") {
                $pythonCmd = $cmd
                $pythonArgs = @()
                break
            }
        }
    }
}

if (-not $pythonCmd) {
    Write-Error "[ERROR] Python 3.10 or higher was not detected. Please install Python 3.10+ and add it to your PATH."
    exit 1
}

$pyVersion = & $pythonCmd @pythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$pyPath = & $pythonCmd @pythonArgs -c "import sys; print(sys.executable)"
Write-Host "[OK] Detected Python $pyVersion at: $pyPath" -ForegroundColor Green

# 2. Virtual Environment Creation
$venvPath = Join-Path $PSScriptRoot ".venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "[INFO] Creating virtual environment at '$venvPath'..." -ForegroundColor Yellow
    & $pythonCmd @pythonArgs -m venv $venvPath
} else {
    Write-Host "[INFO] Virtual environment '$venvPath' already exists." -ForegroundColor Gray
}

# 3. Activate Virtual Environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
& $activateScript

# 4. Pip installation
$venvPy = Join-Path $venvPath "Scripts\python.exe"
Write-Host "[INFO] Upgrading pip tooling..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip setuptools wheel --quiet

Write-Host "[INFO] Installing requirements from requirements.txt..." -ForegroundColor Yellow
& $venvPy -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

# 5. Diagnostic Verification
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "           Environment Verification Diagnostics           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$diagCode = @"
import sys, torch, numpy, fastapi, tiktoken
print(f' Python Executable : {sys.executable}')
print(f' Python Version    : {sys.version.split()[0]}')
print(f' PyTorch Version   : {torch.__version__}')
print(f' NumPy Version     : {numpy.__version__}')
print(f' FastAPI Version   : {fastapi.__version__}')
print(f' TikToken Version  : {tiktoken.__version__}')
is_cuda = torch.cuda.is_available()
dev_label = f'CUDA ({torch.cuda.get_device_name(0)})' if is_cuda else 'CPU (Standard local inference target)'
print(f' Target Device     : {dev_label}')
t = torch.randn(2, 2)
print(f' Tensor Check      : Successfully allocated {tuple(t.shape)} tensor on {t.device}.')
"@

& $venvPy -c $diagCode

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [SUCCESS] CommandLLM environment is configured and ready! " -ForegroundColor Green
Write-Host " Activate in PowerShell: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
