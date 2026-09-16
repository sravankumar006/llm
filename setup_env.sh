#!/usr/bin/env bash
set -e

# ==============================================================================
# CommandLLM - Local CPU Environment Setup Script (Linux / macOS / WSL)
# ==============================================================================

echo "=========================================================="
echo "    CommandLLM: Initializing Local Development Env"
echo "=========================================================="

# 1. Locate Python 3.10+ executable
PYTHON_BIN=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &> /dev/null; then
        # Check version >= 3.10
        IS_VALID=$("$cmd" -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>/dev/null || echo "0")
        if [ "$IS_VALID" = "1" ]; then
            PYTHON_BIN="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3.10 or higher was not found on your PATH."
    echo "Please install Python 3.10+ and re-run this script."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "[OK] Found compatible Python: $PYTHON_BIN (version $PYTHON_VERSION)"

# 2. Setup virtual environment (.venv)
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at '$VENV_DIR'..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[INFO] Virtual environment '$VENV_DIR' already exists."
fi

# 3. Activate virtual environment
echo "[INFO] Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 4. Upgrade pip and install requirements
echo "[INFO] Upgrading pip tooling..."
pip install --upgrade pip setuptools wheel --quiet

echo "[INFO] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Verification & Diagnostics
echo ""
echo "=========================================================="
echo "           Environment Verification Diagnostics           "
echo "=========================================================="
python - << 'EOF'
import sys
import torch
import numpy
import fastapi
import tiktoken

print(f" Python Executable : {sys.executable}")
print(f" Python Version    : {sys.version.split()[0]}")
print(f" PyTorch Version   : {torch.__version__}")
print(f" NumPy Version     : {numpy.__version__}")
print(f" FastAPI Version   : {fastapi.__version__}")
print(f" TikToken Version  : {tiktoken.__version__}")

is_cuda = torch.cuda.is_available()
device_label = f"CUDA (GPU: {torch.cuda.get_device_name(0)})" if is_cuda else "CPU (Standard local inference target)"
print(f" Target Device     : {device_label}")
print(f" Tensor Allocation : {torch.randn(2, 2).device} test tensor successfully created.")
EOF

echo ""
echo "=========================================================="
echo " [SUCCESS] CommandLLM environment is configured & ready!  "
echo " Activate with: source .venv/bin/activate"
echo "=========================================================="
