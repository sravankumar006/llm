# CommandLLM 💻⚡
> **Domain-Specific 124M Transformer Language Model for Natural Language to Linux & PowerShell Translation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React + Vite](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8.svg)](https://tailwindcss.com/)

---

## 📌 Project Overview
**CommandLLM** is an end-to-end causal decoder Transformer (~124M parameters) engineered from scratch in raw PyTorch. It translates ambiguous natural language sysadmin instructions into precise, executable **GNU/Linux (Bash)** and **Windows PowerShell** commands.

---

## 🏛️ Deep Learning Architecture
- **Model**: Custom causal decoder-only Transformer (`CoreCommandLLM`).
- **Layers & Attention**: 12 Transformer blocks, 12 attention heads, 768 hidden embedding dimension.
- **Context Length**: 256 tokens.
- **Vocabulary**: 50,263 tokens (50,257 base GPT-2 BPE tokens + 6 domain delimiter tokens).
- **Weight Tying**: Input embedding table (`wte.weight`) shared directly with the LM projection head (`lm_head.weight`), saving ~38.6M redundant parameters.
- **Causal Alignment**: Standard $t \to t+1$ logit-target alignment with selective loss masking (`ignore_index=-100`) on conditioning prefixes.
- **Special Tokens**:
  - `<|start|>` (50257): Sequence start delimiter.
  - `<|os|>` (50258): Operating system conditioning tag (`linux` or `powershell`).
  - `<|prompt|>` (50259): User natural language instruction marker.
  - `<|cmd|>` (50260): Supervised terminal command start marker.
  - `<|end|>` (50261): Command completion EOS token.
  - `<|pad|>` (50262): Uniform sequence padding token.

---

## 🚀 Quickstart

### 1. Python Environment Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Local CPU Inference Server (FastAPI)
```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```
API Docs available at `http://127.0.0.1:8000/docs`.

### 3. Developer Web Terminal Showcase (React + Vite + Tailwind)
```bash
cd web
npm install
npm run dev
```
Access the interactive web terminal at `http://127.0.0.1:5173`.

---

## ☁️ Google Colab Training Pipeline
1. Upload `commandllm_colab.zip` or clone this repository into Google Colab.
2. Open [`colab_train.ipynb`](./colab_train.ipynb).
3. Run the cells to fine-tune on GPU with Automatic Mixed Precision (AMP) and validation accuracy tracking (> 98% target threshold).
4. Download the newly produced checkpoint `checkpoints/terminal_model_final.pt`.

---

## 📂 Project Structure
```text
commandllm/
├── core/                   # Raw PyTorch Transformer Architecture
│   ├── config.py           # Hyperparameter specifications
│   ├── attention.py        # Causal multi-head self-attention
│   ├── layers.py           # Pre-LayerNorm decoder blocks & MLP
│   ├── model.py            # CoreCommandLLM decoder model
│   └── tokenizer.py        # BPE Tokenizer + special token mappings
├── data/                   # Dataset generation & preprocessing
│   ├── build_dataset.py    # Synthetic paired dataset generator
│   ├── tokenize_dataset.py # Tokenizer compiler & selective loss masking
│   ├── dataset.py          # PyTorch memory-mapped Dataset loader
│   ├── train.jsonl         # Training dataset pairs
│   └── val.jsonl           # Validation dataset pairs
├── scripts/                # Training, porting & export scripts
│   ├── port_weights.py     # Port base GPT-2 weights into CoreCommandLLM
│   ├── train.py            # AdamW + Cosine LR training loop with accuracy metrics
│   ├── make_notebook.py    # Generator for Google Colab training notebook
│   └── test_serving.py     # Automated API verification test suite
├── server/                 # Production FastAPI Inference Service
│   └── main.py             # Lifespan model loader, /api/generate & /api/compare
├── web/                    # React + Vite + TypeScript + Tailwind Terminal UI
│   ├── src/
│   │   ├── api/client.ts   # Typed API client
│   │   ├── components/     # TerminalWindow, DualTerminalView, TelemetryBar, ModelInspector
│   │   └── App.tsx         # Main interactive terminal application
│   └── vite.config.ts      # Reverse proxy to backend
├── colab_train.ipynb       # Google Colab GPU training notebook
├── commandllm_colab.zip    # Standalone training bundle for Colab
├── requirements.txt        # Local Python dependencies
└── colab_requirements.txt  # Colab GPU dependencies
```

---

## 📄 License
MIT License
