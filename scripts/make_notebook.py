"""Utility to generate the colab_train.ipynb notebook artifact."""
import json
from pathlib import Path

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CommandLLM: Google Colab Training & Fine-Tuning Pipeline (Target > 98% Accuracy)\n",
                "This notebook fine-tunes **CoreCommandLLM** (124M Custom Decoder Transformer) on domain-specific Linux & PowerShell sysadmin commands.\n",
                "\n",
                "- **Base Architecture**: 12-layer decoder Transformer with weight tying (`wte` ↔ `lm_head`) and 50,263 vocabulary\n",
                "- **Causal Alignment**: Standard $t \\to t+1$ logit-target shifting with selective loss masking (`ignore_index=-100`)\n",
                "- **Objective**: Achieve token & sequence validation accuracy > 98.0%\n",
                "- **Hardware**: NVIDIA GPU with Automatic Mixed Precision (AMP)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Environment & GPU Verification\n",
                "!nvidia-smi\n",
                "\n",
                "import torch\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available : {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'Device Name    : {torch.cuda.get_device_name(0)}')\n",
                "    print(f'Total VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')\n",
                "    print(f'BF16 Supported : {torch.cuda.is_bf16_supported()}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 2: Project Setup & Workspace Directory\n",
                "Upload your project zip (`commandllm.zip`) to Colab or clone your repository."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Extract project archive or navigate to workspace\n",
                "import os\n",
                "\n",
                "if os.path.exists('/content/commandllm.zip'):\n",
                "    !unzip -q /content/commandllm.zip -d /content/\n",
                "    %cd /content/commandllm\n",
                "elif os.path.exists('/content/commandllm'):\n",
                "    %cd /content/commandllm\n",
                "else:\n",
                "    print('Current working directory:', os.getcwd())\n",
                "\n",
                "!ls -la"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 3: Install Required Dependencies\n",
                "Installs dependencies from `colab_requirements.txt`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Install dependencies\n",
                "!pip install -r colab_requirements.txt"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 3.5: Port Base GPT-2 Weights to CoreCommandLLM Checkpoint\n",
                "Downloads pre-trained GPT-2 (124M) weights from Hugging Face and initializes `checkpoints/custom_base_checkpoint.pt` directly on the Colab instance in ~30 seconds (no 500MB upload needed)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3.5: Build custom base checkpoint from Hugging Face GPT-2\n",
                "!python scripts/port_weights.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 4: Launch Fine-Tuning Pipeline (Target Accuracy > 98%)\n",
                "Trains CoreCommandLLM with AdamW, Cosine LR scheduling with linear warmup, token-level accuracy tracking, and automatic checkpointing on best validation accuracy record."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Execute fine-tuning pipeline with calibrated hyperparameters\n",
                "!python -u scripts/train.py \\\n",
                "    --epochs 5 \\\n",
                "    --batch_size 16 \\\n",
                "    --learning_rate 5e-5 \\\n",
                "    --min_lr 5e-6 \\\n",
                "    --warmup_steps 250 \\\n",
                "    --weight_decay 0.01 \\\n",
                "    --grad_clip 1.0 \\\n",
                "    --checkpoint_path checkpoints/custom_base_checkpoint.pt \\\n",
                "    --save_path checkpoints/terminal_model_final.pt"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 5: Final Validation Benchmark (20 Unseen Prompts Evaluation)\n",
                "Evaluates model generation against 20 unseen validation prompts, compares generated commands against ground truth, and outputs formal benchmark accuracy."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Benchmark evaluation on 20 unseen validation prompts\n",
                "import os\n",
                "import json\n",
                "import torch\n",
                "from core.config import CommandLMConfig\n",
                "from core.model import CoreCommandLLM\n",
                "from core.tokenizer import CommandTokenizer\n",
                "from scripts.train import check_argument_preservation, normalize_command\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Running Benchmark on Device: {device}')\n",
                "\n",
                "# Load model and tokenizer\n",
                "config = CommandLMConfig()\n",
                "model = CoreCommandLLM(config)\n",
                "ckpt_path = 'checkpoints/terminal_model_final.pt'\n",
                "state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)\n",
                "model.load_state_dict(state_dict)\n",
                "model.to(device)\n",
                "model.eval()\n",
                "tokenizer = CommandTokenizer()\n",
                "\n",
                "# Read 20 unseen validation samples\n",
                "val_samples = []\n",
                "val_file = 'data/val.jsonl'\n",
                "with open(val_file, 'r', encoding='utf-8') as f:\n",
                "    for line in f:\n",
                "        line = line.strip()\n",
                "        if line:\n",
                "            val_samples.append(json.loads(line))\n",
                "        if len(val_samples) >= 20:\n",
                "            break\n",
                "\n",
                "print('=' * 95)\n",
                "print(f' {\"OS\":<10} | {\"PROMPT\":<30} | {\"MATCH\":<5} | {\"ARGS\":<5} | {\"GENERATED COMMAND\"}')\n",
                "print('=' * 95)\n",
                "\n",
                "total_eval = len(val_samples)\n",
                "exact_matches = 0\n",
                "norm_matches = 0\n",
                "arg_preserved = 0\n",
                "token_correct_sum = 0\n",
                "token_total_sum = 0\n",
                "\n",
                "for idx, sample in enumerate(val_samples, 1):\n",
                "    os_type = sample['os']\n",
                "    prompt = sample['prompt']\n",
                "    ground_truth = sample['cmd'].strip()\n",
                "\n",
                "    prefix = tokenizer.format_prompt(os_type, prompt)\n",
                "    input_ids = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)\n",
                "\n",
                "    with torch.no_grad():\n",
                "        out = model.generate(\n",
                "            input_ids,\n",
                "            max_new_tokens=48,\n",
                "            temperature=0.0,\n",
                "            top_k=None,\n",
                "            eos_token_id=tokenizer.eos_token_id,\n",
                "            repetition_penalty=1.0,\n",
                "        )\n",
                "\n",
                "    gen_tokens = out[0, len(input_ids[0]):].tolist()\n",
                "    raw_decoded = tokenizer.decode(gen_tokens)\n",
                "    pred_cmd = raw_decoded.split('<|end|>')[0].strip()\n",
                "\n",
                "    is_match = (pred_cmd == ground_truth or pred_cmd.lower() == ground_truth.lower())\n",
                "    if is_match:\n",
                "        exact_matches += 1\n",
                "\n",
                "    if normalize_command(pred_cmd).lower() == normalize_command(ground_truth).lower():\n",
                "        norm_matches += 1\n",
                "\n",
                "    ok_args, missing_args = check_argument_preservation(prompt, pred_cmd)\n",
                "    if ok_args:\n",
                "        arg_preserved += 1\n",
                "\n",
                "    # Token-level overlap\n",
                "    gt_tokens = tokenizer.encode(ground_truth)\n",
                "    pred_toks = tokenizer.encode(pred_cmd)\n",
                "    min_len = min(len(gt_tokens), len(pred_toks))\n",
                "    t_correct = sum(1 for i in range(min_len) if gt_tokens[i] == pred_toks[i])\n",
                "    token_correct_sum += t_correct\n",
                "    token_total_sum += max(len(gt_tokens), len(pred_toks), 1)\n",
                "\n",
                "    status_sym = '✓' if is_match else '~'\n",
                "    arg_sym = '✓' if ok_args else '✗'\n",
                "    prompt_trunc = (prompt[:28] + '...') if len(prompt) > 30 else prompt\n",
                "    print(f' {os_type:<10} | {prompt_trunc:<30} | {status_sym:^5} | {arg_sym:^5} | {pred_cmd}')\n",
                "    if not is_match:\n",
                "        print(f' {\"\":<10} | {\"  -> Expected:\":<30} | {\"\":^5} | {\"\":^5} | {ground_truth}')\n",
                "\n",
                "accuracy = (token_correct_sum / max(1, token_total_sum)) * 100.0\n",
                "seq_acc = (exact_matches / total_eval) * 100.0\n",
                "norm_acc = (norm_matches / total_eval) * 100.0\n",
                "arg_acc = (arg_preserved / total_eval) * 100.0\n",
                "print('=' * 95)\n",
                "print(f' Exact Sequence Matches  : {exact_matches}/{total_eval} ({seq_acc:.1f}%)')\n",
                "print(f' Normalized Exact Matches: {norm_matches}/{total_eval} ({norm_acc:.1f}%)')\n",
                "print(f' Argument Preservation   : {arg_preserved}/{total_eval} ({arg_acc:.1f}%)')\n",
                "print(f' Token-Level Accuracy    : {accuracy:.2f}%')\n",
                "print('=' * 95)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 6: Save / Download Trained Model Checkpoint\n",
                "Download `checkpoints/terminal_model_final.pt` directly or copy to Google Drive."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Download or save checkpoint to Google Drive\n",
                "import os\n",
                "from google.colab import files\n",
                "\n",
                "ckpt_path = 'checkpoints/terminal_model_final.pt'\n",
                "if os.path.exists(ckpt_path):\n",
                "    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)\n",
                "    print(f'Checkpoint verified: {ckpt_path} ({size_mb:.2f} MB)')\n",
                "    \n",
                "    # Option A: Download directly to your local computer\n",
                "    print('Triggering browser download...')\n",
                "    files.download(ckpt_path)\n",
                "    \n",
                "    # Option B: Uncomment below to save to Google Drive\n",
                "    # from google.colab import drive\n",
                "    # drive.mount('/content/drive')\n",
                "    # !cp checkpoints/terminal_model_final.pt /content/drive/MyDrive/\n",
                "else:\n",
                "    print(f'Error: {ckpt_path} does not exist. Check training output.')"
            ]
        }
    ]
}

dest = Path("colab_train.ipynb")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created {dest.resolve()} ({dest.stat().st_size} bytes)")
