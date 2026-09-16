"""
CommandLLM Production Inference Server (FastAPI)
=================================================
Serves domain-specific natural language to Linux/Bash and Windows/PowerShell
command generation using the custom CoreCommandLLM decoder Transformer.
"""

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from core.config import CommandLMConfig
from core.model import CoreCommandLLM
from core.tokenizer import CommandTokenizer


# ==============================================================================
# 1. Pydantic Request and Response Schemas
# ==============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Natural language user instruction or query")
    os: Literal["linux", "powershell"] = Field(..., description="Target operating system")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(40, ge=1, le=200, description="Top-K logit filtering parameter")


class GenerateResponse(BaseModel):
    os: str
    prompt: str
    command: str
    latency_ms: float
    raw_tokens: List[int]


class CompareRequest(BaseModel):
    prompt: str = Field(..., description="Natural language user instruction or query")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(40, ge=1, le=200, description="Top-K logit filtering parameter")


class CompareResponse(BaseModel):
    prompt: str
    linux: str
    powershell: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    total_parameters: int
    context_window: int
    vocab_size: int


# ==============================================================================
# 2. Inference Core Utility
# ==============================================================================

def run_inference(
    model: CoreCommandLLM,
    tokenizer: CommandTokenizer,
    prompt: str,
    target_os: str,
    temperature: float = 0.2,
    top_k: Optional[int] = 40,
    max_new_tokens: int = 64,
) -> tuple[str, float, List[int]]:
    """
    Executes autoregressive command generation on CPU.

    Returns:
        tuple of (cleaned_command_string, latency_in_ms, raw_generated_token_ids)
    """
    # 1. Construct canonical conditioning prefix
    prefix_str = tokenizer.format_prompt(os_type=target_os, prompt_text=prompt)
    input_tokens = tokenizer.encode(prefix_str)
    x = torch.tensor([input_tokens], dtype=torch.long, device=torch.device("cpu"))

    # 2. Measure inference wall-clock latency
    t0 = time.perf_counter()

    with torch.no_grad():
        output_tensor = model.generate(
            idx=x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Extract generated token sequence (omit prompt conditioning tokens)
    full_seq = output_tensor[0].tolist()
    gen_tokens = full_seq[len(input_tokens):]

    # 4. Cut off at <|end|> token ID if present
    cmd_tokens = []
    for token_id in gen_tokens:
        if token_id == tokenizer.eos_token_id:
            break
        cmd_tokens.append(token_id)

    # 5. Decode tokens into raw string and clean delimiters
    raw_decoded = tokenizer.decode(cmd_tokens)

    # Clean any residual delimiter strings or padding
    clean_command = raw_decoded.replace("<|end|>", "").replace("<|pad|>", "").strip()

    return clean_command, latency_ms, gen_tokens


# ==============================================================================
# 3. Lifespan Context Manager
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Loads the fine-tuned CoreCommandLLM checkpoint and tokenizer into memory on startup.
    """
    checkpoint_path = WORKSPACE_ROOT / "checkpoints" / "terminal_model_final.pt"
    if not checkpoint_path.is_file():
        # Fallback check for custom base checkpoint if final is missing
        base_checkpoint = WORKSPACE_ROOT / "checkpoints" / "custom_base_checkpoint.pt"
        if base_checkpoint.is_file():
            checkpoint_path = base_checkpoint
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path} or {base_checkpoint}."
            )

    print("=" * 72)
    print(" CommandLLM Inference Server Initializing...")
    print(f" Loading checkpoint: {checkpoint_path}")

    config = CommandLMConfig(
        vocab_size=50263,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    model = CoreCommandLLM(config)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    # Ensure weight tying between embedding and output projection
    model.transformer.wte.weight = model.lm_head.weight
    model.eval()

    tokenizer = CommandTokenizer()

    # Cache loaded instances in app.state for request handlers
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.config = config
    app.state.checkpoint_name = checkpoint_path.name

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model ready on CPU ({total_params:,} parameters).")
    print("=" * 72)

    yield

    print("CommandLLM Inference Server shutting down...")


# ==============================================================================
# 4. FastAPI Application Setup & Middleware
# ==============================================================================

app = FastAPI(
    title="CommandLLM Inference API",
    description="Domain-specific natural language to Linux/Bash and PowerShell command generator.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for local development and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 5. API Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Service health and model metadata inspection."""
    model: CoreCommandLLM = app.state.model
    config: CommandLMConfig = app.state.config
    total_params = sum(p.numel() for p in model.parameters())
    return HealthResponse(
        status="healthy",
        model=app.state.checkpoint_name,
        device="cpu",
        total_parameters=total_params,
        context_window=config.block_size,
        vocab_size=config.vocab_size,
    )


@app.post("/api/generate", response_model=GenerateResponse)
@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    """
    Translates a natural language instruction into an executable shell command
    for the specified operating system.
    """
    model: CoreCommandLLM = app.state.model
    tokenizer: CommandTokenizer = app.state.tokenizer

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    command, latency_ms, raw_tokens = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        target_os=req.os,
        temperature=req.temperature if req.temperature is not None else 0.2,
        top_k=req.top_k if req.top_k is not None else 40,
        max_new_tokens=64,
    )

    return GenerateResponse(
        os=req.os,
        prompt=req.prompt,
        command=command,
        latency_ms=round(latency_ms, 2),
        raw_tokens=raw_tokens,
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare_endpoint(req: CompareRequest):
    """
    Executes command translation concurrently/sequentially for both Linux and PowerShell,
    enabling side-by-side comparison of generated commands.
    """
    model: CoreCommandLLM = app.state.model
    tokenizer: CommandTokenizer = app.state.tokenizer

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    t_start = time.perf_counter()

    # Generate Linux command
    cmd_linux, _, _ = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        target_os="linux",
        temperature=req.temperature if req.temperature is not None else 0.2,
        top_k=req.top_k if req.top_k is not None else 40,
        max_new_tokens=64,
    )

    # Generate PowerShell command
    cmd_ps, _, _ = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        target_os="powershell",
        temperature=req.temperature if req.temperature is not None else 0.2,
        top_k=req.top_k if req.top_k is not None else 40,
        max_new_tokens=64,
    )

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    return CompareResponse(
        prompt=req.prompt,
        linux=cmd_linux,
        powershell=cmd_ps,
        latency_ms=round(total_latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=False)
