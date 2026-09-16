/**
 * CommandLLM API Client
 * Typed fetch client communicating with the FastAPI inference backend.
 */

export interface GenerateRequest {
  prompt: string;
  os: "linux" | "powershell";
  temperature?: number;
  top_k?: number;
}

export interface GenerateResponse {
  os: string;
  prompt: string;
  command: string;
  latency_ms: number;
  raw_tokens: number[];
}

export interface CompareRequest {
  prompt: string;
  temperature?: number;
  top_k?: number;
}

export interface CompareResponse {
  prompt: string;
  linux: string;
  powershell: string;
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model: string;
  device: string;
  total_parameters: number;
  context_window: number;
  vocab_size: number;
}

const API_BASE = "";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, {
    headers: {
      "Accept": "application/json",
    },
  });

  if (!res.ok) {
    throw new Error(`Health check failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function generateCommand(req: GenerateRequest): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify({
      prompt: req.prompt,
      os: req.os,
      temperature: req.temperature ?? 0.2,
      top_k: req.top_k ?? 40,
    }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Generation request failed with status ${res.status}`);
  }

  return res.json();
}

export async function compareCommands(req: CompareRequest): Promise<CompareResponse> {
  const res = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify({
      prompt: req.prompt,
      temperature: req.temperature ?? 0.2,
      top_k: req.top_k ?? 40,
    }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Comparison request failed with status ${res.status}`);
  }

  return res.json();
}
