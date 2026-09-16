import React from 'react';
import { X, Cpu, Layers, Link2, ShieldCheck, Hash, Terminal } from 'lucide-react';
import { HealthResponse } from '../api/client';

interface ModelInspectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthResponse | null;
}

export const ModelInspectorModal: React.FC<ModelInspectorModalProps> = ({
  isOpen,
  onClose,
  health,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-[#0d121c] border border-slate-700/60 rounded-xl shadow-2xl terminal-glow text-slate-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 bg-[#0a0e17]/90 backdrop-blur-md border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2 font-mono">
                CoreCommandLLM Architecture
              </h2>
              <p className="text-xs text-slate-400">
                Ground-Up Custom PyTorch Decoder Transformer
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            title="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-6 text-sm">
          {/* Live Telemetry Card */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Parameters</span>
              <span className="font-mono text-cyan-300 font-semibold text-base">
                {health?.total_parameters ? `${(health.total_parameters / 1e6).toFixed(1)}M` : '123.9M'}
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Context Window</span>
              <span className="font-mono text-emerald-300 font-semibold text-base">
                {health?.context_window || 256} tokens
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Vocab Size</span>
              <span className="font-mono text-indigo-300 font-semibold text-base">
                {health?.vocab_size ? health.vocab_size.toLocaleString() : '50,263'}
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Layers & Heads</span>
              <span className="font-mono text-slate-200 font-medium text-sm">
                12 Layers / 12 Heads
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Embedding Dim</span>
              <span className="font-mono text-slate-200 font-medium text-sm">
                768 (Hidden)
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Compute Target</span>
              <span className="font-mono text-amber-300 font-medium text-sm uppercase">
                {health?.device || 'CPU (Local)'}
              </span>
            </div>
          </div>

          {/* Deep Learning Architectural Highlights */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Key Architectural Innovations
            </h3>

            {/* Feature 1 */}
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-3">
              <div className="p-2 rounded-md bg-blue-500/10 text-blue-400 mt-0.5">
                <Link2 className="w-4 h-4" />
              </div>
              <div className="space-y-1">
                <h4 className="font-medium text-slate-200">
                  Weight Tying (<code className="font-mono text-xs text-blue-300">wte ↔ lm_head</code>)
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Input token embedding weights are directly shared with the final language modeling projection head. This eliminates ~38.6 million redundant parameters, decreases RAM consumption on CPU, and enforces dense semantic representation symmetry.
                </p>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-3">
              <div className="p-2 rounded-md bg-emerald-500/10 text-emerald-400 mt-0.5">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div className="space-y-1">
                <h4 className="font-medium text-slate-200">
                  Selective Loss Masking (<code className="font-mono text-xs text-emerald-300">ignore_index = -100</code>)
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  During training, all conditioning tokens (prefix, OS tag, user prompt) are masked with -100. Backpropagation only penalizes incorrect prediction of the supervised shell command tokens and the terminal <code className="font-mono text-xs text-cyan-300">&lt;|end|&gt;</code> marker.
                </p>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-3">
              <div className="p-2 rounded-md bg-amber-500/10 text-amber-400 mt-0.5">
                <Layers className="w-4 h-4" />
              </div>
              <div className="space-y-1">
                <h4 className="font-medium text-slate-200">
                  Causal Shift ($t \to t+1$)
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Standard autoregressive cross-entropy aligns <code className="font-mono text-xs text-amber-300">shift_logits = logits[..., :-1, :]</code> against <code className="font-mono text-xs text-amber-300">shift_targets = targets[..., 1:]</code>, ensuring token representation at step <code className="font-mono text-xs">t</code> models the probability distribution of the subsequent token.
                </p>
              </div>
            </div>

            {/* Feature 4: Custom Special Tokens */}
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-3">
              <div className="p-2 rounded-md bg-cyan-500/10 text-cyan-400 mt-0.5">
                <Hash className="w-4 h-4" />
              </div>
              <div className="space-y-2 w-full">
                <h4 className="font-medium text-slate-200">
                  Domain Delimiter Tokens
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs">
                  <div className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-cyan-400">&lt;|start|&gt;</span>
                    <span className="text-slate-500">50257</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-cyan-400">&lt;|os|&gt;</span>
                    <span className="text-slate-500">50258</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-cyan-400">&lt;|prompt|&gt;</span>
                    <span className="text-slate-500">50259</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-cyan-400">&lt;|cmd|&gt;</span>
                    <span className="text-slate-500">50260</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-cyan-400">&lt;|end|&gt;</span>
                    <span className="text-slate-500">50261</span>
                  </div>
                  <div className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-cyan-400">&lt;|pad|&gt;</span>
                    <span className="text-slate-500">50262</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 bg-[#0a0e17] border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            Model Checkpoint: <span className="font-mono text-slate-300">{health?.model || 'terminal_model_final.pt'}</span>
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
