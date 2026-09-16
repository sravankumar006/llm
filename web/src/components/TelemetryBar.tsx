import React from 'react';
import { Terminal, Activity, Info, ShieldCheck, AlertCircle } from 'lucide-react';
import { HealthResponse } from '../api/client';

interface TelemetryBarProps {
  health: HealthResponse | null;
  isConnected: boolean;
  isLoadingHealth: boolean;
  onOpenInspector: () => void;
}

export const TelemetryBar: React.FC<TelemetryBarProps> = ({
  health,
  isConnected,
  isLoadingHealth,
  onOpenInspector,
}) => {
  return (
    <header className="sticky top-0 z-30 w-full border-b border-slate-800/80 bg-[#07090e]/85 backdrop-blur-md px-4 lg:px-8 py-3 transition-colors">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
        {/* Brand & Connection State */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2.5">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shadow-sm shadow-cyan-500/10">
              <Terminal className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono font-bold tracking-tight text-slate-100 text-base">
                  Command<span className="text-cyan-400">LLM</span>
                </span>
                <span className="text-[10px] uppercase font-semibold font-mono tracking-wider px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/60">
                  v1.0-CPU
                </span>
              </div>
            </div>
          </div>

          <div className="h-4 w-px bg-slate-800 hidden sm:block" />

          {/* Pulsing Status Dot */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900/90 border border-slate-800/80 text-xs font-mono">
            {isLoadingHealth ? (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                <span className="text-amber-300">Connecting...</span>
              </>
            ) : isConnected ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-medium flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3" />
                  Online
                </span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                <span className="text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  Backend Offline
                </span>
              </>
            )}
          </div>
        </div>

        {/* Telemetry Pills & Inspector CTA */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="hidden md:flex items-center gap-2 font-mono text-xs">
            <div className="px-2.5 py-1 rounded-md bg-slate-900/60 border border-slate-800 text-slate-300">
              <span className="text-slate-500 mr-1.5">Model:</span>
              <span className="text-cyan-400 font-medium">124M Params</span>
            </div>
            <div className="px-2.5 py-1 rounded-md bg-slate-900/60 border border-slate-800 text-slate-300">
              <span className="text-slate-500 mr-1.5">Context:</span>
              <span className="text-emerald-400 font-medium">
                {health?.context_window || 256}
              </span>
            </div>
            <div className="px-2.5 py-1 rounded-md bg-slate-900/60 border border-slate-800 text-slate-300">
              <span className="text-slate-500 mr-1.5">Device:</span>
              <span className="text-amber-400 font-medium uppercase">
                {health?.device || 'CPU'}
              </span>
            </div>
            <div className="px-2.5 py-1 rounded-md bg-slate-900/60 border border-slate-800 text-slate-300">
              <span className="text-slate-500 mr-1.5">Vocab:</span>
              <span className="text-indigo-400 font-medium">
                {health?.vocab_size ? health.vocab_size.toLocaleString() : '50,263'}
              </span>
            </div>
          </div>

          {/* Model Inspector Button */}
          <button
            onClick={onOpenInspector}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-xs font-mono font-medium transition-all hover:scale-102 active:scale-98 shadow-sm hover:shadow-cyan-500/10"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Architecture Specs</span>
            <Info className="w-3 h-3 text-cyan-400/70" />
          </button>
        </div>
      </div>
    </header>
  );
};
