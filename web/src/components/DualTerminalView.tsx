import React from 'react';
import { TerminalWindow } from './TerminalWindow';
import { CompareResponse, GenerateResponse } from '../api/client';
import { Columns, Clock } from 'lucide-react';

interface DualTerminalViewProps {
  compareResult: CompareResponse | null;
  isLoading: boolean;
  promptValue: string;
  setPromptValue: (val: string) => void;
  onExecute: (prompt: string) => void;
}

export const DualTerminalView: React.FC<DualTerminalViewProps> = ({
  compareResult,
  isLoading,
  promptValue,
  setPromptValue,
  onExecute,
}) => {
  // Map compareResult to individual GenerateResponse structures for the terminal components
  const linuxResult: GenerateResponse | null = compareResult
    ? {
        os: 'linux',
        prompt: compareResult.prompt,
        command: compareResult.linux,
        latency_ms: compareResult.latency_ms / 2, // Approximate split
        raw_tokens: [],
      }
    : null;

  const psResult: GenerateResponse | null = compareResult
    ? {
        os: 'powershell',
        prompt: compareResult.prompt,
        command: compareResult.powershell,
        latency_ms: compareResult.latency_ms / 2,
        raw_tokens: [],
      }
    : null;

  return (
    <div className="space-y-4">
      {/* Dual Header Indicator */}
      <div className="flex items-center justify-between px-2 text-xs font-mono text-slate-400">
        <div className="flex items-center gap-2">
          <Columns className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-200 font-medium">Dual-Mode Shell Comparison</span>
          <span className="text-slate-500">• Evaluating simultaneous Bash & PowerShell mappings</span>
        </div>
        {compareResult && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-cyan-300">
            <Clock className="w-3 h-3 text-cyan-400" />
            <span>Total Dual Latency: {compareResult.latency_ms.toFixed(1)} ms</span>
          </div>
        )}
      </div>

      {/* Side-by-side Terminal Panes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Linux Terminal */}
        <TerminalWindow
          os="linux"
          title="GNU/Linux Bash Comparison Pane"
          result={linuxResult}
          isLoading={isLoading}
          onExecute={onExecute}
          promptValue={promptValue}
          setPromptValue={setPromptValue}
          showInput={false}
        />

        {/* PowerShell Terminal */}
        <TerminalWindow
          os="powershell"
          title="Windows PowerShell Comparison Pane"
          result={psResult}
          isLoading={isLoading}
          onExecute={onExecute}
          promptValue={promptValue}
          setPromptValue={setPromptValue}
          showInput={false}
        />
      </div>

      {/* Shared Unified Command Prompt Input for Dual Mode */}
      <div className="p-4 rounded-xl bg-[#0b0e14] border border-slate-800/80 shadow-lg">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono font-bold text-cyan-400 shrink-0">
            dual-eval &gt;
          </span>
          <input
            type="text"
            value={promptValue}
            onChange={(e) => setPromptValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !isLoading && promptValue.trim()) {
                onExecute(promptValue);
              }
            }}
            placeholder="Type query to evaluate Linux vs PowerShell side-by-side (e.g. kill port 8080)"
            disabled={isLoading}
            className="flex-1 bg-transparent border-none outline-none text-slate-100 placeholder:text-slate-600 focus:ring-0 text-sm font-mono"
            autoFocus
          />
          <button
            onClick={() => onExecute(promptValue)}
            disabled={isLoading || !promptValue.trim()}
            className="px-4 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-mono font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {isLoading ? 'Translating...' : 'Compare Both'}
          </button>
        </div>
      </div>
    </div>
  );
};
