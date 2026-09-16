import React, { useState, useEffect, useRef } from 'react';
import { Copy, Check, Terminal as TerminalIcon, Clock, Sparkles, CornerDownLeft } from 'lucide-react';
import { GenerateResponse } from '../api/client';

interface TerminalWindowProps {
  os: 'linux' | 'powershell';
  title?: string;
  result: GenerateResponse | null;
  isLoading: boolean;
  onExecute: (prompt: string) => void;
  promptValue: string;
  setPromptValue: (val: string) => void;
  showInput?: boolean;
}

export const TerminalWindow: React.FC<TerminalWindowProps> = ({
  os,
  title,
  result,
  isLoading,
  onExecute,
  promptValue,
  setPromptValue,
  showInput = true,
}) => {
  const [copied, setCopied] = useState(false);
  const [displayedText, setDisplayedText] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const isPowerShell = os === 'powershell';
  const promptPrefix = isPowerShell ? 'PS C:\\CommandLLM>' : 'user@commandllm:~$';
  const displayTitle = title || (isPowerShell ? 'Windows PowerShell (CoreCommandLLM)' : 'GNU/Linux Bash (CoreCommandLLM)');

  // Typewriter effect when new result arrives
  useEffect(() => {
    if (!result?.command) {
      setDisplayedText('');
      return;
    }

    let current = 0;
    const fullText = result.command;
    setDisplayedText('');

    const interval = setInterval(() => {
      current++;
      setDisplayedText(fullText.slice(0, current));
      if (current >= fullText.length) {
        clearInterval(interval);
      }
    }, 12);

    return () => clearInterval(interval);
  }, [result?.command]);

  const handleCopy = () => {
    if (!result?.command) return;
    navigator.clipboard.writeText(result.command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isLoading && promptValue.trim()) {
      onExecute(promptValue);
    }
  };

  return (
    <div className="flex flex-col w-full rounded-xl bg-[#0b0e14] border border-slate-800/80 shadow-2xl overflow-hidden terminal-glow transition-all">
      {/* Window Title Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#080b10] border-b border-slate-800/80 select-none">
        {/* macOS / Ubuntu traffic lights */}
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-rose-500/80 hover:bg-rose-500 transition-colors" />
          <div className="w-3 h-3 rounded-full bg-amber-500/80 hover:bg-amber-500 transition-colors" />
          <div className="w-3 h-3 rounded-full bg-emerald-500/80 hover:bg-emerald-500 transition-colors" />
          <span className="ml-2 text-xs font-mono font-medium text-slate-400 flex items-center gap-1.5">
            <TerminalIcon className={`w-3.5 h-3.5 ${isPowerShell ? 'text-blue-400' : 'text-emerald-400'}`} />
            {displayTitle}
          </span>
        </div>

        {/* Latency & Telemetry Metric Badge */}
        {result?.latency_ms !== undefined && (
          <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-cyan-300">
              <Clock className="w-3 h-3 text-cyan-400" />
              {result.latency_ms.toFixed(1)} ms
            </span>
            {result.raw_tokens && (
              <span className="hidden sm:inline px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                {result.raw_tokens.length} tokens
              </span>
            )}
          </div>
        )}
      </div>

      {/* Terminal Screen Body */}
      <div className="relative p-5 font-mono text-sm space-y-4 min-h-[220px] flex flex-col justify-between overflow-x-auto">
        {/* CRT Scanline Overlay */}
        <div className="absolute inset-0 terminal-crt-overlay opacity-30" />

        <div className="relative z-10 space-y-4">
          {/* Welcome Message or Prompt History */}
          <div className="text-xs text-slate-500 leading-relaxed border-b border-slate-800/60 pb-3">
            <span>CoreCommandLLM v1.0 [124M Transformer Engine]</span>
            <br />
            <span>Target Shell: <span className={isPowerShell ? 'text-blue-400 font-semibold' : 'text-emerald-400 font-semibold'}>{isPowerShell ? 'PowerShell 7.x' : 'Bash / POSIX'}</span></span>
          </div>

          {/* Prompt line if executed */}
          {result && (
            <div className="space-y-1">
              <div className="flex items-start gap-2 text-slate-400 text-xs">
                <span className={isPowerShell ? 'text-blue-400' : 'text-emerald-400'}>
                  {promptPrefix}
                </span>
                <span className="text-slate-300 italic"># Query: "{result.prompt}"</span>
              </div>
            </div>
          )}

          {/* Generated Command Output Block */}
          {isLoading ? (
            <div className="flex items-center gap-3 py-4 text-cyan-400">
              <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs tracking-wider animate-pulse">
                Evaluating autoregressive next-token logits...
              </span>
            </div>
          ) : result ? (
            <div className="group relative mt-2 p-3.5 rounded-lg bg-[#06080d] border border-slate-800/90 text-slate-100 flex items-center justify-between gap-3 shadow-inner">
              <div className="flex items-center gap-2 overflow-x-auto py-1">
                <span className={isPowerShell ? 'text-blue-400 font-bold select-none' : 'text-emerald-400 font-bold select-none'}>
                  &gt;
                </span>
                <span className="font-semibold text-cyan-200 tracking-wide select-all text-sm sm:text-base">
                  {displayedText}
                </span>
                {displayedText.length < result.command.length && (
                  <span className="inline-block w-2 h-4 bg-cyan-400 animate-cursor-blink" />
                )}
              </div>

              {/* Copy Action Button */}
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700/60 transition-all shrink-0 active:scale-95"
                title="Copy command to clipboard"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-200" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="py-6 text-slate-600 text-xs flex flex-col items-center justify-center gap-2 text-center">
              <Sparkles className="w-5 h-5 text-slate-600 opacity-60" />
              <span>Enter a natural language instruction below to synthesize an executable command.</span>
            </div>
          )}
        </div>

        {/* Input Line (If Enabled) */}
        {showInput && (
          <div className="relative z-10 pt-2 border-t border-slate-800/60">
            <div className="flex items-center gap-2 text-xs sm:text-sm">
              <span className={`font-semibold shrink-0 select-none ${isPowerShell ? 'text-blue-400' : 'text-emerald-400'}`}>
                {promptPrefix}
              </span>
              <input
                ref={inputRef}
                type="text"
                value={promptValue}
                onChange={(e) => setPromptValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. kill process listening on port 8080"
                disabled={isLoading}
                className="flex-1 bg-transparent border-none outline-none text-slate-100 placeholder:text-slate-600 focus:ring-0 text-xs sm:text-sm font-mono py-1"
                autoFocus
              />
              <button
                onClick={() => onExecute(promptValue)}
                disabled={isLoading || !promptValue.trim()}
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-mono font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                title="Press Enter to generate"
              >
                <span>Run</span>
                <CornerDownLeft className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
