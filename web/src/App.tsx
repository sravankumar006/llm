import React, { useState, useEffect } from 'react';
import { TelemetryBar } from './components/TelemetryBar';
import { TerminalWindow } from './components/TerminalWindow';
import { DualTerminalView } from './components/DualTerminalView';
import { ModelInspectorModal } from './components/ModelInspectorModal';
import { 
  fetchHealth, 
  generateCommand, 
  compareCommands, 
  HealthResponse, 
  GenerateResponse, 
  CompareResponse 
} from './api/client';
import { Columns, Sparkles, Sliders, AlertTriangle } from 'lucide-react';

const QUICK_PROMPTS = [
  "kill process listening on port 8080",
  "find all files larger than 100MB",
  "count total commits in git repository",
  "find recursive TODO in current directory",
  "show top 10 memory consuming processes",
  "check if port 443 is open on google.com",
];

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  // Shell & View States
  const [activeOs, setActiveOs] = useState<'linux' | 'powershell'>('linux');
  const [isDualMode, setIsDualMode] = useState<boolean>(false);
  const [promptValue, setPromptValue] = useState<string>('');
  
  // Results & Loading States
  const [singleResult, setSingleResult] = useState<GenerateResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Model Inspector Modal
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(false);

  // Temperature & Top-K Sampling parameters
  const [temperature, setTemperature] = useState<number>(0.2);
  const [topK, setTopK] = useState<number>(40);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  // Initial and periodic health check
  useEffect(() => {
    let isMounted = true;

    const checkHealth = async () => {
      try {
        const data = await fetchHealth();
        if (isMounted) {
          setHealth(data);
          setIsConnected(true);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          setIsConnected(false);
          console.warn('Backend server unreachable:', err);
        }
      } finally {
        if (isMounted) {
          setIsLoadingHealth(false);
        }
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleExecute = async (queryText: string) => {
    const query = queryText.trim();
    if (!query || isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      if (isDualMode) {
        const res = await compareCommands({
          prompt: query,
          temperature,
          top_k: topK,
        });
        setCompareResult(res);
      } else {
        const res = await generateCommand({
          prompt: query,
          os: activeOs,
          temperature,
          top_k: topK,
        });
        setSingleResult(res);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to communicate with CommandLLM inference backend.');
    } finally {
      setIsLoading(false);
    }
  };

  const selectQuickPrompt = (q: string) => {
    setPromptValue(q);
    handleExecute(q);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#07090e] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* 1. System Telemetry Bar */}
      <TelemetryBar
        health={health}
        isConnected={isConnected}
        isLoadingHealth={isLoadingHealth}
        onOpenInspector={() => setIsInspectorOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-6 flex flex-col gap-6">
        {/* Error Alert */}
        {error && (
          <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-mono animate-in fade-in duration-200">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span className="flex-1">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="text-xs text-rose-400 hover:text-rose-200 uppercase font-semibold"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Controls Ribbon: OS Selector, Dual Comparison Switch, Sampler Settings */}
        <div className="flex flex-wrap items-center justify-between gap-4 p-2 rounded-xl bg-[#0b0e14] border border-slate-800/80">
          {/* OS Switcher Pills */}
          <div className="flex items-center gap-1.5 bg-[#07090e] p-1 rounded-lg border border-slate-800/60">
            <button
              onClick={() => {
                setActiveOs('linux');
                if (isDualMode) setIsDualMode(false);
              }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md font-mono text-xs font-medium transition-all ${
                !isDualMode && activeOs === 'linux'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <span className="text-emerald-400 font-bold">$</span>
              <span>Linux (Bash)</span>
            </button>

            <button
              onClick={() => {
                setActiveOs('powershell');
                if (isDualMode) setIsDualMode(false);
              }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md font-mono text-xs font-medium transition-all ${
                !isDualMode && activeOs === 'powershell'
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40 shadow-sm shadow-blue-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <span className="text-blue-400 font-bold">&gt;_</span>
              <span>PowerShell</span>
            </button>
          </div>

          {/* Dual-Mode Toggle & Hyperparameter Settings */}
          <div className="flex items-center gap-3">
            {/* Dual Comparison Mode Switch */}
            <button
              onClick={() => setIsDualMode(!isDualMode)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-mono text-xs font-medium border transition-all ${
                isDualMode
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-sm shadow-cyan-500/15'
                  : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Columns className={`w-3.5 h-3.5 ${isDualMode ? 'text-cyan-400' : 'text-slate-500'}`} />
              <span>Dual Comparison Mode</span>
              <span className={`w-1.5 h-1.5 rounded-full ${isDualMode ? 'bg-cyan-400 animate-pulse' : 'bg-slate-600'}`} />
            </button>

            {/* Inference Tuning Toggle */}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-1.5 rounded-lg border text-xs font-mono transition-colors ${
                showSettings 
                  ? 'bg-slate-800 border-slate-700 text-cyan-400' 
                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle generation parameters"
            >
              <Sliders className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Hyperparameter Settings Drawer (Collapsible) */}
        {showSettings && (
          <div className="p-4 rounded-xl bg-[#0b0e14] border border-slate-800/80 grid grid-cols-1 sm:grid-cols-2 gap-4 font-mono text-xs animate-in slide-in-from-top-2 duration-150">
            <div>
              <div className="flex justify-between mb-1.5 text-slate-400">
                <span>Sampling Temperature:</span>
                <span className="text-cyan-400 font-semibold">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">
                0.0 = deterministic greedy argmax; 0.2 = optimal syntax accuracy.
              </span>
            </div>

            <div>
              <div className="flex justify-between mb-1.5 text-slate-400">
                <span>Top-K Filtering:</span>
                <span className="text-cyan-400 font-semibold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">
                Restricts next-token logits to top K highest probability candidates.
              </span>
            </div>
          </div>
        )}

        {/* 2. Interactive Terminal Workspace */}
        <section className="flex-1 flex flex-col justify-center">
          {isDualMode ? (
            <DualTerminalView
              compareResult={compareResult}
              isLoading={isLoading}
              promptValue={promptValue}
              setPromptValue={setPromptValue}
              onExecute={handleExecute}
            />
          ) : (
            <TerminalWindow
              os={activeOs}
              result={singleResult}
              isLoading={isLoading}
              onExecute={handleExecute}
              promptValue={promptValue}
              setPromptValue={setPromptValue}
              showInput={true}
            />
          )}
        </section>

        {/* 3. Quick-Prompt Recommendation Chips */}
        <section className="space-y-2.5">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-medium text-slate-300">Quick Prompt Benchmark Suite:</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {QUICK_PROMPTS.map((q, idx) => (
              <button
                key={idx}
                onClick={() => selectQuickPrompt(q)}
                disabled={isLoading}
                className="px-3 py-1.5 rounded-lg bg-slate-900/70 hover:bg-slate-800 text-slate-300 hover:text-cyan-300 border border-slate-800/80 hover:border-cyan-500/40 text-xs font-mono transition-all hover:scale-101 active:scale-98 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                "{q}"
              </button>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-slate-900 bg-[#07090e] px-4 py-4 text-center text-xs font-mono text-slate-600">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <span>CommandLLM 124M • PyTorch Autoregressive Transformer Architecture</span>
          <span className="text-slate-500">Connected to FastAPI CPU Backend (127.0.0.1:8000)</span>
        </div>
      </footer>

      {/* Model Architecture Inspector Modal */}
      <ModelInspectorModal
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        health={health}
      />
    </div>
  );
};

export default App;
