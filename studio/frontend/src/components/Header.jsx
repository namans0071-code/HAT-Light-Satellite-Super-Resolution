import React from 'react';
import { 
  Layers, 
  Cpu, 
  Sparkles, 
  Zap, 
  CheckCircle2, 
  Sliders, 
  Share2,
  Terminal
} from 'lucide-react';

export default function Header({ backendStatus, onOpenExport, activeMode, setActiveMode }) {
  const isConnected = backendStatus?.connected;

  return (
    <header className="h-14 border-b border-[#232529] bg-[#0f1011]/90 backdrop-blur-md px-4 flex items-center justify-between z-30 sticky top-0">
      {/* Brand & Title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#5e6ad2] to-[#8b5cf6] flex items-center justify-center shadow-linear-glow">
          <Layers className="w-4 h-4 text-white" />
        </div>
        <div className="flex items-baseline gap-2.5">
          <span className="font-semibold text-sm tracking-tight text-white flex items-center gap-1.5">
            HAT-Light
            <span className="text-xs text-[#8a8f98] font-normal font-mono">v5.0</span>
          </span>
          <span className="text-[#3e424b] text-xs">/</span>
          <span className="text-xs text-[#8a8f98] hidden sm:inline-block">
            Satellite Super-Resolution Studio
          </span>
        </div>
        
        {/* Status Pill */}
        <div className="hidden md:flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-[#232529] bg-[#141517] text-[11px] font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-[#8a8f98]">Target:</span>
          <span className="text-emerald-400 font-medium">CPU Float32 (Zero-GPU)</span>
        </div>
      </div>

      {/* Center View Controls */}
      <div className="flex items-center p-0.5 rounded-lg border border-[#232529] bg-[#141517]">
        <button 
          onClick={() => setActiveMode('slider')}
          className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1.5 font-medium ${
            activeMode === 'slider' 
              ? 'bg-[#232529] text-white shadow-sm' 
              : 'text-[#8a8f98] hover:text-white'
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>Split Slider</span>
        </button>
        <button 
          onClick={() => setActiveMode('sidebyside')}
          className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1.5 font-medium ${
            activeMode === 'sidebyside' 
              ? 'bg-[#232529] text-white shadow-sm' 
              : 'text-[#8a8f98] hover:text-white'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Side-by-Side</span>
        </button>
        <button 
          onClick={() => setActiveMode('diff')}
          className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1.5 font-medium ${
            activeMode === 'diff' 
              ? 'bg-[#232529] text-white shadow-sm' 
              : 'text-[#8a8f98] hover:text-white'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>Difference Map</span>
        </button>
      </div>

      {/* Right Action Tools */}
      <div className="flex items-center gap-2">
        <button
          onClick={onOpenExport}
          className="px-3 py-1.5 rounded-lg bg-[#5e6ad2] hover:bg-[#6e7be0] text-white text-xs font-medium transition-all flex items-center gap-1.5 shadow-sm hover:shadow-linear-glow"
        >
          <Share2 className="w-3.5 h-3.5" />
          <span>Export SR</span>
          <kbd className="hidden lg:inline-block ml-1 px-1 py-0.2 text-[9px] bg-white/20 rounded font-mono">⌘E</kbd>
        </button>
      </div>
    </header>
  );
}
