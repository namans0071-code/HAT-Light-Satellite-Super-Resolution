import React from 'react';
import { Satellite, Cpu, Activity, Zap } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, systemStatus }) {
  const isTraining = systemStatus.trainer_state === 'TRAINING';
  const isPaused = systemStatus.trainer_state === 'PAUSED';

  return (
    <header className="sticky top-0 z-50 bg-[#08090a]/90 backdrop-blur-md border-b border-white/[0.08] px-5 py-2.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-7 h-7 rounded-md bg-[#191b22] border border-white/[0.1] flex items-center justify-center text-[#f7f8f8]">
            <Satellite className="w-3.5 h-3.5 text-[#f7f8f8]" />
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm font-semibold tracking-tight text-[#f7f8f8]">
              HAT-Light SatSR
            </span>
            <span className="text-[11px] text-[#62666d] font-normal">/</span>
            <span className="text-xs text-[#8a8f98] font-medium">
              Hybrid Attention Transformer (4-Band)
            </span>
          </div>
        </div>

        {/* Linear Segmented Navigation */}
        <div className="flex items-center p-0.5 bg-[#14151a] rounded-lg border border-white/[0.08]">
          <button
            onClick={() => setActiveTab('training')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeTab === 'training'
                ? 'bg-[#1f2128] text-[#f7f8f8] shadow-sm border border-white/[0.08]'
                : 'text-[#8a8f98] hover:text-[#f7f8f8]'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Training</span>
            {isTraining && (
              <span className="w-1.5 h-1.5 rounded-full bg-[#4ebb78] animate-pulse ml-0.5" />
            )}
          </button>

          <button
            onClick={() => setActiveTab('inference')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeTab === 'inference'
                ? 'bg-[#1f2128] text-[#f7f8f8] shadow-sm border border-white/[0.08]'
                : 'text-[#8a8f98] hover:text-[#f7f8f8]'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Inference & Demo</span>
          </button>
        </div>

        {/* Hardware & Status Badge */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2 px-2.5 py-1 rounded-md bg-[#14151a] border border-white/[0.08] text-xs">
            <span className="relative flex h-1.5 w-1.5">
              <span className={`inline-flex rounded-full h-1.5 w-1.5 ${
                isTraining ? 'bg-[#4ebb78] animate-ping' :
                isPaused ? 'bg-[#f2994a]' : 'bg-[#4ebb78]'
              }`} />
            </span>
            <span className="text-[#8a8f98] font-mono text-[11px]">
              {systemStatus.gpu_name ? systemStatus.gpu_name.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '') : 'CUDA'}
            </span>
            <span className="text-[#62666d] text-[10px]">|</span>
            <span className="text-[#62666d] font-mono text-[10px]">
              {systemStatus.gpu_memory_used_gb} / {systemStatus.gpu_memory_total_gb} GB
            </span>
          </div>
        </div>

      </div>
    </header>
  );
}
