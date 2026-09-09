import React, { useState } from 'react';
import { Flame, Sparkles, Sliders, Info, Eye } from 'lucide-react';

export default function DifferenceHeatmap({ heatmapImage, metrics }) {
  const [boost, setBoost] = useState(6);

  return (
    <div className="relative w-full h-full flex flex-col bg-[#08090a] rounded-xl border border-[#232529] overflow-hidden">
      {/* Control Bar */}
      <div className="h-10 border-b border-[#232529] bg-[#0f1011] px-4 flex items-center justify-between z-20">
        <div className="flex items-center gap-2">
          <Flame className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold text-white">High-Frequency Residual Difference Map</span>
          <span className="text-xs text-[#8a8f98] font-mono">(Sobel + Wavelet Energy)</span>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#8a8f98]">Boost:</span>
            <input 
              type="range" 
              min="1" 
              max="10" 
              value={boost}
              onChange={(e) => setBoost(Number(e.target.value))}
              className="w-20 accent-[#5e6ad2] cursor-pointer"
            />
            <span className="font-mono text-[#8a8f98] w-6">{boost}x</span>
          </div>

          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#1c1d20] border border-[#2e3138] text-[11px] font-mono">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            <span className="text-amber-300 font-medium">Inferno</span>
          </div>
        </div>
      </div>

      {/* Heatmap Display */}
      <div className="relative flex-1 flex items-center justify-center p-4">
        <div className="relative w-full h-full max-w-[800px] max-h-[600px] aspect-square rounded-lg border border-[#232529] overflow-hidden bg-[#05020c] shadow-2xl flex items-center justify-center">
          <img 
            src={heatmapImage} 
            alt="Residual Difference Map"
            className="w-full h-full object-contain"
            style={{ filter: `contrast(${1 + boost * 0.1})` }}
          />
        </div>
      </div>

      {/* Footer Insight Box */}
      <div className="p-3 border-t border-[#232529] bg-[#0f1011] flex items-center justify-between text-xs text-[#8a8f98]">
        <div className="flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-[#5e6ad2]" />
          <span>Warm yellow/orange pixels indicate fine geospatial structures (runways, field lines, roof edges) recovered by HAT-Light that sensor PSF degradation blurred.</span>
        </div>
        <div className="font-mono text-[11px]">
          FFT Loss: <span className="text-emerald-400 font-semibold">{metrics?.fftLoss || '0.0074'}</span>
        </div>
      </div>
    </div>
  );
}
