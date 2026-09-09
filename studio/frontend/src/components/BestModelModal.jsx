import React, { useState } from 'react';
import { X, Award, ExternalLink, Download, Layers, TrendingUp, Sparkles } from 'lucide-react';

export default function BestModelModal({ data, onClose, onNavigateInference }) {
  if (!data) return null;

  const [activeView, setActiveView] = useState('eval'); // 'eval' or 'curves'
  const [imgLoaded, setImgLoaded] = useState(false);

  const evalUrl = data.eval_image_url || `/api/artifacts/eval/latest?t=${Date.now()}`;
  const curvesUrl = data.curves_image_url || `/api/artifacts/curves/latest?t=${Date.now()}`;
  const currentImgUrl = activeView === 'eval' ? evalUrl : curvesUrl;

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = currentImgUrl;
    link.download = activeView === 'eval' 
      ? `best_model_eval_epoch_${data.epoch}.png`
      : `training_curves_epoch_${data.epoch}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-3 sm:p-5 select-none animate-in fade-in duration-200">
      <div 
        className="bg-[#0f1013] border border-white/[0.12] rounded-2xl w-full max-w-5xl max-h-[92vh] flex flex-col shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.08] bg-[#14151a]">
          <div className="flex items-center space-x-2.5">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              data.is_best 
                ? 'bg-[#4ebb78]/15 border border-[#4ebb78]/30 text-[#4ebb78]' 
                : 'bg-[#5e6ad2]/15 border border-[#5e6ad2]/30 text-[#5e6ad2]'
            }`}>
              {data.is_best ? <Award className="w-4 h-4" /> : <Layers className="w-4 h-4" />}
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-sm font-bold text-[#f7f8f8] tracking-tight">
                  {data.is_best ? '⭐ New Best Model Achieved!' : `Epoch ${data.epoch} Evaluation Report`}
                </h2>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold border ${
                  data.is_best 
                    ? 'bg-[#4ebb78]/20 text-[#4ebb78] border-[#4ebb78]/30' 
                    : 'bg-white/[0.08] text-[#8a8f98] border-white/[0.1]'
                }`}>
                  {data.total_epochs ? `Epoch ${data.epoch} / ${data.total_epochs}` : `Epoch ${data.epoch}`}
                </span>
                {data.psnr_gain != null && (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-[#5e6ad2]/20 text-[#5e6ad2] border border-[#5e6ad2]/30 font-medium">
                    +{data.psnr_gain} dB vs Bicubic
                  </span>
                )}
              </div>
              <p className="text-[11px] text-[#8a8f98]">
                HAT-Sat-Pro Super-Resolution Multi-Patch Report & Metrics
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownload}
              className="p-1.5 rounded-lg bg-[#191b22] border border-white/[0.08] text-[#8a8f98] hover:text-[#f7f8f8] hover:bg-white/[0.05] transition-all"
              title="Download Image Report"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg bg-[#191b22] border border-white/[0.08] text-[#8a8f98] hover:text-[#f7f8f8] hover:bg-[#eb5757]/20 hover:text-[#eb5757] transition-all"
              title="Close Dialog"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="px-5 py-2.5 bg-[#0a0b0d] border-b border-white/[0.06] grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 text-center text-xs">
          <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06]">
            <span className="text-[9px] font-mono text-[#62666d] block uppercase">Val PSNR</span>
            <span className="text-sm font-bold text-[#4ebb78] font-mono">
              {data.psnr || data.best_psnr} dB
            </span>
          </div>
          <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06]">
            <span className="text-[9px] font-mono text-[#62666d] block uppercase">Gain over Bicubic</span>
            <span className="text-sm font-bold text-[#5e6ad2] font-mono">
              +{data.psnr_gain || '0.00'} dB
            </span>
          </div>
          <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06]">
            <span className="text-[9px] font-mono text-[#62666d] block uppercase">Val SSIM</span>
            <span className="text-sm font-bold text-[#38bdf8] font-mono">
              {data.ssim || '--'}
            </span>
          </div>
          <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06]">
            <span className="text-[9px] font-mono text-[#62666d] block uppercase">L1 Pixel Loss</span>
            <span className="text-sm font-bold text-[#f7f8f8] font-mono">
              {typeof data.l1_loss === 'number' ? data.l1_loss.toFixed(4) : (data.train_loss || '--')}
            </span>
          </div>
          <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06]">
            <span className="text-[9px] font-mono text-[#62666d] block uppercase">FFT Frequency Loss</span>
            <span className="text-sm font-bold text-[#ec4899] font-mono">
              {typeof data.fft_loss === 'number' ? data.fft_loss.toFixed(4) : '--'}
            </span>
          </div>
          <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06]">
            <span className="text-[9px] font-mono text-[#62666d] block uppercase">Spatial Gradient</span>
            <span className="text-sm font-bold text-[#f2994a] font-mono">
              {typeof data.grad_loss === 'number' ? data.grad_loss.toFixed(4) : '--'}
            </span>
          </div>
        </div>

        {/* View Mode Segmented Controls */}
        <div className="px-5 py-2 flex items-center justify-between border-b border-white/[0.04]">
          <div className="flex items-center p-0.5 bg-[#14151a] rounded-lg border border-white/[0.08]">
            <button
              onClick={() => { setActiveView('eval'); setImgLoaded(false); }}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-medium transition-all ${
                activeView === 'eval'
                  ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08] shadow-sm'
                  : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <Layers className="w-3 h-3 text-[#4ebb78]" />
              <span>Multi-Patch Visual Comparison (RGB & CIR)</span>
            </button>
            <button
              onClick={() => { setActiveView('curves'); setImgLoaded(false); }}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-medium transition-all ${
                activeView === 'curves'
                  ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08] shadow-sm'
                  : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <TrendingUp className="w-3 h-3 text-[#5e6ad2]" />
              <span>6-Stream Telemetry Curves</span>
            </button>
          </div>

          <span className="text-[10px] text-[#62666d] font-mono">
            Checkpoints: {data.checkpoint_path || 'model/checkpoints/best_model.pth'}
          </span>
        </div>

        {/* High-Resolution Visual Report Body */}
        <div className="flex-1 min-h-[360px] overflow-y-auto p-4 flex items-center justify-center bg-[#07080a]">
          <div className="relative w-full flex justify-center">
            {!imgLoaded && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-[#8a8f98] space-y-2 min-h-[300px]">
                <div className="w-6 h-6 border-2 border-[#5e6ad2] border-t-transparent rounded-full animate-spin" />
                <span className="text-xs">Loading high-resolution report...</span>
              </div>
            )}
            <img
              key={currentImgUrl}
              src={currentImgUrl}
              alt="Evaluation Report"
              onLoad={() => setImgLoaded(true)}
              onError={(e) => {
                // Fallback to latest
                if (!currentImgUrl.includes('latest')) {
                  e.target.src = activeView === 'eval' 
                    ? `/api/artifacts/eval/latest?t=${Date.now()}`
                    : `/api/artifacts/curves/latest?t=${Date.now()}`;
                }
              }}
              className={`max-w-full rounded-xl border border-white/[0.08] shadow-lg transition-opacity duration-300 ${
                imgLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              style={{ maxHeight: 'calc(92vh - 230px)', objectFit: 'contain' }}
            />
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-white/[0.08] bg-[#14151a] flex items-center justify-between">
          <div className="flex items-center space-x-2 text-[11px] text-[#8a8f98]">
            <Sparkles className={`w-3.5 h-3.5 ${data.is_best ? 'text-[#4ebb78]' : 'text-[#5e6ad2]'}`} />
            <span>
              {data.is_best ? (
                <>⭐ Automatic best checkpoint saved to <code className="text-[#f7f8f8] bg-[#0f1013] px-1 py-0.5 rounded border border-white/[0.06]">best_model.pth</code></>
              ) : (
                <>Epoch {data.epoch} checkpoint saved to <code className="text-[#f7f8f8] bg-[#0f1013] px-1 py-0.5 rounded border border-white/[0.06]">last_checkpoint.pth</code> (Peak Val PSNR: <strong className="text-[#4ebb78]">{data.best_psnr ? `${Number(data.best_psnr).toFixed(2)} dB` : '--'}</strong>)</>
              )}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {onNavigateInference && (
              <button
                onClick={() => {
                  onClose();
                  onNavigateInference();
                }}
                className="px-3 py-1.5 rounded-lg bg-[#1f2128] border border-white/[0.08] hover:bg-white/[0.05] text-[#f7f8f8] text-xs font-medium transition-all flex items-center space-x-1.5"
              >
                <span>Test in Inference Tab</span>
                <ExternalLink className="w-3 h-3" />
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg bg-[#5e6ad2] hover:bg-[#6f7be2] text-white text-xs font-medium transition-all shadow-linear-btn"
            >
              Continue Training
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
