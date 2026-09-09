import React, { useRef, useEffect, useState } from 'react';
import { Terminal, Copy, Check, ArrowDown } from 'lucide-react';

export default function YoloTerminal({ logs = [], currentProgressLine = '', isTraining = false }) {
  const terminalEndRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (autoScroll && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, currentProgressLine, autoScroll]);

  const handleCopy = () => {
    const text = [...logs, currentProgressLine].filter(Boolean).join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-[#0c0d10] rounded-xl border border-white/[0.08] overflow-hidden flex flex-col h-[300px] shadow-linear-card">
      {/* Linear Style Terminal Bar */}
      <div className="px-4 py-2 bg-[#121318] border-b border-white/[0.06] flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-[#8a8f98]" />
          <span className="text-xs font-mono font-medium text-[#f7f8f8]">
            stdout: yolo-stream
          </span>
          {isTraining && (
            <span className="px-2 py-0.5 text-[9px] font-mono uppercase tracking-wider rounded bg-[#4ebb78]/15 text-[#4ebb78] border border-[#4ebb78]/30">
              Live
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-2 py-1 rounded text-[11px] font-mono flex items-center space-x-1 transition-all ${
              autoScroll ? 'bg-white/[0.08] text-[#f7f8f8] border border-white/[0.1]' : 'text-[#62666d] hover:text-[#8a8f98]'
            }`}
          >
            <ArrowDown className="w-3 h-3" />
            <span>Scroll</span>
          </button>

          <button
            onClick={handleCopy}
            className="p-1 rounded text-[#8a8f98] hover:text-[#f7f8f8] hover:bg-white/[0.05] transition-all"
            title="Copy Logs"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-[#4ebb78]" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Terminal Content */}
      <div className="flex-1 p-3.5 overflow-y-auto font-mono text-[11px] text-[#f7f8f8] leading-relaxed select-text">
        {/* Fixed Header */}
        <div className="text-[#8a8f98] font-semibold border-b border-white/[0.06] pb-1.5 mb-1.5 select-none tracking-tight">
          {'Epoch'.padStart(10)} {'GPU_mem'.padStart(10)} {'L1_loss'.padStart(10)} {'SSIM_loss'.padStart(10)} {'Total'.padStart(10)} {'PSNR'.padStart(10)} {'SSIM'.padStart(10)} {'Instances'.padStart(10)} {'Size'.padStart(10)}
        </div>

        {logs.length === 0 && !currentProgressLine && (
          <div className="text-[#62666d] italic py-6 text-center text-xs">
            Training output will stream here dynamically during execution.
          </div>
        )}

        {/* History */}
        {logs.map((log, idx) => (
          <div
            key={idx}
            className={`whitespace-pre font-mono py-0.5 ${
              log.includes('[*] Best model')
                ? 'text-[#4ebb78] font-semibold bg-[#4ebb78]/10 px-1 rounded'
                : 'text-[#d0d3d8]'
            }`}
          >
            {log}
          </div>
        ))}

        {/* Current inline updating line */}
        {currentProgressLine && (
          <div className="whitespace-pre font-mono py-0.5 text-[#f7f8f8] bg-white/[0.05] px-1 rounded border-l-2 border-[#5e6ad2]">
            {currentProgressLine}
          </div>
        )}

        <div ref={terminalEndRef} />
      </div>
    </div>
  );
}
