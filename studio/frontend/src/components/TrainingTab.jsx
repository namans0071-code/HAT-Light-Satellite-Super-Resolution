import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Square, FolderOpen, FileText, CheckCircle, ShieldCheck, Terminal, Award, ExternalLink } from 'lucide-react';
import TelemetryCharts from './TelemetryCharts.jsx';

export default function TrainingTab({
  systemStatus,
  logs = [],
  batchProgress,
  history,
  latestBestReport,
  onOpenBestReport,
  onStartTraining,
  onPauseTraining,
  onResumeTraining,
  onStopTraining
}) {
  const isTraining = systemStatus.trainer_state === 'TRAINING';
  const isPaused = systemStatus.trainer_state === 'PAUSED';

  // Persistent settings with localStorage
  const [epochs, setEpochs] = useState(() => localStorage.getItem('rcan_epochs') || 75);
  const [batchSize, setBatchSize] = useState(() => localStorage.getItem('rcan_batch_size') || 16);
  const [learningRate, setLearningRate] = useState(() => localStorage.getItem('rcan_lr') || 0.0002);
  const [scale, setScale] = useState(() => localStorage.getItem('rcan_scale') || 4);
  const [checkpointPath, setCheckpointPath] = useState(() => localStorage.getItem('rcan_ckpt_path') || '');
  const [exportDir, setExportDir] = useState(() => localStorage.getItem('rcan_export_dir') || 'model/checkpoints');
  const [trainDir, setTrainDir] = useState(() => localStorage.getItem('rcan_train_dir') || 'Dataset/data/Train');
  const [valDir, setValDir] = useState(() => localStorage.getItem('rcan_val_dir') || 'Dataset/data/Val');
  const [earlyStopping, setEarlyStopping] = useState(() => localStorage.getItem('rcan_early_stopping') !== 'false');
  const [patience, setPatience] = useState(() => localStorage.getItem('rcan_patience') || 15);
  const [minDelta, setMinDelta] = useState(() => localStorage.getItem('rcan_min_delta') || 0.01);
  const [gradAccumSteps, setGradAccumSteps] = useState(() => localStorage.getItem('rcan_grad_accum') || 1);

  useEffect(() => {
    localStorage.setItem('rcan_epochs', epochs);
    localStorage.setItem('rcan_batch_size', batchSize);
    localStorage.setItem('rcan_lr', learningRate);
    localStorage.setItem('rcan_scale', scale);
    localStorage.setItem('rcan_ckpt_path', checkpointPath);
    localStorage.setItem('rcan_export_dir', exportDir);
    localStorage.setItem('rcan_train_dir', trainDir);
    localStorage.setItem('rcan_val_dir', valDir);
    localStorage.setItem('rcan_early_stopping', earlyStopping ? 'true' : 'false');
    localStorage.setItem('rcan_patience', patience.toString());
    localStorage.setItem('rcan_min_delta', minDelta.toString());
    localStorage.setItem('rcan_grad_accum', gradAccumSteps.toString());
  }, [epochs, batchSize, learningRate, scale, checkpointPath, exportDir, trainDir, valDir, earlyStopping, patience, minDelta, gradAccumSteps]);

  const handleBrowse = async (mode, currentVal, setter) => {
    try {
      const res = await fetch('/api/utils/browse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_path: currentVal, mode: mode })
      });
      const data = await res.json();
      if (data.selected) {
        setter(data.selected);
      }
    } catch (e) {
      console.error('Browse error:', e);
    }
  };

  const handleStart = () => {
    onStartTraining({
      epochs: parseInt(epochs),
      batch_size: parseInt(batchSize),
      learning_rate: parseFloat(learningRate),
      scale: parseFloat(scale),
      grad_accum_steps: parseInt(gradAccumSteps),
      checkpoint_path: checkpointPath.trim() || null,
      export_dir: exportDir.trim(),
      train_dir: trainDir.trim(),
      val_dir: valDir.trim(),
      early_stopping: earlyStopping,
      patience: parseInt(patience),
      min_delta: parseFloat(minDelta)
    });
  };

  const [leftWidth, setLeftWidth] = useState(() => {
    const saved = localStorage.getItem('training_left_panel_width');
    return saved ? parseInt(saved, 10) : 380;
  });
  const [isDragging, setIsDragging] = useState(false);

  const [consoleHeight, setConsoleHeight] = useState(() => {
    const saved = localStorage.getItem('training_console_height');
    return saved ? parseInt(saved, 10) : 130;
  });
  const [isDraggingConsole, setIsDraggingConsole] = useState(false);

  useEffect(() => {
    if (!isDragging && !isDraggingConsole) return;

    const handleMouseMove = (e) => {
      if (isDragging) {
        const newWidth = Math.min(Math.max(280, e.clientX - 16), 680);
        setLeftWidth(newWidth);
        localStorage.setItem('training_left_panel_width', newWidth.toString());
      }
      if (isDraggingConsole) {
        const windowHeight = window.innerHeight;
        const newHeight = Math.min(Math.max(32, windowHeight - e.clientY - 20), 380);
        setConsoleHeight(newHeight);
        localStorage.setItem('training_console_height', newHeight.toString());
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsDraggingConsole(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, isDraggingConsole]);

  return (
    <div className={`flex gap-2.5 h-[calc(100vh-80px)] overflow-hidden ${isDragging ? 'select-none cursor-col-resize' : ''}`}>
      
      {/* Left Column: Controls, Parameters, Live Progress */}
      <div
        style={{ width: `${leftWidth}px` }}
        className="flex-shrink-0 flex flex-col space-y-2.5 h-full overflow-y-auto pr-1"
      >
        
        {/* Action & Status Card */}
        <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3.5 shadow-linear-card flex-shrink-0">
          <div className="flex items-center justify-between mb-2.5">
            <div>
              <h2 className="text-xs font-semibold text-[#f7f8f8]">Training Controls</h2>
              <span className="text-[10px] text-[#62666d]">HAT-Sat-Pro (~5.09M Params) • 6 RHAG • Hybrid Attention Transformer</span>
            </div>

            <span className={`px-2 py-0.5 text-[10px] font-medium rounded border ${
              isTraining ? 'bg-[#4ebb78]/15 text-[#4ebb78] border-[#4ebb78]/30' :
              isPaused ? 'bg-[#f2994a]/15 text-[#f2994a] border-[#f2994a]/30' :
              'bg-[#14151a] text-[#8a8f98] border-white/[0.08]'
            }`}>
              {systemStatus.trainer_state || 'IDLE'}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 my-2 text-center">
            <div className="bg-[#14151a] p-1.5 rounded-lg border border-white/[0.05]">
              <div className="text-[9px] text-[#62666d] uppercase font-medium">Epoch</div>
              <div className="text-sm font-semibold text-[#f7f8f8] font-mono">
                {systemStatus.current_epoch}/{systemStatus.total_epochs || epochs}
              </div>
            </div>

            <div className="bg-[#14151a] p-1.5 rounded-lg border border-white/[0.05]">
              <div className="text-[9px] text-[#62666d] uppercase font-medium">Best PSNR</div>
              <div className="text-sm font-semibold text-[#4ebb78] font-mono">
                {systemStatus.best_psnr > 0 ? `${systemStatus.best_psnr.toFixed(2)} dB` : '--'}
              </div>
            </div>

            <div className="bg-[#14151a] p-1.5 rounded-lg border border-white/[0.05]">
              <div className="text-[9px] text-[#62666d] uppercase font-medium">Scale</div>
              <div className="text-sm font-semibold text-[#5e6ad2] font-mono">
                {scale}x
              </div>
            </div>
          </div>

          {onOpenBestReport && (
            <button
              onClick={() => onOpenBestReport()}
              disabled={!(systemStatus.best_psnr > 0 || latestBestReport)}
              className="w-full my-2 py-1.5 px-2.5 rounded-lg bg-[#14151a] border border-[#4ebb78]/30 hover:bg-[#4ebb78]/10 text-[#4ebb78] text-[11px] font-medium transition-all flex items-center justify-between space-x-1.5 disabled:opacity-40 disabled:pointer-events-none group"
              title="Opens interactive multi-patch viewer in an independent parallel window with zoom and pan controls"
            >
              <div className="flex items-center space-x-1.5">
                <Award className="w-3.5 h-3.5 text-[#4ebb78]" />
                <span>View Multi-Patch Evaluation Report</span>
              </div>
              <div className="flex items-center space-x-1 text-[10px] text-[#4ebb78]/80 group-hover:text-[#4ebb78]">
                <span>Pop-out</span>
                <ExternalLink className="w-3 h-3" />
              </div>
            </button>
          )}

          <div className="flex items-center space-x-2 pt-2 border-t border-white/[0.06]">
            {!isTraining && !isPaused && (
              <button
                onClick={handleStart}
                className="flex-1 py-1.5 px-3 rounded-md bg-[#5e6ad2] hover:bg-[#6f7be2] text-white font-medium text-xs shadow-linear-btn transition-all flex items-center justify-center space-x-1.5"
              >
                <Play className="w-3 h-3 fill-white" />
                <span>Start Training</span>
              </button>
            )}

            {isTraining && (
              <button
                onClick={onPauseTraining}
                className="flex-1 py-1.5 px-3 rounded-md bg-[#14151a] border border-[#f2994a]/40 text-[#f2994a] hover:bg-[#f2994a]/10 font-medium text-xs transition-all flex items-center justify-center space-x-1.5"
              >
                <Pause className="w-3 h-3 fill-[#f2994a]" />
                <span>Pause</span>
              </button>
            )}

            {isPaused && (
              <button
                onClick={onResumeTraining}
                className="flex-1 py-1.5 px-3 rounded-md bg-[#14151a] border border-[#4ebb78]/40 text-[#4ebb78] hover:bg-[#4ebb78]/10 font-medium text-xs transition-all flex items-center justify-center space-x-1.5"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Resume</span>
              </button>
            )}

            {(isTraining || isPaused) && (
              <button
                onClick={onStopTraining}
                className="py-1.5 px-3 rounded-md bg-[#14151a] border border-[#eb5757]/40 text-[#eb5757] hover:bg-[#eb5757]/10 font-medium text-xs transition-all flex items-center justify-center space-x-1"
              >
                <Square className="w-2.5 h-2.5 fill-[#eb5757]" />
                <span>Stop</span>
              </button>
            )}
          </div>
        </div>

        {/* Structured Progress & Telemetry Container */}
        <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 shadow-linear-card flex-shrink-0">
          <div className="flex items-center justify-between mb-1.5 text-xs">
            <span className="font-semibold text-[#f7f8f8]">Training Progress</span>
            <span className="font-mono text-[10px] text-[#8a8f98]">
              {batchProgress?.it_per_sec ? `${batchProgress.it_per_sec} it/s` : '-- it/s'} | Elapsed: {batchProgress?.elapsed_str || '00:00'} | ETA: {batchProgress?.eta_str || '00:00'}
            </span>
          </div>

          {/* Smooth Linear Progress Bar */}
          <div className="w-full h-2 bg-[#14151a] rounded-full overflow-hidden border border-white/[0.06] mb-2">
            <div
              className="h-full bg-[#5e6ad2] transition-all duration-300 rounded-full"
              style={{ width: `${batchProgress?.pct || 0}%` }}
            />
          </div>

          {/* Structured Telemetry Table Box */}
          <div className="bg-[#14151a] rounded-lg border border-white/[0.06] p-2 font-mono text-[10px]">
            <div className="grid grid-cols-6 gap-1 text-[#62666d] border-b border-white/[0.04] pb-1 mb-1">
              <div>BATCH</div>
              <div>TOTAL</div>
              <div>L1</div>
              <div>FFT</div>
              <div>GRAD</div>
              <div>VRAM</div>
            </div>
            <div className="grid grid-cols-6 gap-1 text-[#f7f8f8]">
              <div>{batchProgress?.batch || 0}/{batchProgress?.total_batches || 0}</div>
              <div className="text-[#5e6ad2]">{batchProgress?.total_loss?.toFixed(4) || '--'}</div>
              <div className="text-[#4ebb78]">{batchProgress?.l1_loss?.toFixed(4) || '--'}</div>
              <div className="text-[#38bdf8]">{batchProgress?.fft_loss?.toFixed(4) || '--'}</div>
              <div className="text-[#ec4899]">{batchProgress?.grad_loss?.toFixed(4) || '--'}</div>
              <div className="text-[#8a8f98]">{batchProgress?.gpu_mem || '--'}</div>
            </div>
          </div>
        </div>

        {/* Hyperparameters & Persistent Browse Fields */}
        <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3.5 shadow-linear-card flex-1 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold text-[#f7f8f8]">Hyperparameters & Paths</h3>
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#5e6ad2]/20 text-[#5e6ad2] border border-[#5e6ad2]/30">
              HAT-Sat-Pro 5.09M
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="grid grid-cols-4 gap-2">
              <div>
                <label className="block text-[10px] text-[#8a8f98] mb-0.5">Epochs</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-[#14151a] border border-white/[0.08] rounded px-2 py-1 text-xs text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-[10px] text-[#8a8f98] mb-0.5">Batch</label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-[#14151a] border border-white/[0.08] rounded px-2 py-1 text-xs text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-[10px] text-[#8a8f98] mb-0.5" title="Gradient Accumulation Steps">Accum</label>
                <select
                  value={gradAccumSteps}
                  onChange={(e) => setGradAccumSteps(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-[#14151a] border border-white/[0.08] rounded px-1.5 py-1 text-xs text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                >
                  <option value="1">1x</option>
                  <option value="2">2x</option>
                  <option value="4">4x</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-[#8a8f98] mb-0.5">LR</label>
                <input
                  type="number"
                  step="0.00005"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-[#14151a] border border-white/[0.08] rounded px-2 py-1 text-xs text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                />
              </div>
            </div>

            {/* Early Stopping Toggle */}
            <div className="bg-[#14151a] p-2 rounded-lg border border-white/[0.06] flex items-center justify-between">
              <div className="flex items-center space-x-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-[#4ebb78]" />
                <span className="text-[11px] text-[#f7f8f8]">Early Stopping</span>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={earlyStopping}
                  onChange={(e) => setEarlyStopping(e.target.checked)}
                  disabled={isTraining}
                  className="accent-[#5e6ad2] rounded"
                />
                <span className="text-[10px] text-[#8a8f98]">Patience:</span>
                <input
                  type="number"
                  value={patience}
                  onChange={(e) => setPatience(e.target.value)}
                  disabled={isTraining || !earlyStopping}
                  className="w-10 bg-[#0f1013] border border-white/[0.08] rounded px-1 text-[10px] text-center text-[#f7f8f8] font-mono"
                  title="Epochs without improvement before stopping"
                />
                <span className="text-[10px] text-[#8a8f98]">Δ dB:</span>
                <input
                  type="number"
                  step="0.01"
                  value={minDelta}
                  onChange={(e) => setMinDelta(e.target.value)}
                  disabled={isTraining || !earlyStopping}
                  className="w-12 bg-[#0f1013] border border-white/[0.08] rounded px-1 text-[10px] text-center text-[#f7f8f8] font-mono"
                  title="Minimum PSNR improvement (dB) to reset patience"
                />
              </div>
            </div>

            {/* Browse Checkpoint */}
            <div>
              <label className="block text-[10px] text-[#8a8f98] mb-0.5">Resume Checkpoint (.pth)</label>
              <div className="flex items-center space-x-1">
                <input
                  type="text"
                  placeholder="Optional pre-existing weights"
                  value={checkpointPath}
                  onChange={(e) => setCheckpointPath(e.target.value)}
                  disabled={isTraining}
                  className="flex-1 bg-[#14151a] border border-white/[0.08] rounded px-2 py-1 text-[11px] text-[#f7f8f8] font-mono truncate focus:border-[#5e6ad2] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => handleBrowse('file', checkpointPath, setCheckpointPath)}
                  disabled={isTraining}
                  className="p-1 rounded bg-[#14151a] border border-white/[0.08] hover:bg-white/[0.05] text-[#8a8f98] hover:text-[#f7f8f8]"
                  title="Browse File"
                >
                  <FileText className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Browse Export Dir */}
            <div>
              <div className="flex items-center justify-between mb-0.5">
                <label className="block text-[10px] text-[#8a8f98]">Model & Artifacts Export Dir</label>
                <span className="text-[8px] text-[#4ebb78] font-mono">Exclusive Target</span>
              </div>
              <div className="flex items-center space-x-1">
                <input
                  type="text"
                  value={exportDir}
                  onChange={(e) => setExportDir(e.target.value)}
                  disabled={isTraining}
                  placeholder="e.g. model/checkpoints/v4"
                  className="flex-1 bg-[#14151a] border border-white/[0.08] rounded px-2 py-1 text-[11px] text-[#f7f8f8] font-mono truncate focus:border-[#5e6ad2] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => handleBrowse('folder', exportDir, setExportDir)}
                  disabled={isTraining}
                  className="p-1 rounded bg-[#14151a] border border-white/[0.08] hover:bg-white/[0.05] text-[#8a8f98] hover:text-[#f7f8f8]"
                  title="Browse Folder"
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Dataset Train Dir */}
            <div>
              <label className="block text-[10px] text-[#8a8f98] mb-0.5">Train Dataset Directory</label>
              <div className="flex items-center space-x-1">
                <input
                  type="text"
                  value={trainDir}
                  onChange={(e) => setTrainDir(e.target.value)}
                  disabled={isTraining}
                  className="flex-1 bg-[#14151a] border border-white/[0.08] rounded px-2 py-1 text-[11px] text-[#f7f8f8] font-mono truncate focus:border-[#5e6ad2] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => handleBrowse('folder', trainDir, setTrainDir)}
                  disabled={isTraining}
                  className="p-1 rounded bg-[#14151a] border border-white/[0.08] hover:bg-white/[0.05] text-[#8a8f98] hover:text-[#f7f8f8]"
                  title="Browse Folder"
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

          </div>
        </div>

      </div>

      {/* Draggable Vertical Divider Handle */}
      <div
        onMouseDown={() => setIsDragging(true)}
        onDoubleClick={() => {
          setLeftWidth(380);
          localStorage.setItem('training_left_panel_width', '380');
        }}
        className={`w-1.5 hover:w-2 hover:bg-[#5e6ad2]/70 cursor-col-resize rounded-full transition-all flex items-center justify-center group flex-shrink-0 ${
          isDragging ? 'bg-[#5e6ad2] w-2' : 'bg-white/[0.04]'
        }`}
        title="Drag to resize panels (Double-click to reset to 380px)"
      >
        <div className="h-8 w-0.5 bg-white/20 group-hover:bg-white/60 rounded-full" />
      </div>

      {/* Right Area: 6 Real-Time Telemetry Graphs in 3x2 Grid + Resizable Live Console Drawer */}
      <div className="flex-1 h-full flex flex-col overflow-hidden min-w-0">
        <div className="flex-1 min-h-0 overflow-hidden">
          <TelemetryCharts history={history} />
        </div>

        {/* Draggable Horizontal Divider Handle for Console Drawer */}
        <div
          onMouseDown={() => setIsDraggingConsole(true)}
          onDoubleClick={() => {
            setConsoleHeight(130);
            localStorage.setItem('training_console_height', '130');
          }}
          className={`h-1.5 hover:h-2 hover:bg-[#5e6ad2]/70 cursor-row-resize rounded-full transition-all flex items-center justify-center group flex-shrink-0 my-0.5 ${
            isDraggingConsole ? 'bg-[#5e6ad2] h-2' : 'bg-white/[0.04]'
          }`}
          title="Drag to resize Live Terminal Output (Double-click to reset to 130px)"
        >
          <div className="w-10 h-0.5 bg-white/20 group-hover:bg-white/60 rounded-full" />
        </div>

        {/* Live Terminal & Telemetry Stream */}
        <div
          style={{ height: `${consoleHeight}px` }}
          className="bg-[#0f1013] rounded-xl border border-white/[0.08] flex flex-col overflow-hidden flex-shrink-0 shadow-linear-card"
        >
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06] bg-[#14151a]">
            <div className="flex items-center space-x-2">
              <Terminal className="w-3.5 h-3.5 text-[#5e6ad2]" />
              <span className="text-[11px] font-semibold text-[#f7f8f8]">Live Training Terminal & Telemetry Stream</span>
              <span className="text-[9px] px-1.5 py-0.2 rounded bg-white/[0.05] text-[#8a8f98] font-mono">
                {logs?.length || 0} events
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] text-[#62666d]">Drag bar to resize</span>
            </div>
          </div>
          <div className="flex-1 p-2.5 overflow-y-auto font-mono text-[11px] text-[#8a8f98] space-y-1 select-text">
            {(!logs || logs.length === 0) ? (
              <div className="text-[#62666d] italic">Waiting for training cycle to begin...</div>
            ) : (
              logs.map((log, idx) => (
                <div
                  key={idx}
                  className={`leading-relaxed whitespace-pre-wrap ${
                    log.includes('[ERROR]') ? 'text-[#eb5757]' :
                    log.includes('[*] Best model') || log.includes('PSNR:') ? 'text-[#4ebb78]' :
                    log.includes('[System]') ? 'text-[#38bdf8]' :
                    'text-[#a0a6b2]'
                  }`}
                >
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

    </div>
  );
}
