import React, { useState } from 'react';
import { 
  X, 
  Download, 
  CheckCircle2, 
  FolderCheck, 
  FileCode, 
  Layers,
  Sparkles
} from 'lucide-react';
import { inferenceService } from '../services/inferenceService.js';

export default function ExportModal({ isOpen, onClose, currentSample, scale = 4.0 }) {
  const [filename, setFilename] = useState(`${currentSample?.id || 'satellite_scene'}_sr_4x`);
  const [format, setFormat] = useState('tif');
  const [bitDepth, setBitDepth] = useState('16-bit');
  const [exportPath, setExportPath] = useState('./exports');
  const [isExporting, setIsExporting] = useState(false);
  const [successNotice, setSuccessNotice] = useState(null);

  if (!isOpen) return null;

  const handleExport = async () => {
    setIsExporting(true);
    setSuccessNotice(null);
    try {
      const res = await inferenceService.exportSuperResolution({
        filename,
        format,
        scale,
        bitDepth,
        exportPath
      });
      setSuccessNotice(`Exported as ${res.filename} in ${res.format} format.`);
      setTimeout(() => {
        setIsExporting(false);
      }, 500);
    } catch (e) {
      setIsExporting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-md bg-[#0f1011] border border-[#232529] rounded-xl shadow-linear-modal overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="h-12 border-b border-[#232529] px-4 flex items-center justify-between bg-[#141517]">
          <div className="flex items-center gap-2">
            <Download className="w-4 h-4 text-[#5e6ad2]" />
            <span className="text-xs font-semibold text-white">Export Super-Resolved Output</span>
          </div>
          <button 
            onClick={onClose}
            className="p-1 rounded text-[#8a8f98] hover:text-white hover:bg-[#1c1d20] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 flex flex-col gap-4 text-xs">
          {/* Filename Field */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-[#8a8f98]">Output Filename:</label>
            <input 
              type="text" 
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              className="bg-[#141517] border border-[#232529] rounded-lg px-3 py-2 text-xs font-mono text-white outline-none focus:border-[#5e6ad2]"
            />
          </div>

          {/* Format Selection (Matches Input Format) */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-[#8a8f98]">
              Target Export Format <span className="text-emerald-400 font-semibold">(Preserves Input Format)</span>:
            </label>
            <div className="grid grid-cols-4 gap-2">
              {[
                { id: 'tif', label: 'GeoTIFF' },
                { id: 'npy', label: 'NumPy' },
                { id: 'png', label: 'PNG' },
                { id: 'jpg', label: 'JPEG' }
              ].map((f) => (
                <button
                  key={f.id}
                  onClick={() => setFormat(f.id)}
                  className={`py-2 px-2.5 rounded-lg border text-xs font-mono transition-all flex flex-col items-center justify-center gap-1 ${
                    format === f.id 
                      ? 'border-[#5e6ad2] bg-[#5e6ad2]/15 text-white font-semibold' 
                      : 'border-[#232529] bg-[#141517] text-[#8a8f98] hover:text-white hover:border-[#3e424b]'
                  }`}
                >
                  <span>{f.label}</span>
                  <span className="text-[9px] text-[#8a8f98]">.{f.id}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Bit Depth & Radiometry */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-[#8a8f98]">Radiometric Calibration:</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setBitDepth('16-bit')}
                className={`p-2 rounded-lg border text-left text-xs transition-all ${
                  bitDepth === '16-bit'
                    ? 'border-emerald-500/40 bg-emerald-500/10 text-white'
                    : 'border-[#232529] bg-[#141517] text-[#8a8f98]'
                }`}
              >
                <div className="font-semibold text-emerald-400">16-bit uint16</div>
                <div className="text-[10px] text-[#8a8f98]">Scientific BOA reflectance</div>
              </button>
              <button
                onClick={() => setBitDepth('8-bit')}
                className={`p-2 rounded-lg border text-left text-xs transition-all ${
                  bitDepth === '8-bit'
                    ? 'border-[#5e6ad2] bg-[#5e6ad2]/10 text-white'
                    : 'border-[#232529] bg-[#141517] text-[#8a8f98]'
                }`}
              >
                <div className="font-semibold text-[#f7f8f8]">8-bit uint8</div>
                <div className="text-[10px] text-[#8a8f98]">Visual preview image</div>
              </button>
            </div>
          </div>

          {/* Target Location */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-[#8a8f98]">Destination Directory:</label>
            <input 
              type="text" 
              value={exportPath}
              onChange={(e) => setExportPath(e.target.value)}
              className="bg-[#141517] border border-[#232529] rounded-lg px-3 py-2 text-xs font-mono text-white outline-none focus:border-[#5e6ad2]"
            />
          </div>

          {/* Success message */}
          {successNotice && (
            <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" />
              <span>{successNotice}</span>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="h-14 border-t border-[#232529] bg-[#141517] px-4 flex items-center justify-end gap-2">
          <button 
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg border border-[#232529] text-xs text-[#8a8f98] hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button 
            onClick={handleExport}
            disabled={isExporting}
            className="px-4 py-1.5 rounded-lg bg-[#5e6ad2] hover:bg-[#6e7be0] text-white text-xs font-medium transition-all flex items-center gap-1.5 shadow-sm"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{isExporting ? 'Exporting...' : 'Save File'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
