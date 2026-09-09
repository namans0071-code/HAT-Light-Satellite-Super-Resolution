import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileCode, 
  FolderOpen, 
  Sparkles, 
  Layers,
  Image as ImageIcon,
  CheckCircle2,
  X,
  FileCheck2,
  HardDrive
} from 'lucide-react';

export default function MultiFormatDropzone({ 
  customFile,
  onFileSelected, 
  customPath, 
  setCustomPath,
  onLoadTrigger 
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const supportedFormats = [
    { ext: 'tif', label: 'GeoTIFF (.tif, .tiff)', badge: '4-Band 10m' },
    { ext: 'npy', label: 'NumPy Array (.npy)', badge: 'uint16 Raw' },
    { ext: 'png', label: 'PNG Image (.png)', badge: 'RGB' },
    { ext: 'jpg', label: 'JPEG Image (.jpg)', badge: 'Visual' },
    { ext: 'jp2', label: 'JPEG 2000 (.jp2)', badge: 'Sentinel-2 L2A' }
  ];

  const handleOpenFileDialog = (e) => {
    e?.preventDefault();
    e?.stopPropagation();
    if (fileInputRef.current) {
      // Reset input value so re-selecting the same file triggers onChange
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setCustomPath(file.name);
      onFileSelected(file, file.name);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setCustomPath(file.name);
      onFileSelected(file, file.name);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getFormatBadge = (filename = '') => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'tif' || ext === 'tiff' || ext === 'geotiff') return { label: 'GeoTIFF (4-Band)', color: 'text-amber-400 border-amber-500/30 bg-amber-500/10' };
    if (ext === 'npy' || ext === 'npz') return { label: 'NumPy Reflectance', color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' };
    if (ext === 'jp2') return { label: 'JPEG 2000 (L2A)', color: 'text-violet-400 border-violet-500/30 bg-violet-500/10' };
    return { label: `${(ext || 'IMG').toUpperCase()} Image`, color: 'text-blue-400 border-blue-500/30 bg-blue-500/10' };
  };

  const handleLoadClick = () => {
    if (customFile) {
      if (onLoadTrigger) onLoadTrigger();
    } else if (customPath && customPath.trim()) {
      onFileSelected(null, customPath.trim());
      if (onLoadTrigger) onLoadTrigger();
    } else {
      // Empty input: trigger file picker popup
      handleOpenFileDialog();
    }
  };

  const badge = getFormatBadge(customFile?.name || customPath);

  return (
    <div className="flex flex-col gap-3">
      {/* Hidden Native File Input for OS File Picker Dialog */}
      <input 
        type="file" 
        ref={fileInputRef}
        accept=".tif,.tiff,.geotiff,.npy,.npz,.png,.jpg,.jpeg,.jp2,.bmp,.webp"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Drop Target & Click-to-Browse Area */}
      <div 
        onClick={handleOpenFileDialog}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-xl p-5 transition-all flex flex-col items-center justify-center text-center cursor-pointer group ${
          isDragging 
            ? 'border-[#5e6ad2] bg-[#5e6ad2]/15 scale-[0.99]' 
            : customFile 
              ? 'border-emerald-500/50 bg-emerald-500/5 hover:border-emerald-500/70' 
              : 'border-[#232529] hover:border-[#5e6ad2]/60 bg-[#141517]/70 hover:bg-[#18191c]'
        }`}
      >
        {customFile ? (
          /* Active Selected File Preview Card */
          <div className="flex flex-col items-center gap-2 w-full">
            <div className="w-11 h-11 rounded-full bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <div className="flex flex-col items-center max-w-full px-2">
              <div className="text-xs font-semibold text-white truncate max-w-[280px]">
                {customFile.name}
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono border font-semibold ${badge.color}`}>
                  {badge.label}
                </span>
                {customFile.size && (
                  <span className="text-[10px] font-mono text-[#8a8f98]">
                    {formatFileSize(customFile.size)}
                  </span>
                )}
              </div>
            </div>
            <button
              type="button"
              onClick={handleOpenFileDialog}
              className="mt-2 text-[11px] font-mono text-[#5e6ad2] hover:text-[#7f8be5] underline transition-colors"
            >
              Click to choose a different file
            </button>
          </div>
        ) : (
          /* Initial Upload Prompt */
          <>
            <div className="w-11 h-11 rounded-full bg-[#1c1d20] border border-[#232529] group-hover:border-[#5e6ad2]/40 flex items-center justify-center mb-2 text-[#5e6ad2] transition-colors">
              <UploadCloud className="w-5 h-5 group-hover:scale-110 transition-transform" />
            </div>
            <p className="text-xs font-semibold text-white mb-1">
              Click to browse or drop satellite scene
            </p>
            <p className="text-[11px] text-[#8a8f98] max-w-[260px]">
              Opens native file selector dialog. GeoTIFF, NumPy uint16, JP2 & standard images supported.
            </p>

            {/* Format Badges */}
            <div className="flex flex-wrap gap-1 mt-3 justify-center">
              {supportedFormats.map((f) => (
                <span 
                  key={f.ext}
                  className="px-1.5 py-0.5 rounded border border-[#232529] bg-[#0f1011] text-[9px] font-mono text-[#8a8f98]"
                >
                  .{f.ext}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Manual Path or OS Popup File Picker Field */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-[11px] font-mono text-[#8a8f98]">
          <span className="flex items-center gap-1">
            <HardDrive className="w-3 h-3 text-[#5e6ad2]" />
            Specify system path or browse file:
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="relative flex-1">
            <input 
              type="text" 
              value={customPath}
              onChange={(e) => setCustomPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleLoadClick();
              }}
              placeholder="e.g. D:/Scenes/sentinel2_tile.tif"
              className="w-full bg-[#141517] border border-[#232529] focus:border-[#5e6ad2] rounded-lg pl-3 pr-8 py-2 text-xs font-mono text-white placeholder-[#454954] outline-none transition-colors"
            />
            {customPath && (
              <button
                type="button"
                onClick={() => {
                  setCustomPath('');
                  onFileSelected(null, '');
                }}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#8a8f98] hover:text-white"
                title="Clear input"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Browse Button: Always opens the native OS popup file selector */}
          <button 
            type="button"
            onClick={handleOpenFileDialog}
            title="Open native file selection popup"
            className="px-2.5 py-2 rounded-lg bg-[#1c1d20] hover:bg-[#282a30] text-[#f7f8f8] border border-[#2e3138] hover:border-[#5e6ad2] text-xs font-medium transition-all flex items-center gap-1.5 flex-shrink-0"
          >
            <FolderOpen className="w-3.5 h-3.5 text-amber-400" />
            <span>Browse...</span>
          </button>

          {/* Load Button */}
          <button 
            type="button"
            onClick={handleLoadClick}
            className="px-3 py-2 rounded-lg bg-[#5e6ad2] hover:bg-[#6e7be0] text-xs font-semibold text-white transition-all flex items-center gap-1.5 flex-shrink-0 shadow-sm"
          >
            <span>Load</span>
          </button>
        </div>
      </div>
    </div>
  );
}
