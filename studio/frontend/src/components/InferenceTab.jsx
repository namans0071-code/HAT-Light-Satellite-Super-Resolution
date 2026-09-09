import React, { useState, useEffect, useRef } from 'react';
import { 
  Sliders, 
  Layers, 
  Zap, 
  Sparkles, 
  BarChart3, 
  FileText,
  Cpu,
  AlertTriangle,
  Download,
  Flame,
  CheckCircle2,
  UploadCloud,
  FolderOpen,
  Image as ImageIcon,
  Copy,
  Check
} from 'lucide-react';
import ComparisonSlider from './ComparisonSlider.jsx';

export default function InferenceTab({ onRunInfer, onRunZeroShot, onRunImageFileInfer, isInferring = false }) {
  // Input Mode: 'patch' (curated dataset) or 'custom' (JPEG, PNG, GeoTIFF)
  const [inputMode, setInputMode] = useState(() => localStorage.getItem('rcan_infer_input_mode') || 'patch');
  const [customInputPath, setCustomInputPath] = useState(() => localStorage.getItem('rcan_infer_custom_input') || '');
  const [customOutputPath, setCustomOutputPath] = useState(() => localStorage.getItem('rcan_infer_custom_output') || '');
  const [customResult, setCustomResult] = useState(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [copiedPath, setCopiedPath] = useState(false);
  const fileInputRef = useRef(null);

  const [selectedCategory, setSelectedCategory] = useState(() => localStorage.getItem('rcan_infer_category') || 'airport');
  const [patchList, setPatchList] = useState({});
  const [selectedPatch, setSelectedPatch] = useState(() => localStorage.getItem('rcan_infer_patch') || '');
  const [availableModels, setAvailableModels] = useState([]);
  const [checkpointPath, setCheckpointPath] = useState(() => localStorage.getItem('rcan_infer_ckpt') || 'model/checkpoints/best_model.pth');
  const [scale, setScale] = useState(() => parseFloat(localStorage.getItem('rcan_infer_scale') || '4'));
  const [colorMode, setColorMode] = useState(() => localStorage.getItem('rcan_infer_colormode') || 'rgb');
  const [subTab, setSubTab] = useState(() => localStorage.getItem('rcan_infer_subtab') || 'slider');
  
  const [inferResult, setInferResult] = useState(null);
  const [zeroShotResult, setZeroShotResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  // Tiling Inference Settings (Default True for custom high-res satellite images)
  const [useTiling, setUseTiling] = useState(() => {
    const saved = localStorage.getItem('rcan_use_tiling');
    return saved !== null ? saved === 'true' : true;
  });
  const [tileSize, setTileSize] = useState(() => parseInt(localStorage.getItem('rcan_tile_size') || '128', 10));
  const [overlap, setOverlap] = useState(() => parseInt(localStorage.getItem('rcan_tile_overlap') || '32', 10));

  // Resizable Left Panel Width
  const [leftWidth, setLeftWidth] = useState(() => {
    const saved = localStorage.getItem('rcan_infer_left_panel_width');
    return saved ? parseInt(saved, 10) : 340;
  });
  const [isDraggingLeft, setIsDraggingLeft] = useState(false);

  useEffect(() => {
    localStorage.setItem('rcan_infer_input_mode', inputMode);
    localStorage.setItem('rcan_infer_custom_input', customInputPath);
    localStorage.setItem('rcan_infer_custom_output', customOutputPath);
    localStorage.setItem('rcan_infer_ckpt', checkpointPath);
    localStorage.setItem('rcan_infer_category', selectedCategory);
    if (selectedPatch) localStorage.setItem('rcan_infer_patch', selectedPatch);
    localStorage.setItem('rcan_infer_scale', scale.toString());
    localStorage.setItem('rcan_infer_colormode', colorMode);
    localStorage.setItem('rcan_infer_subtab', subTab);
    localStorage.setItem('rcan_use_tiling', useTiling ? 'true' : 'false');
    localStorage.setItem('rcan_tile_size', tileSize.toString());
    localStorage.setItem('rcan_tile_overlap', overlap.toString());
  }, [inputMode, customInputPath, customOutputPath, checkpointPath, selectedCategory, selectedPatch, scale, colorMode, subTab, useTiling, tileSize, overlap]);

  // Load available patches and available model checkpoints
  useEffect(() => {
    fetch('/api/patches/list')
      .then(res => res.json())
      .then(data => {
        setPatchList(data);
        const savedCat = localStorage.getItem('rcan_infer_category') || 'airport';
        const savedPatch = localStorage.getItem('rcan_infer_patch');
        const list = data[savedCat] || [];
        if (savedPatch && list.some(p => p.path === savedPatch)) {
          setSelectedPatch(savedPatch);
        } else if (list.length > 0) {
          setSelectedPatch(list[0].path);
        }
      })
      .catch(err => console.error('Failed to load patches:', err));

    fetch('/api/models/list')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setAvailableModels(data);
          // If no checkpoint is currently saved or path doesn't exist in list, pick the first
          if (data.length > 0 && (!checkpointPath || !data.some(m => m.path === checkpointPath))) {
            const best = data.find(m => m.name.includes('best_model') && m.path.includes('v4')) || data[0];
            setCheckpointPath(best.path);
          }
        }
      })
      .catch(err => console.error('Failed to load models:', err));
  }, []);

  // Left sidebar dragging
  useEffect(() => {
    if (!isDraggingLeft) return;

    const handleMouseMove = (e) => {
      const newWidth = Math.min(Math.max(280, e.clientX - 16), 560);
      setLeftWidth(newWidth);
      localStorage.setItem('rcan_infer_left_panel_width', newWidth.toString());
    };

    const handleMouseUp = () => {
      setIsDraggingLeft(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingLeft]);

  const handleBrowseFile = async () => {
    try {
      const res = await fetch('/api/utils/browse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_path: checkpointPath, mode: 'file' })
      });
      const data = await res.json();
      if (data.selected) {
        setCheckpointPath(data.selected);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleCategoryChange = (cat) => {
    setSelectedCategory(cat);
    const list = patchList[cat] || [];
    if (list.length > 0) {
      setSelectedPatch(list[0].path);
    }
  };

  const handleRunStandard = async () => {
    if (!selectedPatch) return;
    setErrorMsg('');
    try {
      const result = await onRunInfer({
        patch_path: selectedPatch,
        checkpoint_path: checkpointPath.trim() || null,
        scale: parseFloat(scale),
        use_tiling: useTiling,
        tile_size: parseInt(tileSize),
        overlap: parseInt(overlap)
      });
      if (result && result.images) {
        setInferResult(result);
      } else if (result && result.detail) {
        setErrorMsg(result.detail);
      }
    } catch (err) {
      setErrorMsg(err.message || 'Inference failed');
    }
  };

  const handleRunZeroShot = async () => {
    if (!selectedPatch) return;
    setErrorMsg('');
    try {
      const result = await onRunZeroShot({
        patch_path: selectedPatch,
        checkpoint_path: checkpointPath.trim() || null,
        scale: parseFloat(scale),
        use_tiling: useTiling,
        tile_size: parseInt(tileSize),
        overlap: parseInt(overlap)
      });
      if (result && result.images) {
        setZeroShotResult(result);
      } else if (result && result.detail) {
        setErrorMsg(result.detail);
      }
    } catch (err) {
      setErrorMsg(err.message || '0-Shot failed');
    }
  };

  const suggestOutputPath = (inputPath, targetScale = scale) => {
    if (!inputPath || !inputPath.trim()) return '';
    const clean = inputPath.trim().replace(/\\/g, '/');
    const lastDot = clean.lastIndexOf('.');
    const lastSlash = clean.lastIndexOf('/');
    if (lastDot > lastSlash && lastDot !== -1) {
      const ext = clean.substring(lastDot);
      const stem = clean.substring(lastSlash + 1, lastDot);
      const dir = lastSlash !== -1 ? clean.substring(0, lastSlash) : '.';
      return `${dir}/${stem}_SR_${targetScale}x${ext}`;
    }
    return `${clean}_SR_${targetScale}x.png`;
  };

  const processSelectedFile = async (file) => {
    setErrorMsg('');
    // In Electron, file.path contains native absolute filesystem path
    if (file.path) {
      const normPath = file.path.replace(/\\/g, '/');
      setCustomInputPath(normPath);
      setCustomOutputPath(suggestOutputPath(normPath, scale));
      return;
    }

    // In Web browser, upload file to backend temp directory
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/infer/upload-image', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Upload failed with status ${res.status}`);
      const data = await res.json();
      setCustomInputPath(data.path);
      setCustomOutputPath(suggestOutputPath(data.path, scale));
    } catch (err) {
      setErrorMsg(`File upload error: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDropFile = async (e) => {
    e.preventDefault();
    setIsDraggingFile(false);
    if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
    await processSelectedFile(e.dataTransfer.files[0]);
  };

  const handleFileInputChange = async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    await processSelectedFile(e.target.files[0]);
  };

  const handleBrowseCustomInput = async () => {
    try {
      const res = await fetch('/api/utils/browse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_path: customInputPath, mode: 'file', file_types: 'image' })
      });
      const data = await res.json();
      if (data.selected) {
        setCustomInputPath(data.selected);
        setCustomOutputPath(suggestOutputPath(data.selected, scale));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleBrowseCustomOutput = async () => {
    try {
      const res = await fetch('/api/utils/browse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_path: customOutputPath || customInputPath, mode: 'save', file_types: 'image' })
      });
      const data = await res.json();
      if (data.selected) {
        setCustomOutputPath(data.selected);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleRunCustomInfer = async () => {
    if (!customInputPath || !customInputPath.trim()) {
      setErrorMsg('Please select or drag & drop an image file first.');
      return;
    }
    setErrorMsg('');
    try {
      const result = await onRunImageFileInfer({
        input_path: customInputPath.trim(),
        output_path: customOutputPath.trim() || null,
        checkpoint_path: checkpointPath.trim() || null,
        scale: parseFloat(scale),
        use_tiling: useTiling,
        tile_size: parseInt(tileSize),
        overlap: parseInt(overlap)
      });
      if (result && result.images) {
        setCustomResult(result);
        if (result.output_path) {
          setCustomOutputPath(result.output_path);
        }
      } else if (result && result.detail) {
        setErrorMsg(result.detail);
      }
    } catch (err) {
      setErrorMsg(err.message || 'Custom image inference failed');
    }
  };

  const handleExportImage = () => {
    const activeImage = inputMode === 'custom'
      ? (colorMode === 'cir' && customResult?.images?.after_cir ? customResult?.images?.after_cir : customResult?.images?.after_rgb)
      : (subTab === 'zeroshot' 
        ? (colorMode === 'rgb' ? zeroShotResult?.images?.after_rgb : zeroShotResult?.images?.after_cir)
        : (colorMode === 'rgb' ? inferResult?.images?.sr_rgb : inferResult?.images?.sr_cir));

    if (!activeImage) return;

    const link = document.createElement('a');
    link.href = activeImage;
    const baseName = inputMode === 'custom'
      ? (customResult?.filename || 'sr_output').replace(/\.[^/.]+$/, "")
      : (inferResult?.patch_name || zeroShotResult?.patch_name || 'sr_output').replace('.npy', '');
    link.download = `HATLight_SR_${baseName}_${scale}x_${colorMode.toUpperCase()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopyOutputPath = () => {
    if (customResult?.output_path) {
      navigator.clipboard.writeText(customResult.output_path);
      setCopiedPath(true);
      setTimeout(() => setCopiedPath(false), 2000);
    }
  };

  const currentCategoryPatches = patchList[selectedCategory] || [];

  return (
    <div className={`flex gap-2.5 h-[calc(100vh-80px)] overflow-hidden ${isDraggingLeft ? 'select-none cursor-col-resize' : ''}`}>
      
      {/* Left Column: Resizable Controls, Patch Selection & Actions */}
      <div
        style={{ width: `${leftWidth}px` }}
        className="flex-shrink-0 flex flex-col space-y-2.5 h-full overflow-y-auto pr-1"
      >
        
        {/* Mode Switcher Tabs: Dataset Patches vs Custom Image */}
        <div className="flex p-0.5 bg-[#14151a] rounded-lg border border-white/[0.08] flex-shrink-0">
          <button
            type="button"
            onClick={() => setInputMode('patch')}
            className={`flex-1 py-1.5 px-2 rounded-md text-[11px] font-medium transition-all ${
              inputMode === 'patch'
                ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08] shadow-sm'
                : 'text-[#8a8f98] hover:text-[#f7f8f8]'
            }`}
          >
            Sample Patches (.npy)
          </button>
          <button
            type="button"
            onClick={() => setInputMode('custom')}
            className={`flex-1 py-1.5 px-2 rounded-md text-[11px] font-medium transition-all flex items-center justify-center space-x-1 ${
              inputMode === 'custom'
                ? 'bg-[#5e6ad2] text-white shadow-sm'
                : 'text-[#8a8f98] hover:text-[#f7f8f8]'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Custom Image (JPG/PNG/TIF)</span>
          </button>
        </div>

        {/* Option A: Curated Patch Selection Card */}
        {inputMode === 'patch' && (
          <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 shadow-linear-card flex-shrink-0">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-xs font-semibold text-[#f7f8f8]">Target Sentinel-2 Patch</h2>
              <span className="text-[10px] font-mono text-[#62666d]">{currentCategoryPatches.length} available</span>
            </div>

            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-[10px] text-[#8a8f98] mb-1">Geospatial Category</label>
                <select
                  value={selectedCategory}
                  onChange={(e) => handleCategoryChange(e.target.value)}
                  className="w-full bg-[#14151a] border border-white/[0.08] rounded px-2.5 py-1.5 text-xs text-[#f7f8f8] focus:border-[#5e6ad2] focus:outline-none"
                >
                  <option value="airport">Airports & Runways</option>
                  <option value="military_base">Military Naval & Air Bases</option>
                  <option value="city_roads">Urban Cities & Roads</option>
                  <option value="mountain_outpost">Mountain & Glaciers</option>
                  <option value="vegetation">Forest & Farmland</option>
                  <option value="water_ports">Ports & Harbors</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] text-[#8a8f98] mb-1">Patch File (.npy)</label>
                <select
                  value={selectedPatch}
                  onChange={(e) => setSelectedPatch(e.target.value)}
                  className="w-full bg-[#14151a] border border-white/[0.08] rounded px-2.5 py-1.5 text-xs text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                >
                  {currentCategoryPatches.map((p) => (
                    <option key={p.path} value={p.path}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Option B: Custom Image Source & Export Card */}
        {inputMode === 'custom' && (
          <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 shadow-linear-card flex-shrink-0 space-y-2.5">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold text-[#f7f8f8]">Custom Image Source</h2>
              <span className="text-[10px] font-mono text-[#38bdf8] bg-[#38bdf8]/10 px-1.5 py-0.5 rounded">
                JPEG • PNG • GeoTIFF
              </span>
            </div>

            {/* Drag and Drop Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDraggingFile(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDraggingFile(false); }}
              onDrop={handleDropFile}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-3 text-center cursor-pointer transition-all ${
                isDraggingFile
                  ? 'border-[#5e6ad2] bg-[#5e6ad2]/15 shadow-lg scale-[1.01]'
                  : 'border-white/[0.12] bg-[#14151a]/60 hover:border-white/[0.25]'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.tif,.tiff,.geotiff"
                className="hidden"
                onChange={handleFileInputChange}
              />
              <UploadCloud className={`w-6 h-6 mx-auto mb-1 ${isDraggingFile ? 'text-[#5e6ad2] animate-bounce' : 'text-[#8a8f98]'}`} />
              <p className="text-[11px] font-medium text-[#f7f8f8]">
                {isUploading ? 'Uploading file...' : isDraggingFile ? 'Drop file here!' : 'Drag & drop image or click to select'}
              </p>
              <p className="text-[9px] text-[#62666d] mt-0.5">
                GeoTIFF (.tif, .tiff), PNG, JPEG supported
              </p>
            </div>

            {/* Input Path Field */}
            <div>
              <label className="block text-[10px] text-[#8a8f98] mb-1">Input Image Path</label>
              <div className="flex items-center space-x-1">
                <input
                  type="text"
                  value={customInputPath}
                  onChange={(e) => {
                    setCustomInputPath(e.target.value);
                    setCustomOutputPath(suggestOutputPath(e.target.value, scale));
                  }}
                  placeholder="Drop file or enter path..."
                  className="flex-1 bg-[#14151a] border border-white/[0.08] rounded px-2 py-1.5 text-[11px] text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleBrowseCustomInput}
                  className="p-1.5 rounded bg-[#14151a] border border-white/[0.08] hover:bg-white/[0.05] text-[#8a8f98] hover:text-[#f7f8f8]"
                  title="Browse File"
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Export Destination Path Field */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-[10px] text-[#8a8f98]">Export Output Path</label>
                {customInputPath && (
                  <span className="text-[9px] font-mono text-[#4ebb78]">
                    Format: {customInputPath.toLowerCase().endsWith('.tif') || customInputPath.toLowerCase().endsWith('.tiff') ? 'GeoTIFF' : customInputPath.split('.').pop()?.toUpperCase()}
                  </span>
                )}
              </div>
              <div className="flex items-center space-x-1">
                <input
                  type="text"
                  value={customOutputPath}
                  onChange={(e) => setCustomOutputPath(e.target.value)}
                  placeholder="Select export path..."
                  className="flex-1 bg-[#14151a] border border-white/[0.08] rounded px-2 py-1.5 text-[11px] text-[#f7f8f8] font-mono focus:border-[#5e6ad2] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleBrowseCustomOutput}
                  className="p-1.5 rounded bg-[#14151a] border border-white/[0.08] hover:bg-white/[0.05] text-[#8a8f98] hover:text-[#f7f8f8]"
                  title="Browse Export Location"
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                </button>
              </div>
              <p className="text-[9px] text-[#62666d] mt-1">
                Outputs in identical format (GeoTIFF CRS & GeoTransform preserved)
              </p>
            </div>
          </div>
        )}

        {/* Telemetry Summary Card */}
        <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 shadow-linear-card flex-1 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-1.5 text-xs text-[#8a8f98] mb-2">
              <Cpu className="w-3.5 h-3.5 text-[#38bdf8]" />
              <span className="font-semibold text-[#f7f8f8]">Engine Telemetry</span>
            </div>
            <div className="space-y-1.5 text-[11px] font-mono">
              <div className="flex justify-between text-[#8a8f98]">
                <span>Status:</span>
                <span className={isInferring ? 'text-[#f2994a]' : 'text-[#4ebb78]'}>
                  {isInferring ? 'Executing GPU pass...' : 'Ready'}
                </span>
              </div>
              <div className="flex justify-between text-[#8a8f98]">
                <span>Target Scale:</span>
                <span className="text-[#f7f8f8]">{scale}x</span>
              </div>

              {inputMode === 'custom' && customResult ? (
                <>
                  <div className="flex justify-between text-[#8a8f98]">
                    <span>Format:</span>
                    <span className="text-[#38bdf8] font-semibold">{customResult.format}</span>
                  </div>
                  <div className="flex justify-between text-[#8a8f98]">
                    <span>Input Res:</span>
                    <span className="text-[#f7f8f8]">{customResult.input_shape[1]}x{customResult.input_shape[0]}</span>
                  </div>
                  <div className="flex justify-between text-[#8a8f98]">
                    <span>Output Res:</span>
                    <span className="text-[#4ebb78] font-semibold">{customResult.output_shape[1]}x{customResult.output_shape[0]}</span>
                  </div>
                  {customResult.is_geotiff && (
                    <div className="flex justify-between text-[#8a8f98]">
                      <span>Geo CRS:</span>
                      <span className="text-[#f7f8f8] truncate max-w-[140px]" title={customResult.crs}>{customResult.crs}</span>
                    </div>
                  )}
                  <div className="flex justify-between text-[#8a8f98]">
                    <span>File Size:</span>
                    <span className="text-[#f7f8f8]">{customResult.file_size_mb} MB</span>
                  </div>
                  <div className="flex justify-between text-[#8a8f98]">
                    <span>Last Latency:</span>
                    <span className="text-[#5e6ad2] font-semibold">{customResult.latency_ms} ms</span>
                  </div>
                </>
              ) : (
                inferResult?.latency_ms && (
                  <div className="flex justify-between text-[#8a8f98]">
                    <span>Last Latency:</span>
                    <span className="text-[#5e6ad2]">{inferResult.latency_ms} ms</span>
                  </div>
                )
              )}
            </div>
          </div>

          {errorMsg && (
            <div className="mt-2 p-2 bg-[#eb5757]/10 border border-[#eb5757]/30 rounded text-[11px] text-[#eb5757] flex items-start space-x-1.5">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span className="break-all">{errorMsg}</span>
            </div>
          )}
        </div>

      </div>

      {/* Draggable Vertical Divider Handle */}
      <div
        onMouseDown={() => setIsDraggingLeft(true)}
        onDoubleClick={() => {
          setLeftWidth(340);
          localStorage.setItem('rcan_infer_left_panel_width', '340');
        }}
        className={`w-1.5 hover:w-2 hover:bg-[#5e6ad2]/70 cursor-col-resize rounded-full transition-all flex items-center justify-center group flex-shrink-0 ${
          isDraggingLeft ? 'bg-[#5e6ad2] w-2' : 'bg-white/[0.04]'
        }`}
        title="Drag to resize sidebar (Double-click to reset)"
      >
        <div className="h-8 w-0.5 bg-white/20 group-hover:bg-white/60 rounded-full" />
      </div>

      {/* Right Area: Interactive Viewer, Sub-tabs & Integrated Telemetry */}
      <div className="flex-1 h-full flex flex-col overflow-hidden min-w-0">
        
        {/* Top Control Bar: Sub-Tabs & Composite Mode */}
        <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-2 px-3 shadow-linear-card flex-shrink-0 mb-2 flex items-center justify-between">
          <div className="flex items-center p-0.5 bg-[#14151a] rounded-lg border border-white/[0.08] overflow-x-auto">
            <button
              onClick={() => setSubTab('slider')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                subTab === 'slider' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <Sliders className="w-3 h-3" />
              <span>Split Slider</span>
            </button>

            <button
              onClick={() => setSubTab('sidebyside')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                subTab === 'sidebyside' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <Layers className="w-3 h-3" />
              <span>Side-by-Side</span>
            </button>

            <button
              onClick={() => setSubTab('difference')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                subTab === 'difference' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <Flame className="w-3 h-3 text-[#f2994a]" />
              <span>Difference Map</span>
            </button>

            <button
              onClick={() => {
                setSubTab('zeroshot');
                if (!zeroShotResult && !isInferring) handleRunZeroShot();
              }}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                subTab === 'zeroshot' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <Sparkles className="w-3 h-3 text-[#4ebb78]" />
              <span>0-Shot 10m Mode</span>
            </button>

            <button
              onClick={() => setSubTab('metrics')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                subTab === 'metrics' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
              }`}
            >
              <BarChart3 className="w-3 h-3" />
              <span>Metrics Table</span>
            </button>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-[10px] text-[#62666d]">COMPOSITE:</span>
            <div className="flex items-center p-0.5 bg-[#14151a] rounded-lg border border-white/[0.08]">
              <button
                onClick={() => setColorMode('rgb')}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                  colorMode === 'rgb' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
                }`}
              >
                RGB True Color
              </button>
              <button
                onClick={() => setColorMode('cir')}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                  colorMode === 'cir' ? 'bg-[#1f2128] text-[#f7f8f8] border border-white/[0.08]' : 'text-[#8a8f98] hover:text-[#f7f8f8]'
                }`}
              >
                CIR False Color (NIR)
              </button>
            </div>
          </div>
        </div>

        {/* Central Viewport Filling Remaining Space */}
        <div className="flex-1 min-h-0 overflow-hidden mb-2">
          
          {/* Mode 1: Split Slider */}
          {subTab === 'slider' && (
            inputMode === 'custom' ? (
              customResult?.images ? (
                <div className="h-full w-full flex flex-col">
                  {/* Export Notification Bar */}
                  <div className="bg-[#14151a] border border-[#4ebb78]/30 rounded-lg p-2 px-3 mb-2 flex items-center justify-between shadow-sm flex-shrink-0">
                    <div className="flex items-center space-x-2 min-w-0">
                      <CheckCircle2 className="w-4 h-4 text-[#4ebb78] flex-shrink-0" />
                      <span className="text-xs font-semibold text-[#f7f8f8]">Super-Resolved & Exported</span>
                      <span className="text-[10px] px-1.5 py-0.5 bg-[#4ebb78]/20 text-[#4ebb78] rounded font-mono font-medium">
                        {customResult.format}
                      </span>
                      {customResult.is_geotiff && (
                        <span className="text-[10px] px-1.5 py-0.5 bg-[#5e6ad2]/20 text-[#38bdf8] rounded font-mono">
                          {customResult.crs}
                        </span>
                      )}
                      <span className="text-[11px] font-mono text-[#8a8f98] truncate" title={customResult.output_path}>
                        {customResult.output_path} ({customResult.file_size_mb} MB)
                      </span>
                    </div>
                    <button
                      onClick={handleCopyOutputPath}
                      className="px-2 py-1 rounded bg-[#1f2128] hover:bg-[#282b35] border border-white/[0.08] text-white text-[10px] font-mono flex items-center space-x-1 transition-all flex-shrink-0"
                    >
                      {copiedPath ? <Check className="w-3 h-3 text-[#4ebb78]" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedPath ? 'Copied' : 'Copy Path'}</span>
                    </button>
                  </div>

                  <div className="flex-1 min-h-0">
                    <ComparisonSlider
                      beforeImage={colorMode === 'cir' && customResult.images.before_cir ? customResult.images.before_cir : customResult.images.before_rgb}
                      afterImage={colorMode === 'cir' && customResult.images.after_cir ? customResult.images.after_cir : customResult.images.after_rgb}
                      beforeLabel={`Input Original (${customResult.input_shape[1]}x${customResult.input_shape[0]})`}
                      afterLabel={`HAT-Light Output (${customResult.output_shape[1]}x${customResult.output_shape[0]})`}
                    />
                  </div>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d] p-6 space-y-3">
                  <UploadCloud className="w-10 h-10 text-[#5e6ad2] opacity-50 mb-1" />
                  <p className="text-xs text-[#8a8f98] font-medium">Custom Image Tiled Super-Resolution Ready</p>
                  <p className="text-[11px] text-[#62666d] max-w-md text-center">
                    Drag and drop any JPEG, PNG, or GeoTIFF file in the left panel, configure tiling and export path, then click "Run Full Tiled Inference".
                  </p>
                  {customInputPath && (
                    <button
                      onClick={handleRunCustomInfer}
                      disabled={isInferring}
                      className="px-4 py-2 rounded-md bg-[#5e6ad2] hover:bg-[#6f7be2] text-white text-xs font-medium shadow-linear-btn flex items-center space-x-1.5"
                    >
                      <Zap className="w-3.5 h-3.5 fill-white" />
                      <span>Execute Full Tiled Inference Now</span>
                    </button>
                  )}
                </div>
              )
            ) : (
              inferResult?.images ? (
                <div className="h-full w-full">
                  <ComparisonSlider
                    beforeImage={colorMode === 'rgb' ? inferResult.images.lr_rgb : inferResult.images.lr_cir}
                    afterImage={colorMode === 'rgb' ? inferResult.images.sr_rgb : inferResult.images.sr_cir}
                    beforeLabel={`Bicubic Input (${inferResult.lr_shape[0]}x${inferResult.lr_shape[1]})`}
                    afterLabel={`HAT-Light Output (${inferResult.hr_shape[0]}x${inferResult.hr_shape[1]})`}
                  />
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d] p-6">
                  <Sliders className="w-10 h-10 text-[#8a8f98] opacity-30 mb-3" />
                  <p className="text-xs text-[#8a8f98] font-medium mb-1">Interactive Comparison Slider Ready</p>
                  <p className="text-[11px] text-[#62666d] mb-4">Click "Run Super-Resolution" to launch side-by-side interactive split.</p>
                  <button
                    onClick={handleRunStandard}
                    disabled={isInferring || !selectedPatch}
                    className="px-4 py-2 rounded-md bg-[#5e6ad2] hover:bg-[#6f7be2] text-white text-xs font-medium shadow-linear-btn flex items-center space-x-1.5"
                  >
                    <Zap className="w-3.5 h-3.5 fill-white" />
                    <span>Run Super-Resolution Now</span>
                  </button>
                </div>
              )
            )
          )}

          {/* Mode 2: Side-by-Side View */}
          {subTab === 'sidebyside' && (
            inputMode === 'custom' ? (
              customResult?.images ? (
                <div className="grid grid-cols-2 gap-2.5 h-full">
                  {/* 1. Original Input */}
                  <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 flex flex-col h-full overflow-hidden">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <span className="text-xs font-medium text-[#f2994a]">1. Original Input ({customResult.format})</span>
                      <span className="text-[10px] font-mono text-[#62666d]">{customResult.input_shape[1]}x{customResult.input_shape[0]}</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={colorMode === 'cir' && customResult.images.before_cir ? customResult.images.before_cir : customResult.images.before_rgb} alt="Input" className="w-full h-full object-contain" />
                    </div>
                  </div>

                  {/* 2. HAT-Light Super-Resolved */}
                  <div className="bg-[#0f1013] rounded-xl border border-[#5e6ad2]/50 p-3 flex flex-col h-full overflow-hidden shadow-lg shadow-[#5e6ad2]/10">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <span className="text-xs font-semibold text-[#5e6ad2]">2. HAT-Light Super-Resolved ({scale}x)</span>
                      <span className="text-[10px] font-mono text-[#4ebb78] font-semibold">{customResult.output_shape[1]}x{customResult.output_shape[0]}</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={colorMode === 'cir' && customResult.images.after_cir ? customResult.images.after_cir : customResult.images.after_rgb} alt="SR" className="w-full h-full object-contain" />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d]">
                  <Layers className="w-8 h-8 text-[#8a8f98] opacity-30 mb-2" />
                  <p className="text-xs">Run tiled inference to display side-by-side view.</p>
                </div>
              )
            ) : (
              inferResult?.images ? (
                <div className="grid grid-cols-3 gap-2.5 h-full">
                  {/* 1. Low-Res Input */}
                  <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 flex flex-col h-full overflow-hidden">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <span className="text-xs font-medium text-[#f2994a]">1. Low-Res Input (Bicubic)</span>
                      <span className="text-[10px] font-mono text-[#62666d]">{inferResult.lr_shape[0]}x{inferResult.lr_shape[1]}</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={colorMode === 'rgb' ? inferResult.images.lr_rgb : inferResult.images.lr_cir} alt="LR" className="w-full h-full object-contain" />
                    </div>
                  </div>

                  {/* 2. HAT-Light Super-Resolved */}
                  <div className="bg-[#0f1013] rounded-xl border border-[#5e6ad2]/50 p-3 flex flex-col h-full overflow-hidden shadow-lg shadow-[#5e6ad2]/10">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <span className="text-xs font-semibold text-[#5e6ad2]">2. HAT-Light Super-Resolved</span>
                      <span className="text-[10px] font-mono text-[#4ebb78] font-semibold">{inferResult.hr_shape[0]}x{inferResult.hr_shape[1]}</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={colorMode === 'rgb' ? inferResult.images.sr_rgb : inferResult.images.sr_cir} alt="SR" className="w-full h-full object-contain" />
                    </div>
                  </div>

                  {/* 3. Ground Truth HR */}
                  <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 flex flex-col h-full overflow-hidden">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <span className="text-xs font-medium text-[#4ebb78]">3. Ground Truth High-Res</span>
                      <span className="text-[10px] font-mono text-[#62666d]">{inferResult.hr_shape[0]}x{inferResult.hr_shape[1]}</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={colorMode === 'rgb' ? inferResult.images.hr_rgb : inferResult.images.hr_cir} alt="HR" className="w-full h-full object-contain" />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d]">
                  <Layers className="w-8 h-8 text-[#8a8f98] opacity-30 mb-2" />
                  <p className="text-xs">Run super-resolution to display side-by-side view.</p>
                </div>
              )
            )
          )}

          {/* Mode 3: High-Frequency Difference Heatmap */}
          {subTab === 'difference' && (
            inputMode === 'custom' ? (
              customResult?.images ? (
                <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 flex flex-col h-full overflow-hidden">
                  <div className="flex items-center justify-between mb-2 flex-shrink-0">
                    <div className="flex items-center space-x-1.5">
                      <Flame className="w-3.5 h-3.5 text-[#f2994a]" />
                      <span className="text-xs font-medium text-[#f7f8f8]">High-Frequency Detail Synthesis Map (|SR - Bicubic|)</span>
                    </div>
                    <span className="text-[10px] font-mono text-[#4ebb78]">Reconstructed Details</span>
                  </div>
                  <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                    <img src={customResult.images.diff_heatmap} alt="Difference Heatmap" className="w-full h-full object-contain" />
                  </div>
                  <div className="mt-2 text-[10px] text-[#62666d] text-center flex-shrink-0">
                    Warm colors (yellow/red) indicate high-frequency spatial structures and micro-textures synthesized by the HAT-Light model.
                  </div>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d]">
                  <Flame className="w-8 h-8 text-[#f2994a] opacity-30 mb-2" />
                  <p className="text-xs">Run tiled inference to compute difference maps.</p>
                </div>
              )
            ) : (
              inferResult?.images ? (
                <div className="grid grid-cols-2 gap-2.5 h-full">
                  {/* 1. Detail Synthesis Heatmap (|SR - Bicubic|) */}
                  <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 flex flex-col h-full overflow-hidden">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <div className="flex items-center space-x-1.5">
                        <Flame className="w-3.5 h-3.5 text-[#f2994a]" />
                        <span className="text-xs font-medium text-[#f7f8f8]">High-Frequency Synthesis Map (|SR - Bicubic|)</span>
                      </div>
                      <span className="text-[10px] font-mono text-[#4ebb78]">Reconstructed Details</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={inferResult.images.diff_heatmap} alt="Difference Heatmap" className="w-full h-full object-contain" />
                    </div>
                    <div className="mt-2 text-[10px] text-[#62666d] text-center flex-shrink-0">
                      Warm colors (yellow/red) highlight sub-pixel runway edges, building perimeters, and textures restored by HAT-Light.
                    </div>
                  </div>

                  {/* 2. Reconstruction Error Heatmap (|SR - HR|) */}
                  <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-3 flex flex-col h-full overflow-hidden">
                    <div className="flex items-center justify-between mb-2 flex-shrink-0">
                      <div className="flex items-center space-x-1.5">
                        <BarChart3 className="w-3.5 h-3.5 text-[#38bdf8]" />
                        <span className="text-xs font-medium text-[#f7f8f8]">Residual Error Map (|SR - GroundTruth|)</span>
                      </div>
                      <span className="text-[10px] font-mono text-[#8a8f98]">PSNR: {inferResult.metrics?.psnr} dB</span>
                    </div>
                    <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-white/[0.06] bg-black flex items-center justify-center">
                      <img src={inferResult.images.error_heatmap} alt="Error Heatmap" className="w-full h-full object-contain" />
                    </div>
                    <div className="mt-2 text-[10px] text-[#62666d] text-center flex-shrink-0">
                      Cool colors (dark blue/purple) verify ultra-low residual error across all 4 spectral bands.
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d]">
                  <Flame className="w-8 h-8 text-[#f2994a] opacity-30 mb-2" />
                  <p className="text-xs">Run super-resolution to compute high-frequency difference maps.</p>
                </div>
              )
            )
          )}

          {/* Mode 4: 0-Shot 10m Native Mode */}
          {subTab === 'zeroshot' && (
            zeroShotResult?.images ? (
              <div className="h-full w-full">
                <ComparisonSlider
                  beforeImage={colorMode === 'rgb' ? zeroShotResult.images.before_rgb : zeroShotResult.images.before_cir}
                  afterImage={colorMode === 'rgb' ? zeroShotResult.images.after_rgb : zeroShotResult.images.after_cir}
                  beforeLabel="Native 10m Sentinel-2"
                  afterLabel={`0-Shot HAT-Light (${zeroShotResult.output_resolution})`}
                />
              </div>
            ) : (
              <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d] space-y-3 p-6">
                <Sparkles className="w-10 h-10 text-[#4ebb78] opacity-50" />
                <p className="text-xs text-[#8a8f98] font-medium">0-Shot Native 10m Super-Resolution</p>
                <p className="text-[11px] text-[#62666d] max-w-md text-center">
                  Directly passes raw 10m Sentinel-2 reflectance into HAT-Light without prior downsampling to generate sub-meter simulated resolution.
                </p>
                <button
                  onClick={handleRunZeroShot}
                  disabled={isInferring || !selectedPatch}
                  className="py-2 px-4 rounded bg-[#4ebb78] hover:bg-[#5fc987] text-black font-semibold text-xs shadow-linear-btn flex items-center space-x-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Execute 0-Shot Now</span>
                </button>
              </div>
            )
          )}

          {/* Mode 5: Comprehensive Metrics Table */}
          {subTab === 'metrics' && (
            inputMode === 'custom' ? (
              customResult ? (
                <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-5 h-full overflow-y-auto">
                  <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.08]">
                    <div>
                      <h3 className="text-sm font-semibold text-[#f7f8f8]">File Inference Telemetry</h3>
                      <p className="text-xs text-[#8a8f98]">Execution details for full tiled inference</p>
                    </div>
                    <span className="px-2.5 py-1 rounded-md bg-[#5e6ad2]/15 border border-[#5e6ad2]/30 text-[#38bdf8] text-xs font-mono font-semibold">
                      {customResult.format} ({scale}x Upscaling)
                    </span>
                  </div>

                  <table className="w-full text-left text-xs text-[#8a8f98]">
                    <tbody className="divide-y divide-white/[0.04] font-mono text-xs">
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Input File</td>
                        <td className="py-3 px-3 text-[#f7f8f8]">{customResult.filename}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Input Dimensions</td>
                        <td className="py-3 px-3">{customResult.input_shape[1]} px (W) × {customResult.input_shape[0]} px (H)</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Super-Resolved Dimensions</td>
                        <td className="py-3 px-3 text-[#4ebb78] font-bold">{customResult.output_shape[1]} px (W) × {customResult.output_shape[0]} px (H)</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Exported Destination</td>
                        <td className="py-3 px-3 text-[#38bdf8] break-all">{customResult.output_path}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Export File Size</td>
                        <td className="py-3 px-3">{customResult.file_size_mb} MB</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Georeferencing (CRS)</td>
                        <td className="py-3 px-3">{customResult.crs}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Execution GPU Latency</td>
                        <td className="py-3 px-3 text-[#5e6ad2] font-semibold">{customResult.latency_ms} ms</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d]">
                  <BarChart3 className="w-8 h-8 text-[#8a8f98] opacity-30 mb-2" />
                  <p className="text-xs">Run tiled inference to view telemetry details.</p>
                </div>
              )
            ) : (
              inferResult?.metrics ? (
                <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-5 h-full overflow-y-auto">
                  <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.08]">
                    <div>
                      <h3 className="text-sm font-semibold text-[#f7f8f8]">Quantitative Evaluation Metrics</h3>
                      <p className="text-xs text-[#8a8f98]">Ground-truth verification for Sentinel-2 surface reflectance</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="px-2.5 py-1 rounded-md bg-[#4ebb78]/15 border border-[#4ebb78]/30 text-[#4ebb78] text-xs font-mono font-semibold">
                        +{inferResult.metrics.psnr_gain} dB Gain over Bicubic
                      </span>
                    </div>
                  </div>

                  <table className="w-full text-left text-xs text-[#8a8f98]">
                    <thead className="text-[10px] text-[#62666d] uppercase font-mono border-b border-white/[0.06]">
                      <tr>
                        <th className="py-2.5 px-3">Metric</th>
                        <th className="py-2.5 px-3">Bicubic Baseline</th>
                        <th className="py-2.5 px-3 text-[#5e6ad2]">HAT-Light Model</th>
                        <th className="py-2.5 px-3 text-[#4ebb78]">Fidelity Improvement</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04] font-mono text-xs">
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">PSNR (Full 4-Band)</td>
                        <td className="py-3 px-3">{inferResult.metrics.bicubic_psnr} dB</td>
                        <td className="py-3 px-3 text-[#5e6ad2] font-bold text-sm">{inferResult.metrics.psnr} dB</td>
                        <td className="py-3 px-3 text-[#4ebb78] font-semibold">+{inferResult.metrics.psnr_gain} dB</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Structural SSIM</td>
                        <td className="py-3 px-3">{inferResult.metrics.bicubic_ssim}</td>
                        <td className="py-3 px-3 text-[#5e6ad2] font-bold text-sm">{inferResult.metrics.ssim}</td>
                        <td className="py-3 px-3 text-[#4ebb78] font-semibold">+{(inferResult.metrics.ssim - inferResult.metrics.bicubic_ssim).toFixed(4)}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">Mean Absolute Error (MAE)</td>
                        <td className="py-3 px-3">--</td>
                        <td className="py-3 px-3 text-[#f7f8f8]">{inferResult.metrics.mae}</td>
                        <td className="py-3 px-3 text-[#62666d]">Reflectance Outlier Resistant</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">PSNR (RGB True Color)</td>
                        <td className="py-3 px-3">--</td>
                        <td className="py-3 px-3 text-[#f7f8f8]">{inferResult.metrics.psnr_rgb} dB</td>
                        <td className="py-3 px-3 text-[#62666d]">Channels 0, 1, 2 (B04, B03, B02)</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-3 font-medium text-[#f7f8f8]">PSNR (NIR Band B08)</td>
                        <td className="py-3 px-3">--</td>
                        <td className="py-3 px-3 text-[#f7f8f8]">{inferResult.metrics.psnr_nir} dB</td>
                        <td className="py-3 px-3 text-[#62666d]">Vegetation & Moisture Channel</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="h-full bg-[#0f1013] rounded-xl border border-dashed border-white/[0.08] flex flex-col items-center justify-center text-[#62666d]">
                  <BarChart3 className="w-8 h-8 text-[#8a8f98] opacity-30 mb-2" />
                  <p className="text-xs">Run inference to view metrics table.</p>
                </div>
              )
            )
          )}
        </div>

        {/* Bottom Telemetry Strip */}
        <div className="bg-[#0f1013] rounded-xl border border-white/[0.08] p-2.5 px-4 shadow-linear-card flex-shrink-0 flex items-center justify-between text-xs">
          {inputMode === 'custom' ? (
            customResult ? (
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center space-x-4">
                  <span className="text-[10px] text-[#62666d] uppercase">Format:</span>
                  <span className="font-mono text-[#f7f8f8] text-xs font-semibold">{customResult.format}</span>
                  <span className="text-[#62666d]">|</span>
                  <span className="text-[10px] text-[#62666d] uppercase">Resolution:</span>
                  <span className="font-mono text-[#f7f8f8] text-xs">{customResult.input_shape[1]}x{customResult.input_shape[0]}</span>
                  <span className="text-[#62666d]">→</span>
                  <span className="font-mono text-[#4ebb78] text-xs font-semibold">{customResult.output_shape[1]}x{customResult.output_shape[0]} ({scale}x)</span>
                </div>
                <div className="flex items-center space-x-4 font-mono text-xs">
                  <div className="flex items-center space-x-1.5">
                    <span className="text-[10px] text-[#62666d]">LATENCY:</span>
                    <span className="text-[#5e6ad2] font-semibold">{customResult.latency_ms} ms</span>
                  </div>
                  <span className="text-[#62666d]">|</span>
                  <div className="flex items-center space-x-1.5">
                    <span className="text-[10px] text-[#62666d]">SIZE:</span>
                    <span className="text-[#4ebb78] font-semibold">{customResult.file_size_mb} MB</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="w-full text-center text-xs text-[#62666d]">
                Ready for custom image inference. Drag and drop any JPEG, PNG, or GeoTIFF image file.
              </div>
            )
          ) : subTab === 'zeroshot' && zeroShotResult ? (
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center space-x-4">
                <span className="text-[10px] text-[#62666d] uppercase">0-Shot Resolution:</span>
                <span className="font-mono text-[#f7f8f8] text-xs font-semibold">{zeroShotResult.input_resolution}</span>
                <span className="text-[#62666d]">→</span>
                <span className="font-mono text-[#4ebb78] text-xs font-semibold">{zeroShotResult.output_resolution}</span>
              </div>
              <div className="flex items-center space-x-3 font-mono text-xs">
                <span className="text-[#62666d]">GPU LATENCY:</span>
                <span className="text-[#5e6ad2] font-semibold">{zeroShotResult.latency_ms} ms</span>
              </div>
            </div>
          ) : inferResult?.metrics ? (
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] text-[#62666d] uppercase">PSNR:</span>
                  <span className="font-mono text-[#4ebb78] text-xs font-bold">{inferResult.metrics.psnr} dB</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] text-[#62666d] uppercase">SSIM:</span>
                  <span className="font-mono text-[#f7f8f8] text-xs font-medium">{inferResult.metrics.ssim}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] text-[#62666d] uppercase">GAIN:</span>
                  <span className="font-mono text-[#5e6ad2] text-xs font-bold">+{inferResult.metrics.psnr_gain} dB</span>
                </div>
              </div>

              <div className="flex items-center space-x-4 font-mono text-xs">
                <div className="flex items-center space-x-1.5">
                  <span className="text-[10px] text-[#62666d]">LATENCY:</span>
                  <span className="text-[#f7f8f8] font-semibold">{inferResult.latency_ms} ms</span>
                </div>
                <span className="text-[#62666d]">|</span>
                <div className="flex items-center space-x-1.5">
                  <span className="text-[10px] text-[#62666d]">SCALE:</span>
                  <span className="text-[#38bdf8] font-semibold">{scale}x</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="w-full text-center text-xs text-[#62666d]">
              Ready for inference. Select patch and click "Run Super-Resolution" or "Run 0-Shot".
            </div>
          )}
        </div>

      </div>

    </div>
  );
}

