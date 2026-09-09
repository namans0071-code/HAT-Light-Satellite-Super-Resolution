import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header.jsx';
import SidebarControls from './components/SidebarControls.jsx';
import LinearComparisonSlider from './components/LinearComparisonSlider.jsx';
import DifferenceHeatmap from './components/DifferenceHeatmap.jsx';
import ExportModal from './components/ExportModal.jsx';
import { inferenceService } from './services/inferenceService.js';

export default function App() {
  const [activeMode, setActiveMode] = useState('slider'); // 'slider' | 'sidebyside' | 'diff'
  const [inputMode, setInputMode] = useState('curated'); // 'curated' | 'custom'
  
  // Real dataset state
  const [patches, setPatches] = useState([]);
  const [categories, setCategories] = useState({});
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoadingPatches, setIsLoadingPatches] = useState(true);

  // Selected sample & execution parameters (Locked 4x scale benchmark)
  const [selectedSampleId, setSelectedSampleId] = useState('patch_airport_001428.npy');
  const [customFile, setCustomFile] = useState(null);
  const [customPath, setCustomPath] = useState('');
  const scale = 4.0;
  const [colorMode, setColorMode] = useState('rgb');

  const [backendStatus, setBackendStatus] = useState(null);
  const [isInferring, setIsInferring] = useState(false);
  const [result, setResult] = useState(null);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  // Load real patches from backend
  const loadPatches = useCallback(async (cat, search) => {
    setIsLoadingPatches(true);
    const data = await inferenceService.getTestPatches(cat, search, 80);
    setPatches(data.patches || []);
    setCategories(data.categories || {});
    setIsLoadingPatches(false);
    return data.patches || [];
  }, []);

  // Initial load
  useEffect(() => {
    inferenceService.checkHealth().then(setBackendStatus);
    loadPatches('all', '').then((loaded) => {
      if (loaded.length > 0) {
        setSelectedSampleId(loaded[0].id);
      }
    });
  }, [loadPatches]);

  // Refetch patches when category or search changes
  useEffect(() => {
    const timer = setTimeout(() => {
      loadPatches(selectedCategory, searchQuery);
    }, 150);
    return () => clearTimeout(timer);
  }, [selectedCategory, searchQuery, loadPatches]);

  // Run real inference whenever sample or parameters change
  const executeInference = useCallback(async () => {
    setIsInferring(true);
    try {
      if (inputMode === 'custom') {
        if (customFile) {
          const res = await inferenceService.uploadAndInfer(customFile, 4.0, colorMode);
          setResult(res);
        } else if (customPath && customPath.trim()) {
          const res = await inferenceService.runInference({
            patchPath: customPath.trim(),
            inputMode: 'custom',
            scale: 4.0,
            colorMode
          });
          setResult(res);
        }
      } else {
        const res = await inferenceService.runInference({
          patchPath: selectedSampleId,
          inputMode: 'curated',
          scale: 4.0,
          colorMode
        });
        setResult(res);
      }
    } catch (e) {
      console.error('Inference error:', e);
    } finally {
      setIsInferring(false);
    }
  }, [inputMode, customFile, customPath, selectedSampleId, colorMode]);

  // Trigger inference when parameters change
  useEffect(() => {
    if (inputMode === 'curated' && selectedSampleId) {
      executeInference();
    } else if (inputMode === 'custom' && (customFile || (customPath && customPath.trim()))) {
      executeInference();
    }
  }, [inputMode, selectedSampleId, customFile, customPath, colorMode, executeInference]);

  const currentSample = (inputMode === 'custom' && (customFile || customPath)) ? {
    id: customFile ? customFile.name : customPath,
    name: customFile ? customFile.name : (customPath.split(/[\\/]/).pop() || customPath)
  } : (patches.find(s => s.id === selectedSampleId || s.stem === selectedSampleId) || {
    id: selectedSampleId || 'patch_airport_001428.npy',
    name: 'Selected Satellite Scene'
  });

  return (
    <div className="flex flex-col h-screen w-screen bg-[#08090a] text-[#f7f8f8] overflow-hidden bg-linear-grid">
      {/* Linear Inspired Top Navigation Header */}
      <Header 
        backendStatus={backendStatus}
        onOpenExport={() => setIsExportModalOpen(true)}
        activeMode={activeMode}
        setActiveMode={setActiveMode}
      />

      {/* Main Studio Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Interactive Control Panel */}
        <SidebarControls 
          inputMode={inputMode}
          setInputMode={setInputMode}
          patches={patches}
          categories={categories}
          selectedCategory={selectedCategory}
          setSelectedCategory={setSelectedCategory}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          selectedSampleId={selectedSampleId}
          setSelectedSampleId={setSelectedSampleId}
          customFile={customFile}
          setCustomFile={setCustomFile}
          customPath={customPath}
          setCustomPath={setCustomPath}
          colorMode={colorMode}
          setColorMode={setColorMode}
          onRunInference={executeInference}
          isInferring={isInferring}
          isLoadingPatches={isLoadingPatches}
        />

        {/* Center Canvas Viewport */}
        <main className="flex-1 p-4 overflow-hidden flex flex-col items-center justify-center">
          {result && (
            activeMode === 'diff' ? (
              <DifferenceHeatmap 
                heatmapImage={result.images.differenceHeatmap}
                metrics={result.metrics}
              />
            ) : activeMode === 'sidebyside' ? (
              <div className="grid grid-cols-2 gap-4 w-full h-full">
                <div className="flex flex-col bg-[#0f1011] rounded-xl border border-[#232529] overflow-hidden">
                  <div className="h-9 border-b border-[#232529] px-3 flex items-center justify-between text-xs font-mono text-[#8a8f98]">
                    <span>Native Sentinel-2 Input (10m)</span>
                    <span className="text-[10px] text-[#525660]">{colorMode.toUpperCase()}</span>
                  </div>
                  <div className="flex-1 p-4 flex items-center justify-center bg-[#0a0b0d]">
                    <img 
                      src={result.images.lrInput || result.images.bicubic} 
                      alt="Native Sentinel-2 Input" 
                      className="max-h-full object-contain rounded" 
                    />
                  </div>
                </div>
                <div className="flex flex-col bg-[#0f1011] rounded-xl border border-[#232529] overflow-hidden">
                  <div className="h-9 border-b border-[#232529] px-3 flex items-center justify-between text-xs font-mono text-emerald-400">
                    <span>2.5m HAT-Light Super-Resolved</span>
                    <span className="text-[10px] text-emerald-400 font-semibold font-mono">
                      {result.metrics?.outputResolution || '512x512'}
                    </span>
                  </div>
                  <div className="flex-1 p-4 flex items-center justify-center bg-[#0a0b0d]">
                    <img 
                      src={result.images.superResolved} 
                      alt="2.5m HAT-Light SR" 
                      className="max-h-full object-contain rounded" 
                    />
                  </div>
                </div>
              </div>
            ) : (
              <LinearComparisonSlider 
                beforeImage={result.images.lrInput || result.images.bicubic}
                afterImage={result.images.superResolved}
                beforeLabel={`Native Input (${colorMode.toUpperCase()})`}
                afterLabel={`HAT-Light SR (${colorMode.toUpperCase()})`}
                metrics={result.metrics}
                scale={scale}
              />
            )
          )}
        </main>
      </div>

      {/* Export Dialog Popup */}
      <ExportModal 
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        currentSample={currentSample}
        scale={scale}
      />
    </div>
  );
}
