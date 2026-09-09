import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sliders, ZoomIn, ZoomOut, RotateCcw, Move } from 'lucide-react';

export default function ComparisonSlider({
  beforeImage,
  afterImage,
  beforeLabel = 'Bicubic Input',
  afterLabel = 'HAT-Light Super-Resolved'
}) {
  const [sliderPos, setSliderPos] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0 });

  const containerRef = useRef(null);
  const imageBoxRef = useRef(null);

  // Slider Drag Handler
  const updateSliderFromClientX = useCallback((clientX) => {
    if (!imageBoxRef.current) return;
    const rect = imageBoxRef.current.getBoundingClientRect();
    const relativeX = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (relativeX / rect.width) * 100));
    setSliderPos(percentage);
  }, []);

  const handleMouseDownSlider = (e) => {
    e.stopPropagation();
    setIsDraggingSlider(true);
    updateSliderFromClientX(e.clientX);
  };

  const handleTouchStartSlider = (e) => {
    e.stopPropagation();
    setIsDraggingSlider(true);
    if (e.touches && e.touches[0]) {
      updateSliderFromClientX(e.touches[0].clientX);
    }
  };

  // Pan Handlers (when zoomed in)
  const handleMouseDownPan = (e) => {
    if (zoom <= 1) return;
    if (e.button === 0) {
      setIsPanning(true);
      panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const handleWheelZoom = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.25 : -0.25;
    setZoom((prev) => {
      const nextZoom = Math.min(Math.max(1, +(prev + delta).toFixed(2)), 6);
      if (nextZoom === 1) setPan({ x: 0, y: 0 });
      return nextZoom;
    });
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDraggingSlider) {
        updateSliderFromClientX(e.clientX);
      } else if (isPanning && zoom > 1) {
        setPan({
          x: e.clientX - panStartRef.current.x,
          y: e.clientY - panStartRef.current.y
        });
      }
    };

    const handleTouchMove = (e) => {
      if (isDraggingSlider && e.touches && e.touches[0]) {
        updateSliderFromClientX(e.touches[0].clientX);
      }
    };

    const handleMouseUp = () => {
      setIsDraggingSlider(false);
      setIsPanning(false);
    };

    if (isDraggingSlider || isPanning) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDraggingSlider, isPanning, zoom, updateSliderFromClientX]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheelZoom, { passive: false });
    return () => el.removeEventListener('wheel', handleWheelZoom);
  }, [handleWheelZoom]);

  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSliderPos(50);
  };

  if (!beforeImage || !afterImage) {
    return (
      <div className="w-full h-full bg-[#0f1013] rounded-xl border border-white/[0.08] flex flex-col items-center justify-center text-[#62666d] p-6">
        <Sliders className="w-8 h-8 mb-2 opacity-30 text-[#8a8f98]" />
        <p className="text-xs">No inference data loaded yet. Run super-resolution to activate the comparison slider.</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col bg-[#0f1013] rounded-xl border border-white/[0.08] overflow-hidden shadow-linear-card select-none">
      {/* Top Toolbar */}
      <div className="px-3 py-2 bg-[#14151a] border-b border-white/[0.06] flex items-center justify-between text-xs text-[#8a8f98] flex-shrink-0">
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1.5 bg-[#0f1013] px-2 py-1 rounded-md border border-white/[0.08]">
            <Sliders className="w-3.5 h-3.5 text-[#5e6ad2]" />
            <span className="text-[11px] font-medium text-[#f7f8f8]">Interactive Split Slider</span>
          </div>
          <span className="text-[10px] text-[#62666d] hidden sm:inline">
            Drag divider left/right • Scroll or click zoom to inspect high-frequency detail
          </span>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center space-x-1.5">
          <div className="flex items-center bg-[#0f1013] p-0.5 rounded-lg border border-white/[0.08]">
            <button
              onClick={() => {
                const next = Math.max(1, +(zoom - 0.5).toFixed(1));
                setZoom(next);
                if (next === 1) setPan({ x: 0, y: 0 });
              }}
              disabled={zoom <= 1}
              className="p-1 rounded text-[#8a8f98] hover:text-[#f7f8f8] disabled:opacity-30 transition-all"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-1.5 text-[10px] font-mono text-[#f7f8f8] min-w-[36px] text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((prev) => Math.min(6, +(prev + 0.5).toFixed(1)))}
              disabled={zoom >= 6}
              className="p-1 rounded text-[#8a8f98] hover:text-[#f7f8f8] disabled:opacity-30 transition-all"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>

          {[1, 2, 4].map((level) => (
            <button
              key={level}
              onClick={() => {
                setZoom(level);
                if (level === 1) setPan({ x: 0, y: 0 });
              }}
              className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium transition-all ${
                zoom === level
                  ? 'bg-[#5e6ad2] text-white'
                  : 'bg-[#0f1013] text-[#8a8f98] border border-white/[0.08] hover:text-white'
              }`}
            >
              {level}x
            </button>
          ))}

          {(zoom > 1 || pan.x !== 0 || pan.y !== 0) && (
            <button
              onClick={handleResetView}
              className="p-1 rounded bg-[#0f1013] border border-white/[0.08] text-[#8a8f98] hover:text-[#f7f8f8] transition-all"
              title="Reset View"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport */}
      <div
        ref={containerRef}
        className={`flex-1 min-h-0 relative overflow-hidden bg-[#08090a] flex items-center justify-center ${
          zoom > 1 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : ''
        }`}
        onMouseDown={handleMouseDownPan}
      >
        <div
          ref={imageBoxRef}
          className="relative max-h-full max-w-full aspect-square shadow-2xl flex items-center justify-center overflow-hidden"
          style={{
            height: '92%',
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
            transformOrigin: 'center center',
            transition: isPanning ? 'none' : 'transform 0.08s ease-out'
          }}
        >
          {/* Layer 1: AFTER (Full) */}
          <img
            src={afterImage}
            alt={afterLabel}
            draggable={false}
            className="w-full h-full object-contain pointer-events-none select-none block"
          />

          {/* Layer 2: BEFORE (Clipped) */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none select-none"
            style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
          >
            <img
              src={beforeImage}
              alt={beforeLabel}
              draggable={false}
              className="w-full h-full object-contain pointer-events-none select-none block"
            />
          </div>

          {/* Split Divider Line & Handle */}
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-white pointer-events-none shadow-[0_0_10px_rgba(255,255,255,0.8)] z-20"
            style={{ left: `${sliderPos}%` }}
          >
            <div
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-[#14151a] border-2 border-white flex items-center justify-center shadow-2xl cursor-ew-resize pointer-events-auto active:scale-95 transition-transform group"
              onMouseDown={handleMouseDownSlider}
              onTouchStart={handleTouchStartSlider}
              title="Drag divider left or right"
            >
              <div className="flex space-x-0.5 items-center">
                <div className="w-0.5 h-3 bg-white rounded-full" />
                <div className="w-0.5 h-3 bg-white rounded-full" />
              </div>
            </div>
          </div>

          {/* Pill Badges */}
          <div className="absolute top-3 left-3 pointer-events-none z-10">
            <span className="px-2.5 py-1 rounded-md bg-[#08090a]/90 backdrop-blur-md text-[11px] font-medium text-[#f7f8f8] border border-white/[0.12] flex items-center space-x-1.5 shadow-lg">
              <span className="w-2 h-2 rounded-full bg-[#f2994a] shadow-[0_0_6px_#f2994a]" />
              <span>{beforeLabel}</span>
            </span>
          </div>

          <div className="absolute top-3 right-3 pointer-events-none z-10">
            <span className="px-2.5 py-1 rounded-md bg-[#08090a]/90 backdrop-blur-md text-[11px] font-medium text-[#f7f8f8] border border-white/[0.12] flex items-center space-x-1.5 shadow-lg">
              <span className="w-2 h-2 rounded-full bg-[#4ebb78] shadow-[0_0_6px_#4ebb78]" />
              <span className="font-semibold">{afterLabel}</span>
            </span>
          </div>

          {zoom > 1 && (
            <div className="absolute bottom-3 right-3 pointer-events-none z-10">
              <span className="px-2 py-1 rounded bg-black/80 backdrop-blur-sm text-[10px] font-mono text-[#8a8f98] border border-white/[0.08] flex items-center space-x-1">
                <Move className="w-3 h-3 text-[#38bdf8]" />
                <span>Click & drag to pan</span>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Footer Bar */}
      <div className="px-4 py-2 bg-[#0c0d10] border-t border-white/[0.06] flex items-center justify-between text-[11px] text-[#8a8f98] flex-shrink-0">
        <div className="flex items-center space-x-3">
          <span className="text-[#62666d]">Reveal Ratio:</span>
          <span className="font-mono text-xs text-[#f7f8f8] font-medium">
            {Math.round(sliderPos)}% <span className="text-[#62666d]">LR Bicubic</span> / {100 - Math.round(sliderPos)}% <span className="text-[#4ebb78]">HAT-Light SR</span>
          </span>
        </div>

        <div className="flex items-center space-x-3">
          {zoom > 1 && (
            <span className="text-[10px] font-mono text-[#38bdf8]">
              Zoom: {Math.round(zoom * 100)}%
            </span>
          )}
          <span className="text-[10px] text-[#62666d]">
            Full Sentinel-2 dynamic range preserved
          </span>
        </div>
      </div>
    </div>
  );
}
