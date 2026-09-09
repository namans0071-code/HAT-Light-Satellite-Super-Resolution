import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Layers } from 'lucide-react';

export default function TelemetryCharts({ history = {} }) {
  const epochs = history.epochs || [];
  const trainLoss = history.train_loss || [];
  const valLoss = history.val_loss || [];
  const l1Loss = history.l1_loss || [];
  const ssimLoss = history.ssim_loss || [];
  const psnrAll = history.psnr_all || [];
  const psnrGain = history.psnr_gain || [];
  const psnrRgb = history.psnr_rgb || [];
  const psnrNir = history.psnr_nir || [];
  const lrs = history.learning_rate || [];

  const fftLoss = (history.fft_loss && history.fft_loss.length > 0) ? history.fft_loss : (history.edge_loss || []);
  const gradLoss = (history.grad_loss && history.grad_loss.length > 0) ? history.grad_loss : (history.l1_loss || []);

  // Base theme optimized for compact 6-grid display
  const baseChartTheme = (title, unit = '') => ({
    backgroundColor: '#0f1013',
    textStyle: { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' },
    title: {
      text: title,
      left: 10,
      top: 6,
      textStyle: { color: '#f7f8f8', fontSize: 11, fontWeight: '600' }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#14151a',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      padding: [4, 8],
      textStyle: { color: '#f7f8f8', fontSize: 10 },
      axisPointer: { type: 'cross', lineStyle: { color: 'rgba(255, 255, 255, 0.15)', width: 1, type: 'dashed' } }
    },
    grid: { left: 45, right: 15, bottom: 22, top: 32 },
    xAxis: {
      type: 'category',
      data: epochs.map(e => `${e}`),
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.08)' } },
      axisLabel: { color: '#62666d', fontSize: 9 },
      splitLine: { show: false }
    },
    yAxis: {
      type: 'value',
      name: unit,
      nameTextStyle: { color: '#62666d', fontSize: 9 },
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.08)' } },
      axisLabel: { color: '#8a8f98', fontSize: 9 },
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.04)' } }
    }
  });

  // 1. Total Loss (Exponential Decay)
  const chart1 = {
    ...baseChartTheme('Total Loss (Train vs Val)'),
    legend: { data: ['Train', 'Val'], textStyle: { color: '#8a8f98', fontSize: 10 }, top: 5, right: 10, itemHeight: 8, itemWidth: 12 },
    series: [
      {
        name: 'Train',
        type: 'line',
        smooth: true,
        data: trainLoss,
        itemStyle: { color: '#5e6ad2' },
        lineStyle: { width: 1.8 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(94, 106, 210, 0.25)' }, { offset: 1, color: 'rgba(94, 106, 210, 0.0)' }]
          }
        }
      },
      {
        name: 'Val',
        type: 'line',
        smooth: true,
        data: valLoss,
        itemStyle: { color: '#f2994a' },
        lineStyle: { width: 1.5, type: 'dashed' }
      }
    ]
  };

  // 2. Fourier FFT Frequency Loss (0.1x)
  const chart2 = {
    ...baseChartTheme('Fourier FFT Frequency Loss (0.1x)'),
    series: [
      {
        name: 'FFT Loss',
        type: 'line',
        smooth: true,
        data: fftLoss,
        itemStyle: { color: '#4ebb78' },
        lineStyle: { width: 1.8 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(78, 187, 120, 0.2)' }, { offset: 1, color: 'rgba(78, 187, 120, 0.0)' }]
          }
        }
      }
    ]
  };

  // 3. Spatial Gradient Loss (0.05x)
  const chart3 = {
    ...baseChartTheme('Spatial Gradient Loss (0.05x)'),
    series: [
      {
        name: 'Gradient Loss',
        type: 'line',
        smooth: true,
        data: gradLoss,
        itemStyle: { color: '#ec4899' },
        lineStyle: { width: 1.8 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(236, 72, 153, 0.2)' }, { offset: 1, color: 'rgba(236, 72, 153, 0.0)' }]
          }
        }
      }
    ]
  };

  // 4. PSNR Overall & Gain (Diminishing Returns)
  const chart4 = {
    ...baseChartTheme('Overall PSNR & Gain', 'dB'),
    legend: { data: ['PSNR', 'Gain'], textStyle: { color: '#8a8f98', fontSize: 10 }, top: 5, right: 10, itemHeight: 8, itemWidth: 12 },
    series: [
      {
        name: 'PSNR',
        type: 'line',
        smooth: true,
        data: psnrAll,
        itemStyle: { color: '#4ebb78' },
        lineStyle: { width: 2 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(78, 187, 120, 0.2)' }, { offset: 1, color: 'rgba(78, 187, 120, 0.0)' }]
          }
        }
      },
      {
        name: 'Gain',
        type: 'line',
        smooth: true,
        data: psnrGain,
        itemStyle: { color: '#5e6ad2' },
        lineStyle: { width: 1.5, type: 'dotted' }
      }
    ]
  };

  // 5. Spectral Bands PSNR (RGB vs NIR) (Diminishing Returns)
  const chart5 = {
    ...baseChartTheme('Spectral Bands PSNR (RGB vs NIR)', 'dB'),
    legend: { data: ['RGB', 'NIR (B8)'], textStyle: { color: '#8a8f98', fontSize: 10 }, top: 5, right: 10, itemHeight: 8, itemWidth: 12 },
    series: [
      {
        name: 'RGB',
        type: 'line',
        smooth: true,
        data: psnrRgb,
        itemStyle: { color: '#38bdf8' },
        lineStyle: { width: 1.8 }
      },
      {
        name: 'NIR (B8)',
        type: 'line',
        smooth: true,
        data: psnrNir,
        itemStyle: { color: '#f2994a' },
        lineStyle: { width: 1.8 }
      }
    ]
  };

  // 6. Learning Rate & Optimization Trajectory
  const chart6 = {
    ...baseChartTheme('Learning Rate Schedule'),
    series: [
      {
        name: 'LR',
        type: 'line',
        smooth: true,
        data: lrs,
        itemStyle: { color: '#8a8f98' },
        lineStyle: { width: 1.8 }
      }
    ]
  };

  if (epochs.length === 0) {
    return (
      <div className="h-full bg-[#0f1013] rounded-xl border border-white/[0.08] p-4 flex flex-col items-center justify-center text-[#62666d] space-y-2">
        <Layers className="w-8 h-8 opacity-30 text-[#8a8f98]" />
        <span className="text-xs">No training telemetry yet. Start training to plot the 6 live curves simultaneously.</span>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 grid-rows-2 gap-2.5 h-full w-full">
      <div className="bg-[#0f1013] rounded-lg border border-white/[0.08] overflow-hidden p-1 shadow-sm">
        <ReactECharts option={chart1} style={{ height: '100%', width: '100%' }} notMerge={true} lazyUpdate={true} />
      </div>
      <div className="bg-[#0f1013] rounded-lg border border-white/[0.08] overflow-hidden p-1 shadow-sm">
        <ReactECharts option={chart2} style={{ height: '100%', width: '100%' }} notMerge={true} lazyUpdate={true} />
      </div>
      <div className="bg-[#0f1013] rounded-lg border border-white/[0.08] overflow-hidden p-1 shadow-sm">
        <ReactECharts option={chart3} style={{ height: '100%', width: '100%' }} notMerge={true} lazyUpdate={true} />
      </div>
      <div className="bg-[#0f1013] rounded-lg border border-white/[0.08] overflow-hidden p-1 shadow-sm">
        <ReactECharts option={chart4} style={{ height: '100%', width: '100%' }} notMerge={true} lazyUpdate={true} />
      </div>
      <div className="bg-[#0f1013] rounded-lg border border-white/[0.08] overflow-hidden p-1 shadow-sm">
        <ReactECharts option={chart5} style={{ height: '100%', width: '100%' }} notMerge={true} lazyUpdate={true} />
      </div>
      <div className="bg-[#0f1013] rounded-lg border border-white/[0.08] overflow-hidden p-1 shadow-sm">
        <ReactECharts option={chart6} style={{ height: '100%', width: '100%' }} notMerge={true} lazyUpdate={true} />
      </div>
    </div>
  );
}
