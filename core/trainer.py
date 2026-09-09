"""
Enhanced Satellite Super-Resolution Training Supervisor for HAT-Light.
Features:
- Lightweight Hybrid Attention Transformer (HAT-Light) with Continuous Scale Conditioning.
- Loss Function: L = L1 + 0.1 * L_FFT + 0.05 * L_gradient (No 3-channel VGG perceptual loss).
- Mixed Precision (torch.cuda.amp.autocast(dtype=torch.float16)).
- Gradient Accumulation (configurable steps = 1 or 2).
- Early Stopping (configurable patience & min delta).
- Automatic Multi-Patch Visual Comparison export each time a new best model is saved.
- Automatic 6-Graph Telemetry Curves export (training_curves_latest.png).
"""

import os
import sys
import time
import math
import shutil
import threading
import gc
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.hat_light import HATLightSR, check_cuda_device
from core.dataset import (
    SatelliteSRDataset,
    make_rgb_composite,
    make_cir_composite,
    denormalize_patch,
    normalize_patch
)
from core.losses import CompositeSatelliteLoss, calculate_band_metrics, calculate_psnr


class TrainingState:
    IDLE = "IDLE"
    TRAINING = "TRAINING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FINISHED = "FINISHED"
    EARLY_STOPPED = "EARLY_STOPPED"
    ERROR = "ERROR"


class SRTrainer:
    """Supervisor for training HAT-Light with telemetry & artifact exports."""
    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.device = check_cuda_device()
        self.callback = callback

        self.state = TrainingState.IDLE
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.model: Optional[HATLightSR] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion = CompositeSatelliteLoss().to(self.device)

        # Progress tracking
        self.current_epoch = 0
        self.total_epochs = 75
        self.batch_size = 16
        self.grad_accum_steps = 1
        self.learning_rate = 2e-4
        self.scale = 4.0
        self.best_psnr = -1.0
        self.bicubic_baseline_psnr = 0.0
        self.export_dir = Path("model/checkpoints")

        # Early Stopping
        self.early_stopping_enabled = True
        self.patience = 15
        self.min_delta = 0.01
        self.epochs_without_improvement = 0

        # Telemetry history
        self.history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "l1_loss": [],
            "fft_loss": [],
            "edge_loss": [],
            "grad_loss": [],
            "sam_loss": [],
            "ssim_loss": [],
            "psnr_all": [],
            "psnr_gain": [],
            "psnr_rgb": [],
            "psnr_nir": [],
            "ssim_all": [],
            "learning_rate": []
        }

    def start_training(
        self,
        data_train_dir: str = "Dataset/data/Train",
        data_val_dir: str = "Dataset/data/Val",
        export_dir: str = "model/checkpoints",
        checkpoint_path: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        scale: Union[int, float] = 4.0,
        grad_accum_steps: int = 1,
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 0.02
    ):
        if self.state == TrainingState.TRAINING:
            return {"status": "error", "message": "Training is already running."}

        self.state = TrainingState.TRAINING
        self._stop_event.clear()
        self._pause_event.set()

        self.total_epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scale = float(scale)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.early_stopping_enabled = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.export_dir = Path(export_dir).resolve()
        self.export_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Trainer] Outputs exclusively configured for directory: {self.export_dir}")

        self._thread = threading.Thread(
            target=self._run_training_loop,
            args=(data_train_dir, data_val_dir, checkpoint_path),
            daemon=True
        )
        self._thread.start()
        return {"status": "ok", "message": "Training started."}

    def pause_training(self):
        if self.state == TrainingState.TRAINING:
            self.state = TrainingState.PAUSED
            self._pause_event.clear()
            self._save_checkpoint("paused_checkpoint.pth")
            self._emit({"type": "state_change", "state": self.state, "message": "Training paused."})
            return {"status": "ok", "message": "Training paused."}
        return {"status": "ignored", "message": f"Cannot pause from state {self.state}."}

    def resume_training(self):
        if self.state == TrainingState.PAUSED:
            self.state = TrainingState.TRAINING
            self._pause_event.set()
            self._emit({"type": "state_change", "state": self.state, "message": "Training resumed."})
            return {"status": "ok", "message": "Training resumed."}
        return {"status": "ignored", "message": f"Cannot resume from state {self.state}."}

    def _cleanup_gpu_memory(self):
        """Immediately offloads model to CPU, deletes optimizer, and releases all CUDA cache."""
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                del self.optimizer
                self.optimizer = None
            if hasattr(self, "scheduler") and self.scheduler is not None:
                del self.scheduler
                self.scheduler = None
            if hasattr(self, "scaler") and self.scaler is not None:
                del self.scaler
                self.scaler = None
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                mem_mb = torch.cuda.memory_reserved(0) / (1024**2)
                print(f"[Trainer] VRAM released. Active GPU VRAM: {mem_mb:.1f} MB")
        except Exception as e:
            print(f"[Trainer] Memory cleanup notice: {e}")

    def stop_training(self):
        if self.state in [TrainingState.TRAINING, TrainingState.PAUSED]:
            self.state = TrainingState.STOPPED
            self._stop_event.set()
            self._pause_event.set()
            self._save_checkpoint("stopped_checkpoint.pth")
            if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
                try:
                    self._thread.join(timeout=1.5)
                except Exception:
                    pass
            self._cleanup_gpu_memory()
            self._emit({"type": "state_change", "state": self.state, "message": "Training stopped by user. VRAM released."})
            return {"status": "ok", "message": "Training stopped. VRAM released."}
        return {"status": "ignored", "message": "No active training to stop."}

    def _save_checkpoint(self, filename: str) -> str:
        if not self.model:
            return ""
        save_path = self.export_dir / filename
        state_dict = {
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "scale": self.scale,
            "architecture": "HAT-Sat-Pro",
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_psnr": self.best_psnr,
            "history": self.history
        }
        torch.save(state_dict, save_path)
        return str(save_path)

    def _load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            print(f"[Trainer] Checkpoint not found at {checkpoint_path}. Training from clean weights.")
            return

        print(f"[Trainer] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[Trainer] Loaded model state from {checkpoint_path}")
        except Exception as e:
            print(f"[Trainer Notice] Non-exact key match ({e}). Loading matching weights...")
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

        if isinstance(ckpt, dict):
            if "epoch" in ckpt:
                self.current_epoch = ckpt["epoch"]
            if "best_psnr" in ckpt:
                self.best_psnr = ckpt["best_psnr"]
            if "history" in ckpt and ckpt["history"]:
                self.history = ckpt["history"]
            if self.optimizer and "optimizer_state" in ckpt and ckpt["optimizer_state"]:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception:
                    pass
            print(f"[Trainer] Resumed from {checkpoint_path} at epoch {self.current_epoch} (Best PSNR: {self.best_psnr:.2f} dB)")

    def _emit(self, event_data: Dict[str, Any]):
        if self.callback:
            try:
                self.callback(event_data)
            except Exception as e:
                print(f"[Callback Error] {e}")

    def _run_training_loop(self, train_dir: str, val_dir: str, checkpoint_path: Optional[str]):
        try:
            print(f"\n" + "=" * 75)
            print(f"  Lightweight Hybrid Attention Transformer (HAT-Light) Training Loop (Scale: {self.scale}x)")
            # Auto-detect dimensions from checkpoint if available, else use upgraded high-capacity defaults
            embed_dim = 96
            num_rhag = 6
            num_heads = 8
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    ckpt_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    sd = ckpt_preview.get("model_state", ckpt_preview) if isinstance(ckpt_preview, dict) else ckpt_preview
                    if isinstance(sd, dict) and "input_layer.0.weight" in sd:
                        embed_dim = sd["input_layer.0.weight"].shape[0]
                    rhag_idx = [int(k.split('.')[1]) for k in sd.keys() if k.startswith("rhags.") and k.split('.')[1].isdigit()]
                    if rhag_idx:
                        num_rhag = max(rhag_idx) + 1
                    if embed_dim % 8 == 0:
                        num_heads = 8
                    elif embed_dim % 6 == 0:
                        num_heads = 6
                except Exception as e:
                    print(f"[Trainer] Checkpoint dimension scan: {e}")

            print(f"  Target Device: {torch.cuda.get_device_name(0)}")
            print(f"  Architecture: HAT-Light • {num_rhag} RHAG Blocks • {num_heads} Heads • Embedding Dim {embed_dim}")
            print(f"  Loss Suite: L1 (1.0) + Fourier FFT (0.15) + Spatial Gradient (0.08)")
            print(f"  Gradient Accumulation Steps: {self.grad_accum_steps} (Effective Batch Size: {self.batch_size * self.grad_accum_steps})")
            print(f"  Early Stopping: {'Enabled (Patience=' + str(self.patience) + ')' if self.early_stopping_enabled else 'Disabled'}")
            print(f"  Output Directory: {self.export_dir} (Exclusive)")
            print("=" * 75 + "\n")

            # 1. Instantiate HAT-Light Network
            self.model = HATLightSR(
                in_channels=4,
                out_channels=4,
                scale=self.scale,
                embed_dim=embed_dim,
                num_rhag=num_rhag,
                num_hab_per_rhag=4,
                num_heads=num_heads,
                window_size=8
            ).to(self.device)

            self.criterion = CompositeSatelliteLoss(
                weight_l1=1.0,
                weight_fft=0.15,
                weight_grad=0.08,
                channels=4
            ).to(self.device)

            self.scaler = torch.amp.GradScaler("cuda")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.99),
                weight_decay=1e-4
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_epochs,
                eta_min=1e-6
            )

            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)

            train_dataset = SatelliteSRDataset(train_dir, scale=int(self.scale), is_train=True, augment=True)
            val_dataset = SatelliteSRDataset(val_dir, scale=int(self.scale), is_train=False, augment=False, max_samples=400)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            num_batches = len(train_loader)
            start_epoch = self.current_epoch + 1

            # Header format
            header = f"{'Epoch':>10} {'GPU_mem':>10} {'L1_loss':>10} {'FFT_loss':>10} {'Grad_loss':>10} {'Total':>10} {'PSNR':>10} {'SSIM':>10} {'Instances':>10} {'Size':>10}"
            print(header)

            for epoch in range(start_epoch, self.total_epochs + 1):
                if self._stop_event.is_set():
                    break

                self.current_epoch = epoch
                self.model.train()

                epoch_start_time = time.time()
                running_l1 = 0.0
                running_fft = 0.0
                running_grad = 0.0
                running_total = 0.0

                self.optimizer.zero_grad(set_to_none=True)

                for batch_idx, (lr, hr) in enumerate(train_loader, start=1):
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        break

                    lr = lr.to(self.device, non_blocking=True)
                    hr = hr.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        sr = self.model(lr, scale=self.scale)
                        loss_dict = self.criterion(sr, hr)
                        total_loss = loss_dict["total"]
                        loss_scaled = total_loss / self.grad_accum_steps

                    self.scaler.scale(loss_scaled).backward()

                    if (batch_idx % self.grad_accum_steps == 0) or (batch_idx == num_batches):
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    l1_v = loss_dict["l1"].item()
                    fft_v = loss_dict["fft"].item()
                    grad_v = loss_dict["grad"].item()
                    tot_v = total_loss.item()

                    running_l1 += l1_v
                    running_fft += fft_v
                    running_grad += grad_v
                    running_total += tot_v

                    now = time.time()
                    elapsed = now - epoch_start_time
                    it_per_sec = batch_idx / max(1e-4, elapsed)
                    eta_sec = (num_batches - batch_idx) / max(1e-4, it_per_sec)

                    gpu_mem = f"{torch.cuda.memory_reserved(0) / (1024**3):.2f}G"
                    avg_l1 = running_l1 / batch_idx
                    avg_fft = running_fft / batch_idx
                    avg_grad = running_grad / batch_idx
                    avg_tot = running_total / batch_idx

                    # Format formatted progress line
                    pct = int((batch_idx / max(1, num_batches)) * 100)
                    progress_line = (
                        f"{epoch:>5}/{self.total_epochs:<4} {gpu_mem:>10} {avg_l1:>10.4f} "
                        f"{avg_fft:>10.4f} {avg_grad:>10.4f} {avg_tot:>10.4f} {'--':>10} {'--':>10} "
                        f"{lr.size(0):>10} {hr.size(2)}x{hr.size(3)}: {pct:3d}% [{int(elapsed//60):02d}:{int(elapsed%60):02d}<{int(eta_sec//60):02d}:{int(eta_sec%60):02d}, {it_per_sec:4.2f}it/s]"
                    )
                    sys.stdout.write(f"\r{progress_line}")
                    sys.stdout.flush()

                    if batch_idx % 2 == 0 or batch_idx == num_batches:
                        self._emit({
                            "type": "batch_progress",
                            "epoch": epoch,
                            "total_epochs": self.total_epochs,
                            "batch": batch_idx,
                            "total_batches": num_batches,
                            "pct": pct,
                            "gpu_mem": gpu_mem,
                            "l1_loss": avg_l1,
                            "charbonnier_loss": avg_l1,
                            "edge_loss": avg_fft,
                            "fft_loss": avg_fft,
                            "grad_loss": avg_grad,
                            "sam_loss": 0.0,
                            "ssim_loss": 0.0,
                            "total_loss": avg_tot,
                            "it_per_sec": round(it_per_sec, 2),
                            "elapsed_str": f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",
                            "eta_str": f"{int(eta_sec//60):02d}:{int(eta_sec%60):02d}",
                            "line": progress_line
                        })

                if self._stop_event.is_set():
                    break

                # Validation & Metrics Calculation
                val_metrics = self._validate(val_loader)
                self.scheduler.step()
                curr_lr = self.scheduler.get_last_lr()[0]

                # Update Telemetry History
                epoch_train_loss = round(running_total / max(1, num_batches), 5)
                epoch_l1 = round(running_l1 / max(1, num_batches), 5)
                epoch_fft = round(running_fft / max(1, num_batches), 5)
                epoch_grad = round(running_grad / max(1, num_batches), 5)
                psnr_gain = round(val_metrics["psnr"] - val_metrics["bicubic_psnr"], 2)

                self.history.setdefault("epochs", []).append(epoch)
                self.history.setdefault("train_loss", []).append(epoch_train_loss)
                self.history.setdefault("val_loss", []).append(round(val_metrics["val_loss"], 5))
                self.history.setdefault("l1_loss", []).append(epoch_l1)
                self.history.setdefault("fft_loss", []).append(epoch_fft)
                self.history.setdefault("edge_loss", []).append(epoch_fft)
                self.history.setdefault("grad_loss", []).append(epoch_grad)
                self.history.setdefault("sam_loss", []).append(0.0)
                self.history.setdefault("ssim_loss", []).append(round(1.0 - val_metrics["ssim"], 4))
                self.history.setdefault("psnr_all", []).append(round(val_metrics["psnr"], 2))
                self.history.setdefault("psnr_gain", []).append(psnr_gain)
                self.history.setdefault("psnr_rgb", []).append(round(val_metrics["psnr_rgb"], 2))
                self.history.setdefault("psnr_nir", []).append(round(val_metrics["psnr_nir"], 2))
                self.history.setdefault("ssim_all", []).append(round(val_metrics["ssim"], 4))
                self.history.setdefault("learning_rate", []).append(curr_lr)

                epoch_summary = (
                    f"\r{epoch:>5}/{self.total_epochs:<4} {gpu_mem:>10} {epoch_l1:>10.4f} "
                    f"{epoch_fft:>10.4f} {epoch_grad:>10.4f} {epoch_train_loss:>10.4f} {val_metrics['psnr']:>10.2f} "
                    f"{val_metrics['ssim']:>10.4f} {self.batch_size:>10} {hr.size(2)}x{hr.size(3)}"
                )
                print(epoch_summary)

                # Check for Best Model
                is_best = val_metrics["psnr"] > (self.best_psnr + self.min_delta)
                if is_best:
                    self.best_psnr = val_metrics["psnr"]
                    self.epochs_without_improvement = 0
                    best_ckpt_path = self._save_checkpoint("best_model.pth")
                    print(f"      |-- [*] Best model saved! (Val PSNR: {self.best_psnr:.2f} dB -> {best_ckpt_path})")
                else:
                    self.epochs_without_improvement += 1
                    if self.early_stopping_enabled:
                        print(f"      |-- [EarlyStopping] {self.epochs_without_improvement}/{self.patience} epochs without {self.min_delta} dB improvement.")

                self._save_checkpoint("last_checkpoint.pth")

                # Export multi-patch comparison test and training curves at EACH epoch
                eval_img, curves_img = self._export_best_model_artifacts(epoch, val_dataset)

                # Emit evaluation report event for GUI popup at EACH epoch
                report_event_type = "new_best_model" if is_best else "epoch_report"
                active_ckpt = str(self.export_dir / ("best_model.pth" if is_best else "last_checkpoint.pth"))
                self._emit({
                    "type": report_event_type,
                    "epoch": epoch,
                    "total_epochs": self.total_epochs,
                    "is_best": is_best,
                    "best_psnr": round(self.best_psnr, 2),
                    "psnr": round(val_metrics["psnr"], 2),
                    "psnr_gain": psnr_gain,
                    "ssim": round(val_metrics["ssim"], 4),
                    "psnr_rgb": round(val_metrics["psnr_rgb"], 2),
                    "psnr_nir": round(val_metrics["psnr_nir"], 2),
                    "train_loss": epoch_train_loss,
                    "val_loss": round(val_metrics["val_loss"], 5),
                    "l1_loss": epoch_l1,
                    "fft_loss": epoch_fft,
                    "grad_loss": epoch_grad,
                    "checkpoint_path": active_ckpt,
                    "eval_image_url": f"/api/artifacts/eval/{epoch}?t={int(time.time())}",
                    "curves_image_url": f"/api/artifacts/curves/{epoch}?t={int(time.time())}",
                    "timestamp": time.time()
                })

                # Emit epoch completion event
                self._emit({
                    "type": "epoch_summary",
                    "epoch": epoch,
                    "total_epochs": self.total_epochs,
                    "train_loss": epoch_train_loss,
                    "val_loss": val_metrics["val_loss"],
                    "l1_loss": epoch_l1,
                    "charbonnier_loss": epoch_l1,
                    "edge_loss": epoch_fft,
                    "fft_loss": epoch_fft,
                    "grad_loss": epoch_grad,
                    "sam_loss": 0.0,
                    "ssim_loss": round(1.0 - val_metrics["ssim"], 4),
                    "psnr": val_metrics["psnr"],
                    "psnr_gain": psnr_gain,
                    "ssim": val_metrics["ssim"],
                    "psnr_rgb": val_metrics["psnr_rgb"],
                    "psnr_nir": val_metrics["psnr_nir"],
                    "is_best": is_best,
                    "best_psnr": self.best_psnr,
                    "learning_rate": curr_lr,
                    "history": self.history,
                    "summary_line": epoch_summary
                })

                # Check Early Stopping
                if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                    self.state = TrainingState.EARLY_STOPPED
                    print(f"\n[Trainer] Early stopping triggered after {self.patience} epochs without improvement.")
                    print(f"[Trainer] Retained best model with PSNR: {self.best_psnr:.2f} dB\n")
                    self._emit({
                        "type": "state_change",
                        "state": self.state,
                        "message": f"Early stopping triggered. Best PSNR: {self.best_psnr:.2f} dB."
                    })
                    break

            if not self._stop_event.is_set() and self.state != TrainingState.EARLY_STOPPED:
                self.state = TrainingState.FINISHED
                print(f"\n[Trainer] Training successfully finished across {self.total_epochs} epochs!")
                self._emit({"type": "state_change", "state": self.state, "message": "Training finished."})

        except Exception as e:
            self.state = TrainingState.ERROR
            err_msg = f"Training error: {str(e)}"
            print(f"\n[Trainer ERROR] {err_msg}", file=sys.stderr)
            traceback.print_exc()
            self._emit({"type": "error", "message": err_msg})
        finally:
            self._cleanup_gpu_memory()

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_val_loss = 0.0
        metric_sums = {"psnr": 0.0, "ssim": 0.0, "psnr_rgb": 0.0, "ssim_rgb": 0.0, "psnr_nir": 0.0, "ssim_nir": 0.0, "bicubic_psnr": 0.0}
        count = 0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr, scale=self.scale)
                loss_dict = self.criterion(sr, hr)
                total_val_loss += loss_dict["total"].item()

                metrics = calculate_band_metrics(sr, hr)
                for k in ["psnr", "ssim", "psnr_rgb", "ssim_rgb", "psnr_nir", "ssim_nir"]:
                    metric_sums[k] += metrics[k]

                bic = torch.clamp(F.interpolate(lr, size=(hr.size(2), hr.size(3)), mode="bicubic", align_corners=False), 0.0, 2.0)
                metric_sums["bicubic_psnr"] += calculate_psnr(bic, hr)
                count += 1

        res = {k: v / max(1, count) for k, v in metric_sums.items()}
        res["val_loss"] = total_val_loss / max(1, count)
        return res

    def _export_best_model_artifacts(self, epoch: int, val_dataset: SatelliteSRDataset):
        """Generates best_model_eval_latest.png and training_curves_latest.png"""
        try:
            if not self.model:
                return

            print(f"      |-- [*] Exporting visual comparison test & training curves...")
            self.model.eval()
            num_samples = min(4, len(val_dataset))
            indices = np.linspace(0, len(val_dataset) - 1, num_samples, dtype=int)

            fig, axes = plt.subplots(num_samples, 4, figsize=(13, 3.2 * num_samples), facecolor="#08090a")
            if num_samples == 1:
                axes = np.array([axes])

            for row, idx in enumerate(indices):
                lr, hr = val_dataset[idx]
                lr_t = lr.unsqueeze(0).to(self.device)
                hr_t = hr.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    sr_t = self.model(lr_t, scale=self.scale)
                    bic_t = torch.clamp(F.interpolate(lr_t, size=(hr.size(1), hr.size(2)), mode="bicubic", align_corners=False), 0.0, 2.0)
                    psnr_val = calculate_psnr(sr_t, hr_t)
                    bic_psnr = calculate_psnr(bic_t, hr_t)

                lr_arr = denormalize_patch(bic_t[0])
                sr_arr = denormalize_patch(sr_t[0])
                hr_arr = denormalize_patch(hr_t[0])

                lr_rgb = make_rgb_composite(lr_arr)
                sr_rgb = make_rgb_composite(sr_arr)
                hr_rgb = make_rgb_composite(hr_arr)
                sr_cir = make_cir_composite(sr_arr)

                # Col 0: Input Bicubic
                axes[row, 0].imshow(lr_rgb)
                axes[row, 0].set_title(f"Input Bicubic\n{bic_psnr:.2f} dB", color="#f2994a", fontsize=9)
                axes[row, 0].axis("off")

                # Col 1: HAT-Light Model
                gain = psnr_val - bic_psnr
                axes[row, 1].imshow(sr_rgb)
                axes[row, 1].set_title(f"HAT-Light Super-Resolved (Epoch {epoch})\n{psnr_val:.2f} dB ({gain:+.2f} dB)", color="#4ebb78", fontsize=9, fontweight="bold")
                axes[row, 1].axis("off")

                # Col 2: Ground Truth HR
                axes[row, 2].imshow(hr_rgb)
                axes[row, 2].set_title("Ground Truth (10m HR)\nSentinel-2 RGB", color="#f7f8f8", fontsize=9)
                axes[row, 2].axis("off")

                # Col 3: Color Infrared (CIR)
                axes[row, 3].imshow(sr_cir)
                axes[row, 3].set_title("Color Infrared (CIR: B8, B4, B3)\nVegetation & Radiometry", color="#38bdf8", fontsize=9)
                axes[row, 3].axis("off")

            plt.suptitle(f"HAT-Light Multi-Patch Inference Comparison — Epoch {epoch} (Best PSNR: {self.best_psnr:.2f} dB)", color="#f7f8f8", fontsize=12, y=0.99)
            plt.tight_layout()

            eval_latest_path = self.export_dir / "best_model_eval_latest.png"
            eval_epoch_path = self.export_dir / f"best_model_eval_epoch_{epoch}.png"
            plt.savefig(eval_latest_path, dpi=140, facecolor="#08090a")
            plt.savefig(eval_epoch_path, dpi=140, facecolor="#08090a")
            plt.close()

            # --- 2. 6-Grid Telemetry Curves Plot ---
            fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), facecolor="#08090a")
            plt.subplots_adjust(wspace=0.3, hspace=0.35)

            ep = self.history["epochs"]

            # Subplot 1: Total Loss
            ax[0, 0].set_facecolor("#0f1013")
            ax[0, 0].plot(ep, self.history["train_loss"], color="#5e6ad2", label="Train Loss", lw=2)
            ax[0, 0].plot(ep, self.history["val_loss"], color="#f2994a", label="Val Loss", lw=1.5, ls="--")
            ax[0, 0].set_title("Total Loss (Exponential Decay)", color="#f7f8f8", fontsize=9)
            ax[0, 0].tick_params(colors="#8a8f98", labelsize=8)
            ax[0, 0].legend(fontsize=8, facecolor="#14151a", edgecolor="#262833", labelcolor="#f7f8f8")
            ax[0, 0].grid(True, color="#1c1e26", ls=":")

            # Subplot 2: L1 Reconstruction Loss
            ax[0, 1].set_facecolor("#0f1013")
            l1_vals = self.history.get("l1_loss", [])
            ax[0, 1].plot(ep, l1_vals, color="#4ebb78", lw=2)
            ax[0, 1].set_title("L1 Reconstruction Loss", color="#f7f8f8", fontsize=9)
            ax[0, 1].tick_params(colors="#8a8f98", labelsize=8)
            ax[0, 1].grid(True, color="#1c1e26", ls=":")

            # Subplot 3: Fourier FFT Loss
            ax[0, 2].set_facecolor("#0f1013")
            fft_vals = self.history.get("fft_loss", self.history.get("edge_loss", []))
            ax[0, 2].plot(ep, fft_vals, color="#ec4899", lw=2)
            ax[0, 2].set_title("Fourier FFT Frequency Loss (0.1x)", color="#f7f8f8", fontsize=9)
            ax[0, 2].tick_params(colors="#8a8f98", labelsize=8)
            ax[0, 2].grid(True, color="#1c1e26", ls=":")

            # Subplot 4: PSNR Overall & Gain
            ax[1, 0].set_facecolor("#0f1013")
            ax[1, 0].plot(ep, self.history["psnr_all"], color="#4ebb78", label="Overall PSNR (dB)", lw=2)
            ax[1, 0].plot(ep, self.history["psnr_gain"], color="#5e6ad2", label="Gain vs Bicubic (dB)", lw=1.5, ls=":")
            ax[1, 0].set_title("PSNR Trajectory & Gain", color="#f7f8f8", fontsize=9)
            ax[1, 0].tick_params(colors="#8a8f98", labelsize=8)
            ax[1, 0].legend(fontsize=8, facecolor="#14151a", edgecolor="#262833", labelcolor="#f7f8f8")
            ax[1, 0].grid(True, color="#1c1e26", ls=":")

            # Subplot 5: Per-Band PSNR
            ax[1, 1].set_facecolor("#0f1013")
            ax[1, 1].plot(ep, self.history["psnr_rgb"], color="#38bdf8", label="RGB Bands", lw=1.5)
            ax[1, 1].plot(ep, self.history["psnr_nir"], color="#ec4899", label="NIR Band (B08)", lw=1.5)
            ax[1, 1].set_title("Per-Band PSNR Fidelity (dB)", color="#f7f8f8", fontsize=9)
            ax[1, 1].tick_params(colors="#8a8f98", labelsize=8)
            ax[1, 1].legend(fontsize=8, facecolor="#14151a", edgecolor="#262833", labelcolor="#f7f8f8")
            ax[1, 1].grid(True, color="#1c1e26", ls=":")

            # Subplot 6: Learning Rate
            ax[1, 2].set_facecolor("#0f1013")
            ax[1, 2].plot(ep, self.history["learning_rate"], color="#f2994a", lw=1.5)
            ax[1, 2].set_title("Cosine Annealing LR", color="#f7f8f8", fontsize=9)
            ax[1, 2].tick_params(colors="#8a8f98", labelsize=8)
            ax[1, 2].grid(True, color="#1c1e26", ls=":")

            curves_latest_path = self.export_dir / "training_curves_latest.png"
            curves_epoch_path = self.export_dir / f"training_curves_epoch_{epoch}.png"
            plt.savefig(curves_latest_path, dpi=140, facecolor="#08090a")
            plt.savefig(curves_epoch_path, dpi=140, facecolor="#08090a")
            plt.close()
            return str(eval_latest_path), str(curves_latest_path)
        except Exception as e:
            print(f"[Export Error] {e}")
            return "", ""


# Backwards compatibility alias
Trainer = SRTrainer
