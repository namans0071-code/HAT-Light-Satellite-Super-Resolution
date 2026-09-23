import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from baselines.edsr import EDSR
from baselines.rcan import RCAN
from baselines.swinir_light import SwinIRLight
from core.dataset import SatelliteSRDataset
from core.losses import calculate_band_metrics

def train_single_baseline(model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 6):
    print(f"\n=======================================================")
    print(f" TRAINING BASELINE: {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"=======================================================")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda")
    
    best_psnr = -1.0
    save_path = Path(f"weights/{model_name.lower()}_best.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()
        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sr = model(lr)
                loss = criterion(sr, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        scheduler.step()
        epoch_time = time.time() - start
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_psnr = []
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                lr_v, hr_v = lr_v.to(device), hr_v.to(device)
                sr_v = model(lr_v)
                m = calculate_band_metrics(sr_v, hr_v)
                val_psnr.append(m["psnr"])
        mean_psnr = float(sum(val_psnr) / len(val_psnr))
        print(f"Epoch {epoch:2d}/{epochs} [{epoch_time:.1f}s] - Train Loss: {avg_loss:.4f} | Val PSNR: {mean_psnr:.2f} dB")
        
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save({
                "model_name": model_name,
                "model_state": model.state_dict(),
                "best_psnr": best_psnr,
                "epoch": epoch
            }, save_path)
            
    print(f"--> Saved best {model_name} checkpoint (PSNR: {best_psnr:.2f} dB) to {save_path}")
    return save_path

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train dataset (using 2,500 diverse samples for swift baseline training)
    train_dir = "D:/Projects/SIH_SatSuperResoulution/First_Personal_Trainer_and_Tester/Dataset/data/Train"
    val_dir = "D:/Projects/SIH_SatSuperResoulution/First_Personal_Trainer_and_Tester/Dataset/data/Val"
    
    train_ds = SatelliteSRDataset(train_dir, scale=4, is_train=True, augment=True, max_samples=2560)
    val_ds = SatelliteSRDataset(val_dir, scale=4, is_train=False, augment=False, max_samples=200)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Baseline Training set: {len(train_ds)} patches | Validation: {len(val_ds)} patches")
    
    # Train EDSR
    edsr = EDSR(n_feats=64, n_resblocks=16)
    train_single_baseline("EDSR", edsr, train_loader, val_loader, device, epochs=6)
    
    # Train RCAN
    rcan = RCAN(n_feats=64, n_resgroups=4, n_rcab=4)
    train_single_baseline("RCAN", rcan, train_loader, val_loader, device, epochs=6)
    
    # Train SwinIR-Light
    swin = SwinIRLight(embed_dim=60, num_rstb=4, depth_per_rstb=4)
    train_single_baseline("SwinIR_Light", swin, train_loader, val_loader, device, epochs=6)
    
    print("\nAll baseline models trained successfully!")

if __name__ == "__main__":
    main()
