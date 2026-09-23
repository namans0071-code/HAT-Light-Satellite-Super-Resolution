"""
HAT-Light: Continuous Multi-Spectral Satellite Super-Resolution Training Entrypoint.
Usage:
    python train.py --data-dir path/to/dataset --epochs 100 --batch-size 16 --scale 4.0
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train HAT-Light Multi-Spectral Satellite Super-Resolution")
    parser.add_argument("--data-dir", type=str, default="data/Train", help="Directory containing HR .npy patches")
    parser.add_argument("--val-dir", type=str, default="data/Val", help="Directory containing Val .npy patches (optional)")
    parser.add_argument("--scale", type=float, default=4.0, help="Super-resolution scaling factor (default: 4.0)")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU step (default: 16)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate (default: 2e-4)")
    parser.add_argument("--embed-dim", type=int, default=96, help="Transformer embedding dimension (default: 96)")
    parser.add_argument("--num-rhag", type=int, default=6, help="Residual Hybrid Attention Groups (default: 6)")
    parser.add_argument("--num-hab", type=int, default=4, help="Hybrid Attention Blocks per RHAG (default: 4)")
    parser.add_argument("--output-dir", type=str, default="weights", help="Directory to save checkpoints (default: weights)")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from core.trainer import SRTrainer
    from core.hat_light import check_cuda_device

    device = check_cuda_device()
    print("=" * 70)
    print("  HAT-Light: Multi-Spectral Satellite Super-Resolution Training")
    print("=" * 70)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Scale: {args.scale}x | Epochs: {args.epochs} | Batch Size: {args.batch_size} (Accum: {args.grad_accum})")
    print(f"Embedding Dim: {args.embed_dim} | RHAGs: {args.num_rhag} | HABs/RHAG: {args.num_hab}")

    trainer = SRTrainer()
    trainer.start_training(
        train_dir=args.data_dir,
        val_dir=args.val_dir if Path(args.val_dir).exists() else None,
        scale=args.scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        checkpoint_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
