# Ablation Studies

Scripts and evaluation data for analyzing individual components and loss terms of the HAT-Light model.

---

## Running Ablation Evaluations

To run the ablation evaluation suite across the test set:

```bash
python benchmarks/ablations/run_ablation.py
```

---

## Configurations Evaluated

1. **Full Proposed HAT-Light:** All 6 RHAG blocks with Window Attention, Shifted Window Attention, Depthwise FFN, and composite physical loss (Charbonnier + FFT + Gradient + SAM).
2. **W/o SAM Loss:** Replaces composite loss with pure L1 loss, testing the effect of spectral angle constraints.
3. **W/o FFT Spectral Loss:** Removes frequency-domain Fourier loss, testing the effect on high-frequency textural detail.
4. **W/o Depthwise FFN:** Replaces depthwise conv FFN with standard linear MLP layers to evaluate spatial inductive bias.
5. **Reduced Attention Groups (4 RHAG):** Evaluates throughput and latency trade-offs with 4 groups instead of 6.

Results are documented in `table_ablation.md`.
