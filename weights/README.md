# Model Weights

This directory contains pre-trained checkpoints for the HAT-Light super-resolution model.

---

## Available Checkpoints

### `best_model.pth`
- **Model:** HAT-Light
- **Size:** ~58.4 MB
- **Input Channels:** 4 (B04 Red, B03 Green, B02 Blue, B08 NIR)
- **Output Channels:** 4
- **Scale:** 4x (10m to 2.5m GSD)
- **Parameters:** ~4.2M
- **Hyperparameters:**
  - `embed_dim`: 96
  - `num_rhag`: 6 (Residual Hybrid Attention Groups)

---

## Loading the Checkpoint in Python

```python
import torch
from core.models.hat_light import HATLight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HATLight(in_channels=4, out_channels=4, embed_dim=96, num_rhag=6).to(device)

checkpoint = torch.load('weights/best_model.pth', map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print('HAT-Light loaded successfully.')
```
