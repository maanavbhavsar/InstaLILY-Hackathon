#!/bin/bash
# Setup training environment on RTX 6000 (96GB VRAM)
set -e

echo "=== Setting up training environment ==="

# Install dependencies
pip install -r training/requirements.txt

# Verify CUDA
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Verify transformers can load model config
python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('google/gemma-3n-E4B-it')
print(f'Model: {c.model_type}')
print('Model config loaded successfully')
"

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run smoke test:  python3 training/train.py --max-steps 2 --batch-size 1"
echo "  2. Full training:   python3 training/train.py"
