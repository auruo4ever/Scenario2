#!/bin/bash
set -e
cd "$(dirname "$0")"
for part in 48k_96k 40k_80k 24k_56k_b3; do
    python scripts/predict.py check data/predictions/qwen_${part}_frozen.csv data/results/qwen_$part.jsonl | tail -6
done
python scripts/predict.py summary data/results/*.jsonl
python -m pytest -q tests
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python -m kvcost.bench configs/tiny.yaml
fi
