#!/usr/bin/env bash
set -euo pipefail
alias="${1:-gptoss20b}"
source .venv/bin/activate
python scripts/train_lora.py --alias "$alias" --config configs/models.yaml --train_file data/train.jsonl
