#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
ALIASES=$(python - <<'PY'
import yaml
cfg=yaml.safe_load(open("configs/models.yaml","r",encoding="utf-8"))
print(" ".join(cfg["models"].keys()))
PY
)
for a in $ALIASES; do
  echo "=== Training $a ==="
  python scripts/train_lora.py --alias "$a"
done
