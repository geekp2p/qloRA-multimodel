#!/usr/bin/env bash
set -euo pipefail

# แปลง LoRA adapters (.safetensors) เป็น GGUF เพื่อนำไปใช้กับ Ollama
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi

ALIASES=$(python - <<'PY'
import yaml
cfg=yaml.safe_load(open("configs/models.yaml","r",encoding="utf-8"))
print(" ".join(cfg["models"].keys()))
PY
)

for a in $ALIASES; do
  in_dir="artifacts/${a}/lora"
  out_dir="artifacts/${a}"
  out_file="${out_dir}/adapter.gguf"
  if [ ! -d "$in_dir" ]; then
    echo "Skip $a (no LoRA at $in_dir)"
    continue
  fi
  mkdir -p "$out_dir"
  echo "=== Converting $a ==="
  python llama.cpp/convert-lora-to-gguf.py --lora-in "$in_dir" --gguf-out "$out_file"
  echo "[OK] Wrote $out_file"
done
