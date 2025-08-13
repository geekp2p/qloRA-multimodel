# Multi-model QLoRA → GGUF → Ollama (Scaffold)

## โครงสร้าง
- configs/models.yaml — รายชื่อโมเดลและไฮเปอร์พารามิเตอร์
- data/train.jsonl — ตัวอย่างชุดข้อมูลแบบ instruction
- scripts/setup_env.sh — เตรียม Python virtualenv + dependencies (WSL/Ubuntu)
- scripts/train_lora.py — เทรน LoRA สำหรับ 1 alias
- scripts/train_one.sh — ช่วยเรียกเทรนทีละตัว
- scripts/train_all.sh — เทรนทั้งหมดตาม YAML
- scripts/convert_all_to_gguf.sh — แปลง LoRA ของทุก alias เป็น GGUF (ใช้ llama.cpp)
- windows/make_modelfiles.ps1 — สร้าง Modelfile / คัดลอก adapter.gguf ไปไว้ใต้ G:\ollama\models\finetunes\<alias> และเรียก `ollama create`

## ขั้นตอนใช้งาน (สรุป)
WSL/Ubuntu:
```
bash scripts/setup_env.sh
source .venv/bin/activate

# เทรน 1 โมเดล
bash scripts/train_one.sh gptoss20b
# หรือทั้งหมด
bash scripts/train_all.sh

# แปลง LoRA -> GGUF
bash scripts/convert_all_to_gguf.sh
```

Windows PowerShell (Run as Administrator ถ้าต้องเขียนลง G:\ollama\models):
```
powershell -ExecutionPolicy Bypass -File .\windows\make_modelfiles.ps1
# จากนั้นทดสอบ
ollama run gpt-oss:20b-ft "สรุปข้อความนี้ ..."
```

> ปรับ `base_hf` ให้ตรงกับชื่อ repo จริงบน HF ของโมเดลฐาน, และปรับไฮเปอร์พารามิเตอร์ใน `configs/models.yaml` ให้เหมาะกับ VRAM
