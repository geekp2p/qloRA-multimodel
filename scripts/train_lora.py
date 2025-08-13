import argparse, os, yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

def build_prompt_chatml(inst: str, inp: str, out: str) -> str:
    if inp:
        return f"""<|user|>
{inst}

{inp}
</|user|>
<|assistant|>
{out}
</|assistant|>
"""
    else:
        return f"""<|user|>
{inst}
</|user|>
<|assistant|>
{out}
</|assistant|>
"""

def build_prompt_llama3(inst: str, inp: str, out: str) -> str:
    sys = ""
    if inp:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>

{inst}

{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{out}<|eot_id|>"""
    else:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>

{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{out}<|eot_id|>"""

def make_formatter(template):
    if template == "chatml":
        return build_prompt_chatml
    elif template == "llama3":
        return build_prompt_llama3
    else:
        return build_prompt_chatml

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--alias", required=True, help="คีย์ของโมเดลใน YAML เช่น gptoss20b")
    p.add_argument("--train_file", default="data/train.jsonl")
    p.add_argument("--out_dir", default="artifacts")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.alias not in cfg["models"]:
        raise SystemExit(f"alias '{args.alias}' not found in {args.config}")
    mc = cfg["models"][args.alias]

    base_hf   = mc["base_hf"]
    template  = mc.get("template", "chatml")
    seq_len   = int(mc.get("seq_len", 512))
    r         = int(mc.get("lora_r", 8))
    lora_alpha= int(mc.get("lora_alpha", 16))
    lora_drop = float(mc.get("lora_dropout", 0.05))
    batch     = int(mc.get("batch_size", 1))
    grad_acc  = int(mc.get("grad_accum", 8))
    epochs    = int(mc.get("epochs", 1))
    lr        = float(mc.get("lr", 2e-4))

    ds = load_dataset("json", data_files={"train": args.train_file})

    fmt = make_formatter(template)
    def map_fn(ex):
        inst = (ex.get("instruction") or "").strip()
        inp  = (ex.get("input") or "").strip()
        out  = (ex.get("output") or "").strip()
        return {"text": fmt(inst, inp, out)}

    ds = ds.map(map_fn, remove_columns=ds["train"].column_names)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_hf,
        max_seq_length = seq_len,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_drop,
        target_modules = "all-linear",
    )

    out_dir = os.path.join(args.out_dir, args.alias, "lora")
    os.makedirs(out_dir, exist_ok=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds["train"],
        dataset_text_field = "text",
        max_seq_length = seq_len,
        args = TrainingArguments(
            output_dir = out_dir,
            per_device_train_batch_size = batch,
            gradient_accumulation_steps = grad_acc,
            num_train_epochs = epochs,
            learning_rate = lr,
            lr_scheduler_type = "cosine",
            warmup_ratio = 0.05,
            logging_steps = 10,
            save_steps = 200,
            fp16 = True,
            gradient_checkpointing = True,
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[OK] Saved LoRA to {out_dir}")

if __name__ == "__main__":
    main()
