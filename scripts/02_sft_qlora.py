#!/usr/bin/env python3
"""
02_sft_qlora.py - Supervised Fine-Tuning with QLoRA on Ministral 3B
for tool-calling + think-plan-act-reflect behavior.

Runs on RTX 5090 (32GB VRAM).

Prerequisites:
    pip install torch transformers trl peft bitsandbytes datasets accelerate

Usage:
    python scripts/02_sft_qlora.py --data data/training_data.jsonl --output models/ministral-3b-sft
    python scripts/02_sft_qlora.py --data data/training_data.jsonl --output models/ministral-3b-sft --epochs 3
"""

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
load_dotenv()

import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ─── Model Configuration ─────────────────────────────────────────────────────
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Ministral-3-8B-Instruct-2512")

# ─── QLoRA Configuration (4-bit quantization for memory efficiency) ──────────
QLORA_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - best for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 for RTX 5090
    bnb_4bit_use_double_quant=True,       # Double quantization saves ~0.4 bits/param
)

# ─── LoRA Configuration ──────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                                 # Rank - higher = more capacity, more VRAM
    lora_alpha=64,                        # Alpha - scaling factor (usually 2x r)
    lora_dropout=0.05,                    # Small dropout for regularization
    target_modules=[                      # Target all attention + MLP projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)


def format_messages_to_text(messages: list, tokenizer) -> str:
    """Convert a list of message dicts to the model's chat template format."""
    # Filter out invalid messages
    formatted = []
    for msg in messages:
        role = msg.get("role", msg.get("from", "user"))
        content = msg.get("content", msg.get("value", ""))

        # Map roles
        if role in ("system", "instruction"):
            formatted.append({"role": "system", "content": str(content)})
        elif role in ("user", "human"):
            formatted.append({"role": "user", "content": str(content)})
        elif role in ("assistant", "gpt", "bot"):
            # Include tool calls in the content if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_text = "\n".join(
                    f'<tool_call>\n{json.dumps({"name": tc.get("function", {}).get("name", ""), "arguments": json.loads(tc.get("function", {}).get("arguments", "{}"))})}\n</tool_call>'
                    for tc in tool_calls
                )
                content = (str(content) + "\n" + tc_text).strip()
            formatted.append({"role": "assistant", "content": str(content)})
        elif role == "tool":
            # Wrap tool responses
            tool_content = f'<tool_response>\n{content}\n</tool_response>'
            formatted.append({"role": "user", "content": tool_content})

    if not formatted:
        return ""

    try:
        text = tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=False)
        return text
    except Exception:
        # Fallback: simple concatenation
        parts = []
        for msg in formatted:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        return "\n".join(parts)


def load_training_data(data_path: str, tokenizer) -> Dataset:
    """Load and format training data from JSONL file."""
    texts = []
    skipped = 0

    with open(data_path, "r") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                messages = sample.get("messages", [])
                if not messages:
                    skipped += 1
                    continue

                text = format_messages_to_text(messages, tokenizer)
                if text and len(text) > 50:  # Skip very short samples
                    texts.append(text)
                else:
                    skipped += 1
            except (json.JSONDecodeError, Exception):
                skipped += 1
                continue

    print(f"  ✅ Loaded {len(texts)} samples, skipped {skipped}")
    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description="SFT with QLoRA on Ministral")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL file")
    parser.add_argument("--output", type=str, default="models/ministral-3b-sft", help="Output directory")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL, help="Base model name/path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "reachy-copilot"), help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔧 SFT with QLoRA - Ministral Tool-Calling Fine-Tune")
    print("=" * 60)

    # ─── Check GPU ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  No GPU detected! This will be very slow.")

    # ─── Initialize Weights & Biases ───────────────────────────────────────
    use_wandb = not args.no_wandb and os.getenv("WANDB_API_KEY")
    if use_wandb:
        run_name = args.wandb_run_name or f"sft-{Path(args.base_model).name}-r{args.lora_r}-ep{args.epochs}-lr{args.lr}"
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "task": "sft-qlora",
                    "base_model": args.base_model,
                    "lora_rank": args.lora_r,
                    "lora_alpha": args.lora_r * 2,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "learning_rate": args.lr,
                    "max_seq_len": args.max_seq_len,
                    "quantization": "4-bit NF4",
                    "optimizer": "adamw_torch_fused",
                    "scheduler": "cosine",
                    "gpu": gpu_name if torch.cuda.is_available() else "none",
                },
                tags=["sft", "qlora", "mistral", "reachy-copilot", "hackathon"],
                settings=wandb.Settings(init_timeout=300),
            )
            print(f"   📊 W&B run: {wandb.run.url}")
        except Exception as e:
            print(f"   ⚠️  W&B init failed ({e}) - continuing without logging")
            use_wandb = False
    else:
        print("   ⚠️  W&B disabled (set WANDB_API_KEY or remove --no-wandb)")

    # ─── Load tokenizer ──────────────────────────────────────────────────
    print(f"\n📦 Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ─── Load training data ──────────────────────────────────────────────
    print(f"\n📊 Loading training data from {args.data}...")
    full_dataset = load_training_data(args.data, tokenizer)
    print(f"   Total samples: {len(full_dataset)}")

    # Train/eval split for best-model checkpointing
    if len(full_dataset) >= 20:
        split = full_dataset.train_test_split(test_size=0.1, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]
        print(f"   Train: {len(dataset)}, Eval: {len(eval_dataset)} (10% holdout)")
    else:
        dataset = full_dataset
        eval_dataset = None
        print("   ⚠️  Dataset too small for eval split - skipping best-model selection")

    # Show a sample
    if len(dataset) > 0:
        sample = dataset[0]["text"][:200]
        print(f"   Sample preview: {sample}...")

    # ─── Load model with QLoRA ───────────────────────────────────────────
    print(f"\n🧠 Loading {args.base_model} with 4-bit QLoRA...")

    # Ministral 3 (Dec 2025) uses Mistral3ForConditionalGeneration (multimodal wrapper)
    # which AutoModelForCausalLM doesn't support - load explicitly
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration
        print(f"   📎 Detected Mistral3 multimodal architecture - loading with Mistral3ForConditionalGeneration")

        # Check if model is already quantized (e.g., FP8) - can't stack BnB on top
        existing_quant = getattr(config, "quantization_config", None)
        if existing_quant and isinstance(existing_quant, dict) and existing_quant.get("quant_method"):
            quant_method = existing_quant["quant_method"]
            print(f"   📎 Model is {quant_method}-quantized - dequantizing to BF16 for training")
            from transformers import FineGrainedFP8Config
            # Dequantize FP8 → BF16 (produces a clean, trainable model ~6.8GB)
            # With LoRA + gradient checkpointing + reduced batch, this fits on 32GB GPU
            model = Mistral3ForConditionalGeneration.from_pretrained(
                args.base_model,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=FineGrainedFP8Config(dequantize=True),
            )
            # Clear quantization state so trainer doesn't reject training
            if hasattr(model.config, 'quantization_config'):
                model.config.quantization_config = None
            if hasattr(model, 'is_quantized'):
                model.is_quantized = False
            if hasattr(model, 'hf_quantizer'):
                model.hf_quantizer = None
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                args.base_model,
                quantization_config=QLORA_CONFIG,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=QLORA_CONFIG,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Update LoRA config with user-specified rank
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─── Training configuration ──────────────────────────────────────────
    print("\n⚙️  Setting up training...")
    # Enable best-model checkpointing if we have an eval set
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,                 # Keep top 3 checkpoints
        eval_strategy="epoch" if has_eval else "no",
        load_best_model_at_end=has_eval,    # Auto-load best checkpoint at end
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,            # Lower eval_loss is better
        bf16=True,                          # Use bf16 on RTX 5090
        tf32=True,                          # TF32 for faster matmul
        gradient_checkpointing=True,        # Save VRAM
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_len,            # TRL 0.29+ renamed max_seq_length → max_length
        packing=True,                       # Pack short sequences for efficiency
        dataset_text_field="text",
        report_to="wandb" if use_wandb else "none",  # W&B experiment tracking
        optim="adamw_torch_fused",          # Fused optimizer for speed
        dataloader_num_workers=4,
        seed=42,
    )

    # ─── Create trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ─── Train! ──────────────────────────────────────────────────────────
    print("\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"   Learning rate: {args.lr}")
    print(f"   Max sequence length: {args.max_seq_len}")
    print(f"   LoRA rank: {args.lora_r}")
    if has_eval:
        print(f"   📌 Best-model checkpointing: ON (metric=eval_loss)")
    print()

    train_result = trainer.train()

    # ─── Extract best metrics ────────────────────────────────────────────
    best_eval_loss = None
    if has_eval:
        try:
            eval_result = trainer.evaluate()
            best_eval_loss = eval_result.get("eval_loss")
            print(f"\n📊 Best eval loss: {best_eval_loss:.4f}")
        except Exception as e:
            print(f"   ⚠️  Eval failed: {e}")

    train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None

    # ─── Save best model ─────────────────────────────────────────────────
    print(f"\n💾 Saving {'best ' if has_eval else ''}model to {args.output}...")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save training info with checkpoint metrics for cross-run comparison
    info = {
        "base_model": args.base_model,
        "lora_rank": args.lora_r,
        "epochs": args.epochs,
        "max_seq_len": args.max_seq_len,
        "data_path": args.data,
        "num_samples": len(dataset),
        "best_eval_loss": best_eval_loss,
        "final_train_loss": train_loss,
        "best_model_checkpoint": str(trainer.state.best_model_checkpoint) if has_eval and trainer.state.best_model_checkpoint else None,
        "load_best_model_at_end": has_eval,
    }
    with open(output_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Log final info to W&B
    if use_wandb:
        wandb.log({"num_samples": len(dataset), "final_epoch": args.epochs})
        # Save model artifact
        artifact = wandb.Artifact(
            name=f"sft-{Path(args.base_model).name}",
            type="model",
            metadata=info,
        )
        artifact.add_dir(str(output_path))
        wandb.log_artifact(artifact)
        wandb.finish()
        print("   📊 W&B run finished & model artifact saved")

    print("\n✅ SFT training complete!")
    print(f"   Model saved to: {args.output}")
    print(f"\n   Next step: python scripts/03_grpo_agent.py --model {args.output}")


if __name__ == "__main__":
    main()
