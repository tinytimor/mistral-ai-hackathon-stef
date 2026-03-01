# 📊 Training Pipeline & Experiment Tracking

> SFT + GRPO training pipeline for distilling Mistral Large 3 → Ministral 3B,
> with full W&B experiment tracking across hyperparameter sweeps.

Back to [README](../README.md) · Agent architecture: [AGENTS.md](../AGENTS.md)

---

## W&B Dashboard

🔗 **Live dashboard:** [wandb.ai/thalamus_ai/reachy-copilot](https://wandb.ai/thalamus_ai/reachy-copilot)

📄 **Public training report:** [W&B Report - SFT + GRPO Training Results](https://wandb.ai/thalamus_ai/reachy-copilot/reports/Reachy-with-Nvidia-Orin-Nano-OpenClaw-SFT-GRPO-Training-Results-Mistral-Worldwide-Hackathon-2026---VmlldzoxNjA3MTY1Ng?accessToken=9xuegttcpd4wiqdujkwn8j8cnsspp1ulu1l3t84harjqnvles8eotub2ka766nwv)

---

## Running Training Experiments

```bash
# SFT with different hyperparameters
python scripts/02_sft_qlora.py --data data/training_data.jsonl --lora-r 16 --lr 1e-4 --wandb-run-name "sft-r16-lr1e4"
python scripts/02_sft_qlora.py --data data/training_data.jsonl --lora-r 64 --lr 2e-4 --wandb-run-name "sft-r64-lr2e4"

# GRPO with different generation counts
python scripts/03_grpo_agent.py --model models/sft-r32 --num-generations 4 --wandb-run-name "grpo-g4"
python scripts/03_grpo_agent.py --model models/sft-r32 --num-generations 8 --wandb-run-name "grpo-g8"

# Disable W&B
python scripts/02_sft_qlora.py --data data/training_data.jsonl --no-wandb
```

---

## Monitor from Your Laptop (or Phone)

```bash
# W&B dashboard - no VPN needed, works over cellular:
open https://wandb.ai/thalamus_ai/reachy-copilot

# Pipeline alerts - get push notifications when training finishes or crashes:
nohup ./run_experiments.sh > pipeline.log 2>&1 & disown
# Check W&B for alerts ✅ or ❌
```

**Tracked metrics:** loss, learning rate, gradient norms, reward scores, GPU utilization, hyperparameters.

---

## Best-Model Checkpointing

The training pipeline automatically selects the best model across all experiment runs:

- **SFT**: 10% eval holdout split → `load_best_model_at_end=True` → lowest `eval_loss` wins
- **GRPO**: Checkpoints every 50 steps → highest `best_reward` score wins
- **Cross-run comparison**: `run_experiments.sh` reads `training_info.json` from each run and auto-selects the winner

Metrics saved in each model's `training_info.json`:
```json
{"best_eval_loss": 0.42, "final_train_loss": 0.38, "best_model_checkpoint": "checkpoint-150"}
```

---

## Resilient Pipeline

The experiment runner (`run_experiments.sh`) continues through failures instead of crashing:
- Individual SFT/GRPO experiment failures are logged and skipped
- Missing training data → training phases skipped, deployment phases still run
- llama.cpp build failures → quantization skipped, model still available
- Final summary shows total failure count for review

---

## GRPO Reward Functions

| Reward | What It Measures | Weight |
|--------|-----------------|--------|
| `format_correctness` | Valid `<tool_call>` JSON output | 0.25 |
| `tool_relevance` | Correct tool for the situation | 0.30 |
| `response_quality` | Empathy, completeness, professionalism | 0.25 |
| `thinking_quality` | Think-plan-act-reflect reasoning | 0.20 |

---

## Training Data Pipeline

```
Mistral Large 3 (Teacher, 675B MoE)
    ↓ generates 500-4000 tool-calling conversations
    ↓ scripts/01_generate_training_data.py

Ministral 3B (Student)
    ↓ SFT with QLoRA (scripts/02_sft_qlora.py)
    ↓   └─ Best-model checkpointing: 10% eval split
    ↓ GRPO with reward functions (scripts/03_grpo_agent.py)
    ↓   └─ Checkpoint every 50 steps, save top 3
    ↓ Quantize to Q4_K_M GGUF (scripts/04_quantize_deploy.py)

Deploy to Orin Nano via Ollama
    ↓ ollama create reachy-copilot -f Modelfile
```
