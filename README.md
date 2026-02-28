# 🤖 Reachy Copilot — Mistral AI Hackathon 2026

> **Embodied AI assistant** combining a Reachy Mini robot with fine-tuned Mistral models, OpenClaw-style personal AI tools, and edge deployment on NVIDIA Orin Nano.

**Mistral Worldwide Hackathon** — Feb 28 – Mar 1, 2026, NYC

## 🏗️ Architecture

```
Voice Input → Voxtral Mini 4B (ASR) → Fine-tuned Ministral 3 8B (Tool Calling)
                                           ↓
                                    18 OpenClaw Tools
                                    (email, calendar, web, smart home, ...)
                                           ↓
                                    Reachy Mini Robot
                                    (look, speak, express)
```

## 📦 Hardware

| Device | Role | Specs |
|--------|------|-------|
| RTX 5090 | Training + Inference | 32GB VRAM, QLoRA fine-tuning |
| NVIDIA Orin Nano Super | Edge deployment | 8GB, 67 TOPS |
| Reachy Mini | Embodied robot | Head tracking, speech, expressions |

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/tinytimor/mistral-ai-hackathon-stef.git
cd mistral-ai-hackathon-stef
cp .env.example .env   # Edit with your API keys
pip install -r requirements.txt

# 2. Test connection
python scripts/00_quickstart.py --provider mistral

# 3. Generate training data (teacher → student distillation)
python scripts/01_generate_training_data.py --provider mistral --model mistral-large-latest --num-samples 500

# 4. Fine-tune on RTX 5090
python scripts/02_sft_qlora.py --data data/training_data.jsonl --output models/ministral-3-8b-sft

# 5. GRPO reinforcement learning
python scripts/03_grpo_agent.py --model models/ministral-3-8b-sft --output models/ministral-3-8b-grpo

# 6. Quantize & deploy to Orin Nano
python scripts/04_quantize_deploy.py --model models/ministral-3-8b-grpo
```

## 📊 Experiment Tracking (Weights & Biases)

All training runs are tracked with [W&B](https://wandb.ai/):

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

Tracked metrics: loss, learning rate, gradient norms, reward scores, GPU utilization.

## 🧠 3-Tier Model Strategy

| Tier | Provider | Models | Use |
|------|----------|--------|-----|
| 1 | Microsoft Foundry | Mistral-Large-3 | Teacher (data gen) |
| 2 | Mistral API | All models (Ministral 3, Voxtral, Magistral) | Fallback teacher |
| 3 | Local RTX 5090 | Open-source (Apache 2.0) | Student training + inference |

## 🔧 18 OpenClaw Tools

`search_web` · `send_email` · `search_email` · `send_imessage` · `send_whatsapp` · `send_signal` · `send_telegram` · `calendar_list_events` · `calendar_create_event` · `browser_action` · `robot_look_at` · `robot_express` · `robot_speak` · `set_reminder` · `smart_home` · `spotify_control` · `post_tweet` · `memory_search`

## 📁 Project Structure

```
scripts/
  00_quickstart.py          # Test API connection
  01_generate_training_data.py  # Teacher generates tool-calling examples
  02_sft_qlora.py           # SFT with QLoRA + W&B tracking
  03_grpo_agent.py          # GRPO RL training + W&B tracking
  04_quantize_deploy.py     # Merge LoRA → GGUF → Ollama
docs/
  ORIN-REACHY-SETUP.md      # Hardware connection guide
```

## 📜 License

Training code: MIT. Models: Apache 2.0 (Ministral 3 family).
