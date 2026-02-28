# 🤖 Reachy Copilot — Mistral AI Hackathon 2026

> **Embodied AI assistant** combining a Reachy Mini robot with fine-tuned Mistral models,
> OpenClaw-style personal AI tools, and edge deployment on NVIDIA Orin Nano.
> Split-brain architecture: edge model for real-time reactions, cloud model for complex reasoning.

**Mistral Worldwide Hackathon** — Feb 28 – Mar 1, 2026, NYC

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT ORCHESTRATION                             │
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │
│  │ 🧠 Reasoning │    │ ⚡ Reactive  │    │ 🦞 OpenClaw Gateway     │ │
│  │   Agent      │    │   Agent      │    │   (Orchestrator)        │ │
│  │ Mistral Large│    │ Ministral 3B │    │   Session + Memory +    │ │
│  │ (Cloud/5090) │    │ (Orin Nano)  │    │   Multi-Channel         │ │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘ │
│         │                   │                        │                 │
│         └───────────────────┼────────────────────────┘                 │
│                             │                                          │
│                     ┌───────▼───────┐                                  │
│                     │ 🤖 Reachy     │                                  │
│                     │   Robot Agent │                                  │
│                     │ (Embodiment)  │                                  │
│                     └───────────────┘                                  │
└────────────────────────────────────────────────────────────────────────┘
```

### Voice Pipeline (Edge)
```
Microphone → Voxtral Mini 3B (ASR + understanding) → Ministral 3B (tool-calling)
                                                           ↓
                                                    18 OpenClaw Tools
                                                    (email, calendar, web, smart home, ...)
                                                           ↓
                                                    Reachy Mini Robot
                                                    (look, speak, express)
```

### Key Insight: 5090 Trains, Orin Runs
The RTX 5090 is used to **train** specialized models (SFT + GRPO distillation), but
at inference time everything runs on the **Orin Nano** (8GB) via Ollama — fully offline,
no 5090 needed. The cloud (Azure AI Foundry / Mistral Large) is an optional fallback
for complex reasoning only.

---

## 📦 Hardware

| Device | Role | Specs |
|--------|------|-------|
| RTX 5090 | Training + cloud inference | 32GB VRAM, Blackwell arch, QLoRA fine-tuning |
| NVIDIA Orin Nano Super | Edge deployment | 8GB, 67 TOPS, JetPack 6.2 |
| Reachy Mini | Embodied robot | 6-DOF head, antennas, camera, 4 mics, speaker |
| MacBook Pro | Remote monitoring | SSH, VNC, W&B dashboard from the train |

---

## 🚀 Quick Start

### Option A: Full Training Pipeline (RTX 5090)

```bash
# 1. Clone & setup
git clone https://github.com/tinytimor/mistral-ai-hackathon-stef.git
cd mistral-ai-hackathon-stef
cp .env.example .env   # Edit with your API keys
pip install -r requirements.txt

# 2. Download base models
python scripts/07_download_models.py

# 3. Test connection
python scripts/00_quickstart.py --provider mistral

# 4. Generate training data (teacher → student distillation)
python scripts/01_generate_training_data.py --provider mistral --model mistral-large-latest --num-samples 500

# 5. Fine-tune on RTX 5090
python scripts/02_sft_qlora.py --data data/training_data.jsonl --output models/ministral-3b-sft

# 6. GRPO reinforcement learning
python scripts/03_grpo_agent.py --model models/ministral-3b-sft --output models/ministral-3b-grpo

# 7. Quantize & deploy to Orin Nano
python scripts/04_quantize_deploy.py --model models/ministral-3b-grpo

# 8. Run the automated sweep (unattended):
nohup ./run_experiments.sh > pipeline.log 2>&1 & disown
```

### Option B: Quick Deploy (No Training, Pre-Quantized)

```bash
# Download pre-quantized GGUFs and deploy directly:
python scripts/07_download_models.py --no-train

# Or download the full edge stack (Ministral + Voxtral audio):
python scripts/07_download_models.py --edge-stack

# Copy to Orin Nano and create Ollama model:
scp -r models/ orin@<ORIN_IP>:~/reachy-model/
ssh orin 'cd ~/reachy-model && ollama create reachy-copilot -f Modelfile'

# Start the bridge server:
python scripts/06_openclaw_bridge.py --standalone --reachy-ip <REACHY_IP>
```

---

## 📊 Experiment Tracking (Weights & Biases)

All training runs are tracked with [W&B](https://wandb.ai/tinytimor/reachy-copilot)
and can be monitored in real-time from any device — your laptop on the train, your phone, etc.

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

### 📱 Monitor from Your Laptop (or Phone)
```bash
# W&B dashboard — no VPN needed, works over cellular:
open https://wandb.ai/tinytimor/reachy-copilot

# Pipeline alerts — get push notifications when training finishes or crashes:
nohup ./run_experiments.sh > pipeline.log 2>&1 & disown
# Check W&B for alerts ✅ or ❌
```

Tracked metrics: loss, learning rate, gradient norms, reward scores, GPU utilization, hyperparameters.

### 📌 Best-Model Checkpointing

The training pipeline automatically selects the best model across all experiment runs:

- **SFT**: 10% eval holdout split → `load_best_model_at_end=True` → lowest `eval_loss` wins
- **GRPO**: Checkpoints every 50 steps → highest `best_reward` score wins
- **Cross-run comparison**: `run_experiments.sh` reads `training_info.json` from each run and auto-selects the winner

Metrics saved in each model’s `training_info.json`:
```json
{"best_eval_loss": 0.42, "final_train_loss": 0.38, "best_model_checkpoint": "checkpoint-150"}
```

### 🛡️ Resilient Pipeline

The experiment runner continues through failures instead of crashing:
- Individual SFT/GRPO experiment failures are logged and skipped
- Missing training data → training phases skipped, deployment phases still run
- llama.cpp build failures → quantization skipped, model still available
- Final summary shows total failure count for review

---

## 🧠 Multi-Agent Model Strategy

| Agent | Model | Location | Latency | Role |
|-------|-------|----------|---------|------|
| Reactive | Ministral 3B Q4_K_M (fine-tuned) | Orin Nano | <1s | Real-time robot control, simple queries |
| Reactive (Audio) | Voxtral Mini 3B Q4_K_M | Orin Nano | <2s | Voice STT + audio understanding + tool-calling |
| Reasoning | Mistral Large 3 (675B MoE) | Azure Foundry / 5090 | 2-5s | Complex reasoning, multi-step planning |
| Gateway | Model-agnostic | 5090 Desktop | — | Session management, memory, routing |

### Edge Memory Budget (Orin Nano 8GB)
```
Text mode (default):   ~5.3 GB total  ✅  (2.7 GB headroom)
Audio mode (Voxtral):  ~5.6 GB total  ✅  (2.4 GB headroom)
Dual Q3 mode (both):   ~6.9 GB total  ✅  (1.1 GB headroom)
```

---

## 🔧 18 OpenClaw Tools

`search_web` · `send_email` · `search_email` · `send_imessage` · `send_whatsapp` · `send_signal` · `send_telegram` · `calendar_list_events` · `calendar_create_event` · `browser_action` · `robot_look_at` · `robot_express` · `robot_speak` · `set_reminder` · `smart_home` · `spotify_control` · `post_tweet` · `memory_search`

---

## 📁 Project Structure

```
scripts/
  00_quickstart.py              # Test API connection to Mistral/Foundry
  01_generate_training_data.py  # Teacher generates tool-calling examples
  02_sft_qlora.py               # SFT with QLoRA + W&B tracking
  03_grpo_agent.py              # GRPO reinforcement learning + W&B tracking
  04_quantize_deploy.py         # Merge LoRA → GGUF → Ollama Modelfile
  05_memory_manager.py          # 3-tier memory system (L1/L2/L3)
  06_openclaw_bridge.py         # Edge bridge: Ollama + Reachy + memory + smart routing
  07_download_models.py         # Download & cache all models (training or pre-quantized)

docs/
  ORIN-REACHY-SETUP.md          # Hardware connection guide (Orin ↔ Reachy)
  REMOTE-ACCESS-MAC.md          # VNC + SSH + Tailscale setup (Mac → Orin)

run_experiments.sh              # Automated training sweep (leave running on 5090)
AGENTS.md                       # Full multi-agent architecture spec
BATTLE-PLAN.md                  # Hackathon strategy & prize targeting
```

---

## 🏆 Prize Strategy

| Prize | How We Hit It |
|-------|--------------|
| **Local 1st-3rd** | Full multi-agent architecture — edge + cloud, robot on the table |
| **Best Voice (ElevenLabs)** | Voice loop: wake → Voxtral STT → LLM → ElevenLabs TTS → robot speaker |
| **Best Use of Mistral Vibe** | Built the project with Vibe + created a Vibe skill for Reachy |
| **Best Architectural Modification** | Split-brain: edge model for reactions, cloud for reasoning |

---

## 🙏 References & Credits

This project builds on the work of many open-source contributors. We gratefully acknowledge:

### Core Projects
| Project | Author(s) | License | How We Use It |
|---------|-----------|---------|---------------|
| [OpenClaw](https://github.com/openclaw/openclaw) | OpenClaw Team | MIT | Gateway orchestration, skill system, multi-channel messaging |
| [clawd-reachy-mini](https://github.com/ArturSkowronski/clawd-reachy-mini) | Artur Skowronski | MIT | Reachy Mini SDK patterns, ElevenLabs TTS, wake word detection |
| [VisionClaw](https://github.com/1rgs/VisionClaw) | 1rgs | MIT | OpenAI-compatible endpoint pattern, single `execute` tool design |
| [Reachy Mini SDK](https://github.com/pollen-robotics/reachy2-sdk) | Pollen Robotics | Apache 2.0 | Robot control (head, antennas, camera, microphone) |

### Models
| Model | Provider | License | Role |
|-------|----------|---------|------|
| [Mistral Large 3](https://mistral.ai/) | Mistral AI | Mistral Research License | Teacher model (data generation) |
| [Ministral 3 3B Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Mistral AI | Apache 2.0 | Student model (fine-tuned for edge) |
| [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Mistral AI | Apache 2.0 | Edge audio (ASR + understanding) |
| [Ministral 3 3B GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF) | Mistral AI | Apache 2.0 | Pre-quantized for Ollama |
| [Voxtral Mini 3B GGUF](https://huggingface.co/mradermacher/Voxtral-Mini-3B-2507-GGUF) | mradermacher | Apache 2.0 | Pre-quantized for edge audio |

### Libraries
[Hugging Face Transformers](https://github.com/huggingface/transformers) ·
[TRL](https://github.com/huggingface/trl) ·
[PEFT](https://github.com/huggingface/peft) ·
[Weights & Biases](https://wandb.ai/) ·
[Ollama](https://ollama.ai/) ·
[FastAPI](https://fastapi.tiangolo.com/) ·
[ElevenLabs](https://elevenlabs.io/) ·
[Piper TTS](https://github.com/rhasspy/piper) ·
[llama.cpp](https://github.com/ggerganov/llama.cpp)

### Research Inspiration
- **DeepSeek-R1** — GRPO reinforcement learning approach for reasoning
- **Mistral Agent Skills** standard — tool-calling format
- **Split-brain architecture** pattern — edge reactions + cloud reasoning

---

## 📜 License

Training code: MIT. Models: Apache 2.0 (Ministral 3 family).
