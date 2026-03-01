# 🤖 Reachy Copilot — Mistral AI Hackathon 2026

> **Embodied AI assistant** powered by the full Mistral model family on NVIDIA edge hardware —
> a fine-tuned Ministral 3B runs locally on a Jetson Orin Nano Super (8GB) via Ollama,
> with Mistral API for vision (Pixtral), voice (Voxtral ASR), and complex reasoning (Mistral Large),
> all embodied in a Reachy Mini robot that **sees, hears, speaks, thinks, and moves**.

**[Mistral Worldwide Hackathon](https://mistral.ai/hackathon)** — Feb 28 – Mar 1, 2026, NYC\
Organized in partnership with **Weights & Biases**, **NVIDIA**, **AWS** · Awards by **ElevenLabs**, **Hugging Face**, **Tilde Research**

> ⚠️ **Work in Progress** — Built in just 2 days (Feb 28–Mar 1). The core pipeline is working:
> local Ministral 3B on Orin Nano + Reachy Mini robot control + Mistral API for ASR/vision/fallback.
> With more time, the goal is to run **all** multimodal capabilities on-device — Mistral's new
> `ministral-3:3b` (3.0 GB, vision + text + tools) is already available on Ollama, making
> fully offline embodied AI on a $249 board a near-term reality. See [Future Work](#-future-work).

---

## 🎬 What It Does

Reachy Copilot is a **physical robot assistant** you can talk to. It:

- 👀 **Sees** — Camera via Reachy's IMX708 → Pixtral vision (Mistral API)
- 👂 **Hears** — 4 onboard mics → Voxtral ASR (Mistral API) for speech-to-text
- 🧠 **Thinks** — Fine-tuned Ministral 3B running **locally on Ollama** with tool-calling
- 🔊 **Speaks** — edge-tts → Reachy's built-in speaker via dmix audio
- 🤖 **Moves** — Head tracking, nodding, antenna emotions via reachy-mini SDK
- 🔍 **Searches** — Brave Search API for real-time web queries
- 🦞 **Orchestrates** — OpenClaw Gateway for session management and tool routing

### Demo
```bash
python demo.py              # Text mode — type to chat with Reachy
python demo.py --voice      # Voice mode — speak to Reachy via mic + Voxtral ASR
```

---

## 🏗️ Architecture — What Runs Where

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHAT RUNS WHERE                               │
│                                                                  │
│  🟢 LOCAL (Orin Nano 8GB)          🔵 MISTRAL API (Cloud)       │
│  ├─ Ministral 3B Q4_K_M (Ollama)  ├─ Vision: Pixtral           │
│  │  Chat + tool-calling (<1s)     │  (via mistral-small)        │
│  ├─ Robot control (reachy SDK)     ├─ ASR: Voxtral Mini         │
│  │  Head, antennas, emotions       │  (speech-to-text)          │
│  ├─ OpenClaw Gateway (:18789)      ├─ Fallback: Mistral Large   │
│  │  Memory, sessions, routing      │  (complex reasoning)       │
│  └─ Camera capture (SSH→Reachy)    └─ Web: Brave Search API     │
│                                                                  │
│  🟡 EDGE-TTS (free)                                             │
│  └─ Text-to-speech → Reachy speaker                             │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Mistral AI provides the entire model stack** — from 3B edge models to 675B MoE reasoning:

| Model | Role | Where | Why Mistral |
|-------|------|-------|-------------|
| **Ministral 3B** | Chat + tool-calling | Orin Nano (local) | Only 2 GB Q4, <1s latency, Apache 2.0 |
| **Voxtral Mini** | Speech-to-text (ASR) | Mistral API* | Native audio understanding, no Whisper needed |
| **Pixtral** (via mistral-small) | Camera → description | Mistral API* | Built-in vision, no separate CLIP model |
| **Mistral Large 3** | Complex reasoning fallback | Mistral API | 675B MoE for multi-step planning |

*\*Currently API calls — see [Future Work](#-future-work) for on-device roadmap.*

**NVIDIA provides the edge compute** — the Orin Nano Super (8GB, 67 TOPS) runs the fine-tuned
model at <1s latency with 2.8 GB of headroom. The RTX 5090 (32GB) was used for SFT + GRPO
training, but is **not needed at inference time**.

### Conversation Flow
```
User speaks → Reachy Mic (4-ch array) → SSH → Orin Nano
                                                ↓
                              Voxtral ASR (Mistral API) → text
                                                ↓
                              Ministral 3B (Ollama, LOCAL, <1s) → response + tool calls
                                                ↓
                              ┌─────────────────────────────────┐
                              │ Tool Calls (executed locally):   │
                              │  search_web → Brave Search API   │
                              │  look_at → reachy SDK            │
                              │  speak → edge-tts → speaker      │
                              │  see → SSH camera → Pixtral API  │
                              │  express → reachy SDK            │
                              └─────────────────────────────────┘
                                                ↓
                              edge-tts → ffmpeg → SCP → aplay on Reachy speaker
```

---

## 📦 Hardware

| Device | Role | Specs |
|--------|------|-------|
| NVIDIA Orin Nano Super | Edge inference + orchestration | 8GB unified, 67 TOPS, JetPack 6.2 |
| Reachy Mini | Embodied robot | 6-DOF head, antennas, IMX708 cam, 4 mics, speaker |
| RTX 5090 | Training only (not runtime) | 32GB VRAM, Blackwell, QLoRA SFT + GRPO |

---

## 🚀 Quick Start — Run the Demo

```bash
# On the Orin Nano (or any machine with Ollama + SSH access to Reachy):

# 1. Ensure Ollama is running with our model
ollama list   # should show reachy-copilot:latest (~2.0 GB)

# 2. Clone this repo and install deps
git clone https://github.com/tinytimor/mistral-ai-hackathon-stef.git
cd mistral-ai-hackathon-stef

# 3. Set up Python env (we use clawd-reachy-mini's venv)
cd ~/clawd-reachy-mini
uv sync --extra dev --extra audio

# 4. Run the demo
.venv/bin/python3 demo.py              # text mode
.venv/bin/python3 demo.py --voice      # voice mode (mic + Voxtral ASR)

# Environment variables (optional — defaults are set in demo.py):
export REACHY_IP=10.0.0.129
export MISTRAL_API_KEY=your-key
export BRAVE_API_KEY=your-key
export MIC_SECONDS=5            # recording duration per voice turn
```

---

## 📊 Experiment Tracking — [Weights & Biases](https://wandb.ai/thalamus_ai/reachy-copilot) *(Hackathon Partner)*

[Weights & Biases](https://wandb.ai/) is an **organizing partner** of the Mistral Worldwide Hackathon.
We use W&B to track all SFT and GRPO training experiments — loss curves, reward scores,
hyperparameters, GPU utilization, and model checkpoints — across our RTX 5090 training runs.

🔗 **Live dashboard:** [wandb.ai/thalamus_ai/reachy-copilot](https://wandb.ai/thalamus_ai/reachy-copilot)
— 6 completed runs (5 SFT sweeps + 1 GRPO) with full metrics.

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
open https://wandb.ai/thalamus_ai/reachy-copilot

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

## 🧠 Multi-Agent Architecture

> **All of Reachy's intelligence comes from the Mistral model family** — from a 3B edge model
> to a 675B cloud reasoner — orchestrated through OpenClaw Gateway on the Orin Nano.

| Agent | Model | Location | Latency | Role |
|-------|-------|----------|---------|------|
| Reactive | Ministral 3B Q4_K_M (fine-tuned) | **Orin Nano (local)** | <1s | Chat, tool-calling, robot control |
| Vision | Pixtral (via mistral-small) | Mistral API | ~2s | Camera → scene description |
| Voice | Voxtral Mini | Mistral API | ~2s | Speech-to-text (ASR) |
| Reasoning | Mistral Large 3 (675B MoE) | Mistral API | 2-5s | Complex reasoning fallback |
| Gateway | OpenClaw | **Orin Nano (local)** | — | Session management, memory, skill routing |

### Edge Memory Budget (Orin Nano 8GB)
```
Production mode (all-on-Orin):  ~5.2 GB total  ✅  (2.8 GB headroom)
With Voxtral swap (voice):     ~5.7 GB total  ✅  (2.3 GB headroom)
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
  06_openclaw_bridge.py         # Legacy bridge (replaced by OpenClaw Gateway)
  07_download_models.py         # Download & cache all models (training or pre-quantized)

docs/
  ORIN-REACHY-SETUP.md          # Hardware connection guide (Orin ↔ Reachy)
  REMOTE-ACCESS-MAC.md          # VNC + SSH + Tailscale setup (Mac → Orin)

run_experiments.sh              # Automated training sweep (leave running on 5090)
AGENTS.md                       # Full agent architecture + deployment guide
BATTLE-PLAN.md                  # Hackathon strategy & prize targeting
```

---

## 🤖 Deploying to Orin Nano

The fine-tuned model runs entirely on the Orin Nano (8GB) via Ollama + OpenClaw Gateway
+ clawd-reachy-mini — no cloud needed.

### What Gets Deployed

| Component | Description |
|-----------|-------------|
| `model-q4_k_m.gguf` (~2.0 GB) | Quantized fine-tuned Ministral 3B (via Ollama) |
| `Modelfile` (~1 KB) | Ollama config: system prompt + Mistral v7 chat template |
| OpenClaw Gateway | Node.js daemon on port 18789, skill orchestration |
| clawd-reachy-mini | Python voice/robot interface, Whisper + ElevenLabs |

### Memory Budget on Orin Nano (8GB)

```
Ollama (reachy-copilot)  : ~2.0 GB
KV Cache (2048 ctx)      : ~0.5 GB
CUDA runtime             : ~0.8 GB
OS + JetPack             : ~1.0 GB
OpenClaw Gateway (Node)  : ~0.3 GB
clawd-reachy-mini (Py)   : ~0.4 GB
reachy-mini SDK          : ~0.2 GB
─────────────────────────────────
Total                    : ~5.2 GB  ✅  (2.8 GB headroom)
```

### Orin Nano Quick Setup

> **Before you start:** Close Chrome and VS Code on the Orin. The Orin uses unified memory
> (CPU + GPU share the same pool). Chrome alone consumes ~1.5 GB, leaving no room for the
> 2 GB model + CUDA runtime. Kill them first: `pkill -f chromium; pkill -f code`

```bash
# 1. Install Ollama on the Orin
curl -fsSL https://ollama.com/install.sh | sh

# 2. Copy model files from the 5090 (or they may already be on the Orin)
mkdir -p ~/reachy-model
# (scp from 5090 — see Option C above)

# 3. Create and test the model
# IMPORTANT: must cd into the model directory — Modelfile uses a relative path
cd ~/reachy-model   # (or wherever model-q4_k_m.gguf lives)
ollama create reachy-copilot -f Modelfile
ollama run reachy-copilot "Hello!"

# 4. Install Node.js >= 22 + OpenClaw Gateway
# Use npm directly — the install.sh script can hang/crash on the Orin
node --version  # must be >= 22 (already installed via nvm on this machine)
npm i -g openclaw

# 5. Run onboarding wizard (answer: Mistral / your API key / mistral-large-latest /
#    No to skills / No to channels / Hatch in TUI)
openclaw onboard --install-daemon
# When TUI opens, press q to exit — daemon keeps running

# 6. Write the openclaw.json config
# NOTE: "memory" and "bind" are NOT valid keys — omit them
# NOTE: "mode": "local" is REQUIRED or the gateway refuses to start
cat > ~/.openclaw/openclaw.json << 'EOF'
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "ollama/reachy-copilot",
        "fallbacks": ["mistral/mistral-large-latest"]
      }
    }
  },
  "gateway": {
    "mode": "local",
    "port": 18789,
    "auth": { "mode": "token", "token": "reachy-hackathon-2026" },
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  },
  "env": {
    "MISTRAL_API_KEY": "<your-mistral-api-key>",
    "BRAVE_API_KEY": "<your-brave-api-key>"
  }
}
EOF

# 7. Validate config and start the gateway service
openclaw doctor --fix
systemctl --user start openclaw-gateway.service
sleep 5 && ss -tlnp | grep 18789   # should show port bound

# Verify gateway started with the right model:
# Look for: [gateway] agent model: ollama/reachy-copilot

# 8. Install uv + clone clawd-reachy-mini
# uv is NOT installed by default — install it first
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # reload PATH

cd ~
git clone https://github.com/ArturSkowronski/clawd-reachy-mini.git
cd clawd-reachy-mini
# uv sync creates .venv automatically — do NOT use conda or pip
# Warning "'reachy-mini' does not have extra 'vision'" is harmless
# Downloads ~300 MB (torch, scipy, opencv, etc.) — allow 5-10 min
uv sync --extra dev --extra audio
uv run clawd-reachy --gateway-host localhost --gateway-port 18789

# 9. Test the full pipeline (OpenClaw HTTP API)
curl -X POST http://127.0.0.1:18789/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer reachy-hackathon-2026" \
  -d '{"model": "ollama/reachy-copilot", "messages": [{"role": "user", "content": "Look at me and say hello!"}]}'
```

**Troubleshooting OpenClaw on Orin:**

| Error | Fix |
|-------|-----|
| `cudaMalloc failed: out of memory` | Close Chrome (`pkill -f chromium`) and VS Code, retry |
| `Gateway start blocked: set gateway.mode=local` | Add `"mode": "local"` to the `gateway` block in `openclaw.json` |
| `Unrecognized key: "memory"` | Remove the `memory` block — it's built-in, not configurable |
| `Invalid input` for `bind` | Remove the `bind` key entirely — let OpenClaw use its default |
| Gateway timed out / port not bound | Run `openclaw doctor --fix` then `systemctl --user restart openclaw-gateway.service` |
| `uv: command not found` | Run `curl -LsSf https://astral.sh/uv/install.sh \| sh && source ~/.bashrc` |

### Verified Tool Calling

The fine-tuned model correctly calls tools in Mistral v7 format:

```
User: "Search the web for weather in DC"
Model: [TOOL_CALLS]search_web[ARGS]{"query": "weather in DC right now", "max_results": 1}

User: "What's on my calendar today?"
Model: [TOOL_CALLS]calendar_list_events[ARGS]{"calendar_id": "primary", "from_date": "2026-02-28T00:00:00", ...}

User: "Look at me and say hello!"
Model: [TOOL_CALLS]look_at[ARGS]{"x": 1.0, "y": 0.0, "z": 0.0, "duration": 1.0}
       [TOOL_CALLS]speak[ARGS]{"text": "Hello! It's great to see you today."}
```

For complete hardware setup instructions, see [docs/ORIN-REACHY-SETUP.md](docs/ORIN-REACHY-SETUP.md).

---

## 📈 Training Results (2-Day Sprint)

> Built in just 2 days (Feb 28 – Mar 1, 2026). Training ran on the RTX 5090 while
> we built the demo on the Orin Nano in parallel. The SFT + GRPO pipeline works
> end-to-end, but with more time we'd scale up training data and GRPO iterations.

### W&B Dashboard — [wandb.ai/thalamus_ai/reachy-copilot](https://wandb.ai/thalamus_ai/reachy-copilot)

All 6 training runs completed and tracked (RTX 5090, ~12h ago):

| Run Name | Phase | Tags | Runtime | Status |
|----------|-------|------|---------|--------|
| `sft-r16-lr2e4` | SFT | qlora, mistral, sft | 3m 14s | ✅ Finished |
| `sft-r32-lr2e4` | SFT | qlora, mistral, sft | 4m 25s | ✅ Finished |
| `sft-r32-lr1e4` | SFT | qlora, mistral, sft | 4m 42s | ✅ Finished |
| `sft-r32-ep5-lr2e4` | SFT | qlora, mistral, sft | 5m 14s | ✅ Finished |
| `sft-r64-lr2e4` | SFT | qlora, mistral, sft | 7m 37s | ✅ Finished |
| `grpo-g4-test-fix2` | GRPO | grpo, rl, tool-calling | 16m 41s | ✅ Finished |

### Best Results

| Phase | Best Model | Metric | Value |
|-------|-----------|--------|-------|
| SFT | `sft-r64-lr2e4` | eval_loss | **0.266** |
| GRPO | `grpo-g4-test-fix2` | best_reward | **-0.5** |

### Training Details
- **Base model:** `mistralai/Ministral-3-3B-Instruct-2512` (FP8 → dequantized to BF16)
- **SFT hyperparameter sweep:** LoRA r ∈ {16, 32, 64}, lr ∈ {1e-4, 2e-4}, epochs ∈ {3, 5}
- **GRPO:** 4 generations, 1 epoch, 4 reward functions (format, tool relevance, response quality, thinking)
- **Quantization:** F16 → Q4_K_M via llama.cpp (6.4 GB → 2.0 GB)
- **GPU:** NVIDIA RTX 5090 (32GB VRAM, Blackwell sm_120)
- **Experiment tracking:** Weights & Biases — all runs, metrics, and hyperparameters logged

**What we'd do with more time:**
- Scale training data from 100 → 1000+ examples via Mistral Large teacher
- More GRPO iterations to improve reward scores
- Fine-tune `ministral-3:3b` with vision (on-device multimodal)
- Swap in Voxtral Mini for on-device ASR (no API needed)

---

## 🔮 Future Work

> **The path to fully offline embodied AI is shorter than you think.**

Right now, we use Mistral API for vision (Pixtral) and ASR (Voxtral). But Mistral's model
family is rapidly converging toward **all-in-one edge models** that can do text + vision +
audio + tool-calling in a single model that fits on a $249 board.

### What's Already Possible (Today)

| Model | Size on Ollama | Capabilities | Fits Orin 8GB? |
|-------|---------------|--------------|----------------|
| `ministral-3:3b` | **3.0 GB** | Text + **Vision** + Tools, 256K ctx | ✅ Yes (swap mode) |
| `ministral-3:8b` | 6.0 GB | Text + **Vision** + Tools, 256K ctx | ⚠️ Tight (~7.5 GB total) |
| `ministral-3:14b` | 9.1 GB | Text + **Vision** + Tools, 256K ctx | ❌ Too large |
| Voxtral Mini 3B GGUF | ~2.5 GB | Audio ASR + understanding | ✅ Yes (swap mode) |

**Key insight:** `ministral-3:3b` on Ollama now includes **built-in vision** (Text + Image input)
at just 3.0 GB. Our current Orin memory budget is ~5.2 GB with 2.8 GB headroom. A vision-enabled
Ministral 3B could replace Pixtral API calls entirely — the robot could **see and understand
its environment without any cloud calls**.

### Roadmap: From Hybrid to Fully On-Device

```
TODAY (Hackathon Demo):
  Text/Chat   → Ministral 3B Q4_K_M (LOCAL, Ollama, 2.0 GB)
  Vision      → Pixtral via Mistral API (CLOUD)
  ASR         → Voxtral via Mistral API (CLOUD)
  TTS         → edge-tts (LOCAL, free)
  Tools       → OpenClaw skills (LOCAL)

NEAR-TERM (weeks):
  Text+Vision → ministral-3:3b (LOCAL, Ollama, 3.0 GB) ← replaces Pixtral API
  ASR         → Voxtral Mini 3B GGUF (LOCAL, swap mode) ← replaces Voxtral API
  TTS         → edge-tts or Piper (LOCAL)
  Tools       → OpenClaw skills (LOCAL)
  🎯 Result: ZERO cloud API calls. Fully offline embodied AI.

FUTURE (months):
  Everything  → Single multimodal Mistral model (text + vision + audio + tools)
  Size        → <4 GB quantized, runs on Orin Nano alongside robot stack
  Latency     → <1s for ALL modalities
  🎯 Result: True edge AI — works on a plane, in a hospital, anywhere.
```

### Why This Matters

- **Privacy**: Patient data (healthcare use case) never leaves the device
- **Latency**: No network round-trips — sub-second response for ALL modalities
- **Reliability**: Works without internet — critical for robotics in the field
- **Cost**: $0/month API costs after initial hardware purchase
- **Mistral + NVIDIA**: Mistral builds the models that fit; NVIDIA builds the hardware that runs them. Together, they make embodied edge AI real.

---

## 🏆 Prize Strategy

| Prize | Value | How We Hit It |
|-------|-------|---------------|
| **Local 1st** | $1,500 + $3,000 credits + 3mo ElevenLabs Pro | Full multi-agent architecture — edge + cloud, physical robot on the table |
| **Local 2nd** | $1,000 + $2,000 credits | Same as above |
| **Local 3rd** | $500 + $1,000 credits | Same as above |
| **Best Voice (ElevenLabs)** | $2,000-6,000 in credits | Voice loop: mic → Voxtral ASR → Ministral 3B → ElevenLabs TTS → robot speaker |
| **Best Architectural Modification (Tilde)** | $500 + internship opportunity | Split-brain: edge model for reactions, cloud for reasoning — quantized + deployed on $249 board |
| **Best Use of Mistral Vibe** | Mistral AirPods | Used Vibe to build the project + Vibe skill for Reachy |

### Hackathon Partner Technologies Used

| Partner | Role in Hackathon | How We Use It |
|---------|-------------------|---------------|
| **Mistral AI** | Organizer + model provider | Entire model stack: Ministral 3B (edge), Voxtral (ASR), Pixtral (vision), Mistral Large (reasoning) |
| **Weights & Biases** | Organizing partner | [6 training runs tracked](https://wandb.ai/thalamus_ai/reachy-copilot) — SFT sweeps + GRPO with full metrics |
| **NVIDIA** | Organizing partner | RTX 5090 (training), Jetson Orin Nano Super (edge deployment, 8GB, 67 TOPS) |
| **ElevenLabs** | Awards sponsor | High-quality TTS for robot voice output via Reachy speaker |
| **Hugging Face** | Awards sponsor | Model hosting (Ministral 3B), Transformers + TRL + PEFT for SFT/GRPO training |

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

### Libraries & Hackathon Partners
[Weights & Biases](https://wandb.ai/) *(hackathon partner — experiment tracking)* ·
[Hugging Face Transformers](https://github.com/huggingface/transformers) *(hackathon partner — model hosting + training)* ·
[ElevenLabs](https://elevenlabs.io/) *(hackathon partner — TTS)* ·
[TRL](https://github.com/huggingface/trl) ·
[PEFT](https://github.com/huggingface/peft) ·
[Ollama](https://ollama.ai/) ·
[FastAPI](https://fastapi.tiangolo.com/) ·
[Piper TTS](https://github.com/rhasspy/piper) ·
[llama.cpp](https://github.com/ggerganov/llama.cpp)

### Research Inspiration
- **DeepSeek-R1** — GRPO reinforcement learning approach for reasoning
- **Mistral Agent Skills** standard — tool-calling format
- **Split-brain architecture** pattern — edge reactions + cloud reasoning

---

## 📜 License

Training code: MIT. Models: Apache 2.0 (Ministral 3 family).
