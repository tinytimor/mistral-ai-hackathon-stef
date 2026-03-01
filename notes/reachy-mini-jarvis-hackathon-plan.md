# 🦞🤖 Reachy Mini + OpenClaw + Orin Nano - Personal Jarvis Hackathon Plan

**Mistral Hackathon Project - Feasibility & Architecture Guide**
**Date:** February 28, 2026

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware & Software Stack](#hardware--software-stack)
- [Architecture](#architecture)
  - [Option A: Split Architecture (Recommended)](#option-a-split-architecture-recommended)
  - [Option B: Everything on Orin Nano](#option-b-everything-on-orin-nano)
- [OpenClaw Overview](#openclaw-overview)
- [Making Reachy Mini an OpenClaw Skill](#making-reachy-mini-an-openclaw-skill)
- [Jarvis Pipeline](#jarvis-pipeline)
- [Fine-Tuning & Distillation Strategy](#fine-tuning--distillation-strategy)
- [Azure Microsoft Foundry Budget](#azure-microsoft-foundry-budget)
- [Feasibility Summary](#feasibility-summary)
- [Getting Started - Step by Step](#getting-started--step-by-step)

---

## Project Overview

Build a **personal Jarvis assistant** that:

1. Runs on a **Jetson Orin Nano Super** connected to a **Reachy Mini** robot
2. Uses **Mistral models** locally (Ministral 3B / Mistral 7B quantized)
3. Is orchestrated by **OpenClaw** - an open-source personal AI assistant gateway
4. Leverages a **5090 GPU** for fine-tuning/distillation of agentic tool-use behavior
5. Uses **Azure Microsoft Foundry** ($150/mo credits) for teacher models, evaluation, and fallback

**Hackathon Narrative:** Demonstrating the full Mistral model family - frontier (Mistral Large as teacher) → distillation → edge deployment (Ministral 3B on a $249 Jetson) - powering a physical robot assistant orchestrated by OpenClaw.

---

## Hardware & Software Stack

### Hardware

| Component | Specs | Role |
|---|---|---|
| **Jetson Orin Nano Super** | 67 TOPS, 8GB unified RAM, 1024-core Ampere GPU, $249 | Edge inference (STT, LLM, TTS) + Reachy Mini brain |
| **Reachy Mini** | Camera, 4 mics, 5W speaker, 6-DOF head, body rotation | Physical embodiment - the robot |
| **RTX 5090** | 32GB GDDR7 | Fine-tuning rig + OpenClaw gateway host + heavy model inference |

### Software

| Component | Technology | Purpose |
|---|---|---|
| **Agentic Framework** | [OpenClaw](https://github.com/openclaw/openclaw) (TypeScript, Node.js ≥22) | Personal AI assistant gateway - sessions, memory, channels, skills |
| **Local LLM Inference** | llama.cpp / Ollama with CUDA | Run Mistral models on Orin Nano GPU |
| **Speech-to-Text** | Whisper.cpp (tiny/base) | Real-time transcription on Orin |
| **Text-to-Speech** | Piper TTS or Coqui TTS | Speech synthesis on Orin |
| **Robot SDK** | [reachy_mini](https://github.com/pollen-robotics/reachy_mini) (Python) | Control Reachy Mini head, antennas, body |
| **Vision** | Pollen Vision (zero-shot detection) | Camera-based perception |
| **Fine-tuning** | unsloth / axolotl (QLoRA) | Train LoRA adapters on 5090 |
| **Quantization** | llama.cpp GGUF export | Convert fine-tuned model for edge deployment |
| **Cloud Models** | Azure Microsoft Foundry - Mistral Large | Teacher model for distillation + fallback |

### Mistral Models by Deployment Target

| Model | Quantization | Size | Target | Use Case |
|---|---|---|---|---|
| **Ministral 3B** | Q4_K_M GGUF | ~2GB | Orin Nano | Fast local inference, real-time responses |
| **Mistral 7B** | Q4_K_M GGUF | ~4GB | Orin Nano | Higher quality, ~5-10 tok/s |
| **Mistral 7B** | FP16/BF16 | ~14GB | 5090 | Fine-tuning base, OpenClaw gateway model |
| **Mistral Large** | API | N/A | Azure Foundry | Teacher model for distillation |

---

## Architecture

### Option A: Split Architecture (Recommended)

```
┌─────────────────────────────────────────────────┐
│          5090 Desktop (or any Linux box)         │
│  ┌───────────────────────────────────────────┐   │
│  │         OpenClaw Gateway (Node.js)        │   │
│  │  • Session memory & persistence           │   │
│  │  • Multi-channel (WhatsApp/Telegram/etc)  │   │
│  │  • Skill orchestration                    │   │
│  │  • Browser control, cron, webhooks        │   │
│  │  • Model routing & failover               │   │
│  └──────────┬──────────┬─────────────────────┘   │
│             │          │                         │
│    ┌────────▼──┐  ┌────▼──────────────────┐      │
│    │ Mistral   │  │ Azure Foundry API     │      │
│    │ Large/Med │  │ (fallback / teacher   │      │
│    │ (local,   │  │  for distillation)    │      │
│    │  5090)    │  │                       │      │
│    └───────────┘  └───────────────────────┘      │
└─────────────────────┬───────────────────────────┘
                      │ Network (LAN / Tailscale)
                      ▼
┌─────────────────────────────────────────────────┐
│           Jetson Orin Nano Super (8GB)           │
│  ┌───────────────────────────────────────────┐   │
│  │  Ministral 3B Q4 (llama.cpp / Ollama)    │   │
│  │  • Fast local inference for latency-     │   │
│  │    sensitive responses (head movement,    │   │
│  │    emotion reactions, quick replies)      │   │
│  ├───────────────────────────────────────────┤   │
│  │  Whisper.cpp (tiny) - real-time STT      │   │
│  │  Piper TTS - speech synthesis            │   │
│  ├───────────────────────────────────────────┤   │
│  │  Reachy Mini Bridge Service (Python)     │   │
│  │  • Exposes robot actions as OpenClaw     │   │
│  │    skill or ACP endpoint                 │   │
│  │  • Camera feed → vision pipeline         │   │
│  │  • Mic/Speaker passthrough               │   │
│  └──────────────┬────────────────────────────┘   │
└─────────────────┼────────────────────────────────┘
                  │ USB / Serial
                  ▼
┌─────────────────────────────────────────────────┐
│              Reachy Mini Robot                   │
│   Camera │ 4 Mics │ Speaker │ 6-DOF Head        │
└─────────────────────────────────────────────────┘
```

**Why this works:**

- OpenClaw runs on the 5090 box where it has plenty of RAM and can hit Mistral Large locally or via Azure Foundry
- The Orin Nano handles **real-time robotics** (STT, TTS, fast reactions) with a small local model
- Complex requests route over the network to the OpenClaw gateway → bigger model
- You keep latency low for physical interactions while having full agentic power for complex tasks

### Option B: Everything on Orin Nano (Simpler but constrained)

```
┌──────────────────────────────────────────────┐
│         Jetson Orin Nano Super (8GB)         │
│                                              │
│  OpenClaw Gateway ─── Ministral 3B Q4       │
│        │                                     │
│  Whisper.cpp ─── Piper TTS                  │
│        │                                     │
│  Reachy Mini Skill (Python)                 │
│        │       ╲                             │
│        │        ╲── Azure Foundry (fallback) │
└────────┼─────────────────────────────────────┘
         │ USB
    Reachy Mini
```

**Tradeoff:** OpenClaw + Node.js uses ~200-400MB RAM, leaving ~3.5-4GB for a Q4 Ministral 3B model plus Whisper. **It fits, but barely.** You lose browser control and heavy skill execution.

---

## OpenClaw Overview

[OpenClaw](https://openclaw.ai) is an **open-source, self-hosted personal AI assistant** created by Peter Steinberger. **240k+ GitHub stars**, MIT licensed.

**Note:** OpenClaw is NOT acquired by OpenAI - OpenAI is a financial sponsor of the project. It is independent and model-agnostic.

### Key Features

| Feature | Details |
|---|---|
| **Multi-channel inbox** | WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Microsoft Teams, Google Chat, Matrix, WebChat |
| **Model-agnostic** | Anthropic, OpenAI, Mistral, local models - any provider |
| **Full system access** | Browser control (CDP), shell commands, file I/O, cron jobs, webhooks |
| **Skills/plugins** | Extensible via ClawHub skill registry; the agent can even write its own skills |
| **Voice** | Voice Wake + Talk Mode on macOS/iOS/Android (via ElevenLabs) |
| **Multi-agent** | Route channels/accounts to isolated agents, agent-to-agent sessions |
| **Companion apps** | macOS menu bar app, iOS node, Android node |
| **Persistent memory** | Sessions and context carry across conversations 24/7 |
| **Node system** | Register external devices (iOS, Android, custom) that expose device-local actions |

### Quick Install

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
# Or via npm
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

### Relevant Links

- GitHub: https://github.com/openclaw/openclaw
- Docs: https://docs.openclaw.ai
- Skills: https://clawhub.com
- ACP Client: https://github.com/openclaw/acpx
- Discord: https://discord.gg/clawd

---

## Making Reachy Mini an OpenClaw Skill

### Skill Definition

Create a skill at `~/.openclaw/workspace/skills/reachy-mini/SKILL.md` that exposes robot actions as tools:

| Tool | What it does |
|---|---|
| `reachy.look_at(x, y, z)` | Move head to look at a position |
| `reachy.express(emotion)` | Play an emotion animation (happy, confused, thinking...) |
| `reachy.speak(text)` | TTS through the robot's speaker |
| `reachy.listen()` | Record and transcribe from the robot's mics |
| `reachy.see()` | Capture and describe what the camera sees |
| `reachy.dance(name)` | Play a dance from the Reachy Mini dance library |

### Bridge Implementation Options

1. **Simple HTTP bridge (recommended for hackathon):** Python FastAPI on the Orin Nano wrapping the Reachy Mini SDK, called from an OpenClaw skill
2. **OpenClaw ACP (Agent Client Protocol):** Orin Nano runs a lightweight ACP server via acpx
3. **OpenClaw node system:** Register Orin Nano as an OpenClaw node exposing device-local actions via node.invoke

### Example Bridge (Python/FastAPI on Orin Nano)

```python
from fastapi import FastAPI
from reachy_mini import ReachyMini

app = FastAPI()
robot = ReachyMini()

@app.post("/look_at")
async def look_at(x: float, y: float, z: float):
    robot.head.look_at(x, y, z)
    return {"status": "ok"}

@app.post("/express")
async def express(emotion: str):
    robot.play_emotion(emotion)
    return {"status": "ok"}

@app.post("/speak")
async def speak(text: str):
    # Route through local Piper TTS -> speaker
    audio = tts.synthesize(text)
    robot.speaker.play(audio)
    return {"status": "ok"}

@app.post("/see")
async def see():
    frame = robot.camera.capture()
    description = vision_model.describe(frame)
    return {"description": description}
```

### Corresponding OpenClaw Skill (SKILL.md)

```markdown
# Reachy Mini Robot Control

You have access to a Reachy Mini robot connected via HTTP.
Base URL: http://<orin-nano-ip>:8000

## Tools

- POST /look_at - Move the robot's head. Body: {"x": float, "y": float, "z": float}
- POST /express - Show an emotion. Body: {"emotion": "happy|confused|thinking|excited|sad"}
- POST /speak - Say something through the robot's speaker. Body: {"text": "..."}
- POST /see - Take a photo and describe what the robot sees. Returns {"description": "..."}
- POST /dance - Play a dance animation. Body: {"name": "wave|nod|shake|celebrate"}
```

---

## Jarvis Pipeline

The end-to-end interaction flow:

```
User speaks → Reachy Mini Mics
                    │
                    ▼
         Whisper.cpp (Orin Nano)
            STT → text
                    │
        ┌───────────┴───────────┐
        │ Simple/fast query?    │
        │                       │
    ┌───▼───┐             ┌─────▼─────┐
    │Local  │             │ OpenClaw  │
    │Ministral│           │ Gateway   │
    │3B Q4  │             │ (5090)    │
    │(Orin) │             │ → Mistral │
    └───┬───┘             │   Large   │
        │                 └─────┬─────┘
        └───────────┬───────────┘
                    │
                    ▼
            Piper TTS (Orin Nano)
              text → audio
                    │
                    ▼
         Reachy Mini Speaker + Head Motion
```

### Component Breakdown

| Component | Tool | Runs on | Notes |
|---|---|---|---|
| Speech-to-Text | Whisper.cpp (tiny/base) | Orin Nano | Real-time, ~200MB RAM |
| Fast LLM (reactions) | Ministral 3B Q4 GGUF | Orin Nano | ~2GB, instant emotion/action decisions |
| Full LLM (complex) | Mistral 7B+ or Mistral Large | 5090 / Azure Foundry | Complex reasoning, tool use, planning |
| Text-to-Speech | Piper TTS | Orin Nano | Fast, lightweight, many voices |
| Vision | Pollen Vision / CLIP | Orin Nano | Zero-shot object detection |
| Robot Control | Reachy Mini Python SDK | Orin Nano | Head, antennas, body rotation |
| Orchestration | OpenClaw Gateway | 5090 | Sessions, memory, multi-channel, skills |

### Existing Reference

Pollen Robotics already has a conversation app for Reachy Mini at https://github.com/pollen-robotics/reachy_mini_conversation_app - adapt this to use your local Mistral model and wire it as an OpenClaw bridge.

---

## Fine-Tuning & Distillation Strategy

### What to Fine-Tune For

| Behavior | Why | Training Signal Source |
|---|---|---|
| **OpenClaw tool-use format** | Model needs to call OpenClaw skills correctly (JSON tool calls) | Synthetic data from Mistral Large via Azure Foundry |
| **Reachy Mini action selection** | When to move head, express emotions, look at user | Record real interaction sessions, annotate with correct actions |
| **Conversational personality** | "Jarvis" style - concise, proactive, dry wit | Curated dialogue examples |
| **Multi-step planning** | "Check my calendar, then remind me via WhatsApp, and nod when done" | Chain-of-thought traces from Mistral Large |

### Distillation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Azure Foundry ($150/mo)                 │
│                                                         │
│  1. Deploy Mistral Large as teacher model               │
│  2. Generate 5-10K synthetic tool-use examples          │
│     using your actual OpenClaw skill definitions        │
│  3. Generate multi-turn Jarvis-style conversations      │
│  4. Run evaluation with Foundry eval tools              │
│                                                         │
│  Cost: ~$50-80 of your $150 budget                      │
└────────────────────┬────────────────────────────────────┘
                     │ training dataset (JSONL)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                RTX 5090 (32GB VRAM)                      │
│                                                         │
│  5. QLoRA fine-tune Ministral 3B on the dataset         │
│     • unsloth or axolotl for fast LoRA training         │
│     • ~2-4 hours for 5K examples                        │
│  6. Merge LoRA weights into base model                  │
│  7. Quantize to GGUF Q4_K_M (~2GB file)                │
│  8. Test locally with llama.cpp before deploying        │
│                                                         │
│  Also fine-tune a Mistral 7B version for the 5090       │
│  to power OpenClaw's gateway with better tool use       │
└────────────────────┬────────────────────────────────────┘
                     │ .gguf file (~2GB)
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Jetson Orin Nano Super                      │
│                                                         │
│  9. Drop GGUF into Ollama/llama.cpp                     │
│  10. Wire as OpenClaw's model endpoint                  │
│      OR run as the local fast-response model            │
│                                                         │
│  Result: Ministral 3B that *natively understands*       │
│  your Reachy Mini tools and Jarvis personality          │
└─────────────────────────────────────────────────────────┘
```

### Fine-Tuning Commands (Quick Reference)

```bash
# Install unsloth (on 5090)
pip install unsloth

# QLoRA fine-tune Ministral 3B
python finetune.py \
  --model mistralai/Ministral-3B-Instruct \
  --dataset ./jarvis-tool-use-dataset.jsonl \
  --lora_r 64 \
  --lora_alpha 128 \
  --epochs 3 \
  --batch_size 4 \
  --output ./ministral-3b-jarvis-lora

# Merge LoRA
python merge_lora.py \
  --base mistralai/Ministral-3B-Instruct \
  --lora ./ministral-3b-jarvis-lora \
  --output ./ministral-3b-jarvis-merged

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py ./ministral-3b-jarvis-merged
llama.cpp/llama-quantize \
  ./ministral-3b-jarvis-merged.gguf \
  ./ministral-3b-jarvis-Q4_K_M.gguf Q4_K_M

# Deploy to Orin Nano
scp ./ministral-3b-jarvis-Q4_K_M.gguf orin-nano:~/models/
```

---

## Azure Microsoft Foundry Budget

**Monthly budget: $150/mo**

| Use Case | Estimated Cost | Purpose |
|---|---|---|
| Mistral Large API (teacher) | ~$50-80 | Generate 5-10K synthetic training examples |
| Evaluation tools | ~$10 | Benchmark fine-tuned vs base model |
| Fallback inference | ~$20-40 | Cloud fallback when Orin can't handle a query |
| Voxtral / Whisper hosting | ~$20 | Optional: offload STT if too heavy on-device |
| **Total** | **~$95-135** | **Within budget** |

### Hybrid Strategy

- **Edge-first:** Most requests handled locally on Orin Nano (Ministral 3B)
- **Cloud-fallback:** Complex queries escalated to Azure Foundry Mistral Large
- **Hackathon pitch:** "Runs locally on $249 hardware, but can scale to cloud when needed"

---

## Feasibility Summary

| Component | Feasibility | Hackathon-ready? | Notes |
|---|---|---|---|
| OpenClaw on 5090 desktop as gateway | ✅ Trivial | ✅ Yes | `npm install -g openclaw` |
| Orin Nano as Reachy Mini brain | ✅ High | ✅ Yes | Python SDK + llama.cpp |
| Reachy Mini as OpenClaw skill | ✅ High | ✅ Yes | Python FastAPI bridge |
| OpenClaw multi-channel (text your Jarvis) | ✅ High | ✅ Yes | Built-in WhatsApp/Telegram |
| Ministral 3B local on Orin | ✅ High | ✅ Yes | Q4 fits easily in 8GB |
| Voice (STT + TTS) on Orin | ✅ High | ✅ Yes | Whisper tiny + Piper |
| Fine-tune on 5090 | ✅ High | ✅ Yes | QLoRA, 2-4 hours |
| Distillation via Azure Foundry | ✅ High | ✅ Yes | Within $150 budget |
| Deploy LoRA/GGUF to Orin | ✅ Trivial | ✅ Yes | Copy one file |
| All-on-Orin (no desktop) | ⚠️ Tight | ⚠️ Possible | Memory-constrained |

---

## Getting Started - Step by Step

### Phase 1: Core Setup (Day 1)

1. **Install OpenClaw on 5090 desktop:**
   ```bash
   curl -fsSL https://openclaw.ai/install.sh | bash
   openclaw onboard --install-daemon
   ```

2. **Set up Orin Nano with JetPack 6 + llama.cpp:**
   ```bash
   # On Orin Nano
   sudo apt update && sudo apt install -y cmake build-essential
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && mkdir build && cd build
   cmake .. -DGGML_CUDA=ON && cmake --build . -j$(nproc)
   ```

3. **Download and test Ministral 3B:**
   ```bash
   wget https://huggingface.co/bartowski/Ministral-3B-Instruct-GGUF/resolve/main/Ministral-3B-Instruct-Q4_K_M.gguf
   ./llama-server -m Ministral-3B-Instruct-Q4_K_M.gguf -ngl 99 --port 8080
   ```

4. **Connect Reachy Mini SDK:**
   ```bash
   pip install reachy-mini
   python -c "from reachy_mini import ReachyMini; r = ReachyMini(); print('Connected!')"
   ```

### Phase 2: Bridge & Skills (Day 2)

5. Build the FastAPI bridge on Orin Nano
6. Create the OpenClaw Reachy Mini skill
7. Wire OpenClaw to use the Orin Nano's Ministral endpoint as a model
8. Test end-to-end: message via Telegram → OpenClaw → Reachy Mini moves + speaks

### Phase 3: Voice Pipeline (Day 3)

9. Set up Whisper.cpp on Orin for real-time STT
10. Set up Piper TTS for speech synthesis
11. Wire mic input → STT → LLM → TTS → speaker output loop

### Phase 4: Fine-Tuning (Day 4-5)

12. Generate synthetic training data via Azure Foundry Mistral Large
13. QLoRA fine-tune Ministral 3B on 5090
14. Quantize, deploy GGUF to Orin Nano
15. A/B test fine-tuned vs base model

### Phase 5: Polish & Demo (Day 6-7)

16. Tune Jarvis personality in OpenClaw's SOUL.md
17. Add cron jobs (morning briefing, proactive check-ins)
18. Record demo video showing the full pipeline
19. Prepare hackathon presentation highlighting Mistral model family story

---

## Key Links

| Resource | URL |
|---|---|
| OpenClaw GitHub | https://github.com/openclaw/openclaw |
| OpenClaw Docs | https://docs.openclaw.ai |
| ClawHub (Skills) | https://clawhub.com |
| Reachy Mini SDK | https://github.com/pollen-robotics/reachy_mini |
| Reachy Mini Conversation App | https://github.com/pollen-robotics/reachy_mini_conversation_app |
| Pollen Robotics (HuggingFace) | https://huggingface.co/pollen-robotics |
| Reachy Mini Emotions Library | https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library |
| Reachy Mini Dances Library | https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library |
| Amazing Hand (open-source hand) | https://github.com/pollen-robotics/AmazingHand |
| PincOpen (open-source gripper) | https://github.com/pollen-robotics/PincOpen |
| Mistral AI Models (HuggingFace) | https://huggingface.co/mistralai |
| NVIDIA Jetson AI Lab | https://www.jetson-ai-lab.com |
| NVIDIA Jetson Orin | https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/ |
| llama.cpp | https://github.com/ggerganov/llama.cpp |
| Whisper.cpp | https://github.com/ggerganov/whisper.cpp |
| Piper TTS | https://github.com/rhasspy/piper |
| Azure AI Foundry | https://ai.azure.com |

---

*Good luck at the hackathon! 🦞🤖🔥*
