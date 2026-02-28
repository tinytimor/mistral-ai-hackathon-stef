# 🎯 HACKATHON BATTLE PLAN — Final Recommendation

**Decision: "Project Reachy Copilot" — Embodied AI Assistant powered by OpenClaw + Mistral + Azure Foundry**

---

## Why This Wins

### 1. Physical Hardware = Instant Wow Factor
95% of hackathon teams will demo a web app or a chatbot. You will have a **robot on the table** that moves, talks, sees, and reacts. Judges remember the thing that moved. Period.

### 2. Full Mistral Ecosystem Story
You demonstrate the **entire Mistral model family** in one project:
- **Mistral Large** (via Azure Foundry) → complex reasoning, tool orchestration
- **Ministral 3B** (quantized, on Orin Nano) → real-time edge inference
- **Mistral Vibe** → you built the project *with* Vibe (meta)
- **Agent Skills** → modular skill system via OpenClaw SKILL.md format

### 3. Targets Multiple Prize Categories
| Prize | How You Hit It |
|---|---|
| **Local 1st-3rd** | Novel physical AI + edge + cloud architecture |
| **Best Voice (ElevenLabs)** | Voice interaction loop: wake → STT → LLM → TTS through robot |
| **Best Use of Mistral Vibe** | Use Vibe to build the project; create a Vibe skill for Reachy |
| **Best Architectural Modification** | Split-brain architecture: edge model for reactions, cloud model for reasoning |

### 4. Azure Foundry = Show-Your-Bosses Story
- Mistral Large deployed as serverless MaaS endpoint on Azure AI Foundry
- Demonstrates enterprise model routing: edge-first with cloud fallback
- Healthcare spin: "Patient-facing companion that runs on $249 hardware, scales to Azure"

### 5. OpenClaw = Hottest Open-Source Project (240K ⭐)
- Using the #1 trending open-source AI assistant as your orchestration layer adds credibility
- Multi-channel: the robot can also respond via WhatsApp/Telegram/Teams
- Skills system maps perfectly to Mistral Agent Skills standard

---

## Architecture (Simplified for Hackathon)

```
┌──────────────────────────────────────────────────────┐
│              RTX 5090 Desktop (32GB)                  │
│                                                      │
│   OpenClaw Gateway ──── Mistral Large (Azure API)    │
│      │                                               │
│      ├── Session memory & context                    │
│      ├── Reachy Mini Skill (HTTP → Orin)             │
│      ├── Healthcare Skills (optional)                │
│      └── ElevenLabs Voice Skill                      │
└──────────────────┬───────────────────────────────────┘
                   │ LAN / Tailscale
                   ▼
┌──────────────────────────────────────────────────────┐
│           Jetson Orin Nano Super (8GB)                │
│                                                      │
│   Ministral 3B Q4 (llama.cpp) ← fast reactions       │
│   Whisper.cpp (tiny) ← real-time STT                 │
│   Piper TTS ← speech synthesis                       │
│   FastAPI Bridge ← robot control endpoints            │
│   Pollen Vision ← camera perception                  │
└──────────────────┬───────────────────────────────────┘
                   │ USB
                   ▼
┌──────────────────────────────────────────────────────┐
│              Reachy Mini Robot                        │
│   Camera │ 4 Mics │ Speaker │ 6-DOF Head             │
└──────────────────────────────────────────────────────┘
```

---

## ⏰ HOUR-BY-HOUR BUILD SCHEDULE

### Phase 1: Foundation (Saturday, Now → +3 hours) — MUST COMPLETE
**Goal: Robot moves, speaks, and responds to voice**

| Time | Task | Status |
|---|---|---|
| Hour 0-1 | Set up Orin Nano: install llama.cpp with CUDA, download Ministral 3B Q4 GGUF, verify inference | ⬜ |
| Hour 0-1 (parallel on desktop) | Install OpenClaw on 5090 desktop, configure with Mistral API key | ⬜ |
| Hour 1-2 | Connect Reachy Mini SDK on Orin, test head movements + emotions + camera | ⬜ |
| Hour 2-3 | Build FastAPI bridge on Orin (look_at, express, speak, see, listen endpoints) | ⬜ |

**Milestone 1:** Robot moves its head and plays emotions via HTTP calls ✅

### Phase 2: Voice Pipeline (Saturday, +3h → +5h) — MUST COMPLETE  
**Goal: Full voice interaction loop**

| Time | Task | Status |
|---|---|---|
| Hour 3-4 | Set up Whisper.cpp (tiny) on Orin for real-time STT from robot mics | ⬜ |
| Hour 3-4 (parallel) | Set up Piper TTS on Orin for speech through robot speaker | ⬜ |
| Hour 4-5 | Wire the loop: Mic → Whisper STT → Ministral 3B → Piper TTS → Speaker | ⬜ |
| Hour 5 | Add head tracking: robot looks at speaker, nods, reacts with emotions | ⬜ |

**Milestone 2:** You can talk to the robot and it talks back with expressions ✅

### Phase 3: OpenClaw Integration (Saturday, +5h → +8h) — HIGH PRIORITY
**Goal: OpenClaw orchestrates everything, multi-channel**

| Time | Task | Status |
|---|---|---|
| Hour 5-6 | Create OpenClaw Reachy Mini skill (SKILL.md + tool definitions) | ⬜ |
| Hour 6-7 | Wire OpenClaw to use Azure Foundry Mistral Large for complex queries | ⬜ |
| Hour 7-8 | Set up smart routing: simple queries → local Ministral 3B, complex → OpenClaw → Mistral Large | ⬜ |
| Hour 8 | Test: send a WhatsApp/Telegram message → robot responds physically | ⬜ |

**Milestone 3:** Full agentic robot controlled by OpenClaw with cloud intelligence ✅

### Phase 4: Healthcare Demo Layer (Saturday night, +8h → +11h) — NICE TO HAVE
**Goal: Add healthcare-specific capabilities**

| Time | Task | Status |
|---|---|---|
| Hour 8-9 | Create "Patient Companion" skill: medication reminders, appointment prep, comfort checks | ⬜ |
| Hour 9-10 | Add vision skill: robot describes what it sees (can identify objects, read text with Pixtral) | ⬜ |
| Hour 10-11 | Create ambient monitoring demo: robot tracks user attention, offers proactive assistance | ⬜ |

### Phase 5: ElevenLabs Voice + Polish (Saturday night, +11h → +14h) — NICE TO HAVE
**Goal: Premium voice + demo recording**

| Time | Task | Status |
|---|---|---|
| Hour 11-12 | Swap Piper TTS for ElevenLabs WebSocket API for premium voice quality | ⬜ |
| Hour 12-13 | Add distinct voice personalities (professional mode vs casual mode) | ⬜ |
| Hour 13-14 | Create "Jarvis personality" via OpenClaw SOUL.md tuning | ⬜ |

### Phase 6: Demo Prep (Sunday, 6:00-8:30 AM) — MUST COMPLETE
**Goal: Bulletproof demo + submission**

| Time | Task | Status |
|---|---|---|
| 6:00-7:00 AM | Record a 2-minute demo video (in case you can't present live in NYC) | ⬜ |
| 7:00-7:30 AM | Write project README with architecture diagram, screenshots | ⬜ |
| 7:30-8:00 AM | Prepare 90-second pitch script for the video | ⬜ |
| 8:00-8:30 AM | Submit project, pack up hardware if traveling with it | ⬜ |

---

## Key Commands to Get Started RIGHT NOW

### On the 5090 Desktop:
```bash
# Install OpenClaw
curl -fsSL https://openclaw.ai/install.sh | bash
openclaw onboard --install-daemon

# Set up Mistral API key (get from console.mistral.ai)
# During onboard, configure model as: mistral/mistral-large-latest
# Or for Azure Foundry: azure/mistral-large
```

### On the Orin Nano:
```bash
# Build llama.cpp with CUDA
sudo apt update && sudo apt install -y cmake build-essential
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON && cmake --build . -j$(nproc)

# Download Ministral 3B
wget https://huggingface.co/bartowski/Ministral-3B-Instruct-GGUF/resolve/main/Ministral-3B-Instruct-Q4_K_M.gguf

# Start inference server
./build/bin/llama-server -m Ministral-3B-Instruct-Q4_K_M.gguf -ngl 99 --port 8080

# Install Reachy Mini SDK
pip install reachy-mini

# Install voice stack
pip install whisper-cpp-python piper-tts fastapi uvicorn
```

---

## 90-Second Pitch Script (Draft)

> "Meet Reachy — your personal AI copilot that lives on your desk, not in your browser.
>
> Reachy runs on a $249 Jetson Orin Nano with a 3-billion parameter Mistral model — completely local, completely private. It listens through its mics, sees through its camera, speaks through its speaker, and expresses emotions through physical movement.
>
> But here's what makes it different: it's orchestrated by OpenClaw, the open-source personal AI gateway with 240,000 stars. When Reachy encounters a complex question, it seamlessly escalates to Mistral Large running on Azure AI Foundry — enterprise-grade reasoning, edge-first privacy.
>
> You can talk to Reachy face-to-face, or message it on WhatsApp, Telegram, or Microsoft Teams. Same brain, any channel.
>
> For healthcare? Imagine this on every patient bedside table. A companion that reminds you about medications, preps you for appointments, and alerts your care team — all running locally on $249 hardware with cloud intelligence when needed.
>
> This is the Mistral model family working together: Ministral 3B at the edge, Mistral Large in the cloud, Agent Skills for extensibility. Physical AI, not another chatbot."

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Reachy Mini SDK issues | Use MuJoCo simulation as fallback (built into SDK) |
| Orin Nano memory issues | Drop to Ministral 3B Q3 or use API-only mode through 5090 |
| OpenClaw setup complexity | **Fork clawd-reachy-mini** (Artur Skowronski) — it's already 80% of what we need |
| Can't present in person | Record demo video before leaving; submit project online |
| Model not agentic enough | Use Mistral Large API as fallback; local model is a bonus |
| Internet search fails | DuckDuckGo (`pip install ddgs`) — free, no API key required |

---

## 🧪 FINETUNING PIPELINE (Tonight, Pre-Travel)

> **Full details:** See [notes/deep-research-synthesis.md](notes/deep-research-synthesis.md)

### Strategy: Distill Mistral Large → Ministral 3B via TRL

```
Stage 1 (1hr):  Generate 500+ tool-calling conversations with Mistral Large API
Stage 2 (1hr):  SFT/GKD with QLoRA on RTX 5090 (Ministral 3B student)
Stage 3 (1hr):  Optional GRPO pass with tool-use reward functions  
Stage 4 (30m):  Quantize to Q4_K_M GGUF → test on Orin Nano
```

**Key TRL features to use:**
- `GRPOTrainer` with `tools=[search_web, robot_express, robot_speak]` — built-in agent training!
- `GKDTrainer` — distill from Mistral 7B teacher → Ministral 3B student
- `SFTTrainer` + `LoraConfig` — fastest option for 500 examples

**Why this matters for judges:**
> "We used HuggingFace TRL's GRPO trainer to teach a 3B model to use tools through 
> reinforcement learning with verifiable rewards — the same technique from DeepSeek-R1."

### Internet Search Tool

```python
from duckduckgo_search import DDGS
# No API key needed! Register as Mistral function-calling tool.
results = DDGS().text("query", max_results=5)
```

### Community Projects to Build On

| Project | Author | Use |
|---|---|---|
| **clawd-reachy-mini** | Artur Skowronski | Fork as foundation — has OpenClaw+Reachy voice interface, ElevenLabs, emotions |
| **VisionClaw** | sseanliu | Architecture pattern — single `execute` tool routing through OpenClaw |

Install clawd-reachy-mini: `uv sync --extra dev --extra audio && uv run clawd-reachy --gateway-host 127.0.0.1`
| Voice quality issues | Piper TTS is good enough; ElevenLabs is a polish layer |

---

## Why NOT the Other Options

| Option | Why Skip |
|---|---|
| Pediatric Hospital Simulator | No physical wow factor, pure software = looks like every other team |
| Code Blue Multi-Agent Sim | Impressive but no hardware differentiator |
| Pathology Active Search | Needs labeled data, niche appeal, no physical element |
| MARL Game (Supercell) | Too far from your expertise and hardware strengths |
| LayerSkip Architectural Mod | Research-heavy, hard to demo, risky for 24 hours |

**The robot wins because it's the one thing nobody else at the hackathon will have.**

---

## After the Hackathon (Bonus)

If you win or just want to continue:
- Fine-tune Ministral 3B with tool-use data on the 5090 (Phase 4 of the Jarvis plan)
- Deploy as an Azure IoT Edge module for enterprise customers
- Package as a "Healthcare Companion Kit" demo for Microsoft Health & Life Sciences customers
- Publish the Reachy Mini skill to ClawHub for the OpenClaw community
