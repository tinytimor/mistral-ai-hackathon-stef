# 🤖 AGENTS.md — Reachy Copilot Agent Architecture

> Embodied AI assistant running on the edge — fine-tuned Ministral 3B
> via Ollama + OpenClaw Gateway + clawd-reachy-mini on Jetson Orin Nano,
> with Mistral API for vision (Pixtral), voice (Voxtral ASR), and complex
> reasoning (Mistral Large). For the Mistral Worldwide Hackathon 2026.

> ⚠️ **Work in Progress** — Built in 2 days (Feb 28–Mar 1, 2026).
> Core pipeline is working: local Ministral 3B + Reachy Mini robot + Mistral API.
> See Section 13 for the roadmap toward fully offline multimodal on-device.

---

## Agent Overview

**Everything runs on the Orin Nano.** The RTX 5090 is only for training.

```
┌──────────────────────────────────────────────────────────────────────────┐
│              ORIN NANO SUPER (10.0.0.232) — All-in-One Edge             │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ 🦞 OpenClaw      │  │ ⚡ Ollama        │  │ 🎤 clawd-reachy-mini │  │
│  │   Gateway        │◄─│   reachy-copilot │  │   (Voice + Robot)    │  │
│  │   :18789         │  │   :11434         │  │   Whisper STT        │  │
│  │   Skills + Mem   │  │   Ministral 3B   │  │   ElevenLabs TTS     │  │
│  │   Multi-Channel  │  │   Q4_K_M (2 GB)  │  │   Wake Word          │  │
│  └────────┬─────────┘  └──────────────────┘  └──────────┬────────────┘  │
│           │                                             │               │
│           │          ┌──────────────────┐                │               │
│           └─────────►│ 🤖 Reachy Mini  │◄───────────────┘               │
│                      │   gRPC :50051   │                                │
│                      │   Head + Camera │                                │
│                      │   4 Mics + Spkr │                                │
│                      └──────────────────┘                               │
│                                                                          │
│  ── Cloud Fallback (Mistral API) ──────────────────────────────────────  │
│  │  mistral/mistral-large-latest — complex reasoning only              │ │
│  └─────────────────────────────────────────────────────────────────────  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Reasoning Agent (Cloud Fallback)

**Model:** Mistral Large 3 (675B MoE, 41B active)
**Location:** Mistral API (`mistral/mistral-large-latest`)
**Role:** Complex reasoning fallback when the edge model can't handle a query

### Capabilities
- Multi-step task decomposition (think → plan → act → reflect)
- Complex tool chaining (search → analyze → respond → act)
- Healthcare reasoning (patient summaries, medication interactions)
- Contextual memory queries across long conversation histories

### When Invoked
- User asks complex, multi-step questions
- Requests requiring web search + analysis + robot action
- Healthcare-related queries needing careful reasoning
- Tasks that require planning across multiple tools

### Configuration
```yaml
# Configured as OpenClaw fallback model in openclaw.json:
agent:
  name: reasoning-agent
  model: mistral/mistral-large-latest  # OpenClaw built-in Mistral provider
  provider: mistral
  temperature: 0.7
  max_tokens: 2048
  # OpenClaw routes to this when edge model escalates
  # All tools are OpenClaw skills — same tools available to both models
```

---

## 2. Reactive Agent (Orin Nano — Primary)

**Model:** Ministral 3B Q4_K_M (fine-tuned via SFT + GRPO)
**Location:** Jetson Orin Nano Super (8GB, 67 TOPS) via Ollama
**Role:** Primary agent — handles all queries locally, escalates to cloud only when needed

### How It Works
OpenClaw Gateway runs on the Orin and uses its **built-in Ollama provider** to route
all queries to our fine-tuned `reachy-copilot` model at `localhost:11434`.
No custom bridge needed — OpenClaw auto-detects Ollama.

### Capabilities
- Sub-second response to user presence (head tracking, nodding, expressions)
- Quick verbal responses via edge-tts → Reachy speaker
- Real-time speech transcription via Voxtral ASR (Mistral API — on-device planned)
- Camera-based perception via Pixtral vision (Mistral API — on-device planned)
- Tool calling for robot control, web search, email, calendar, etc.

### When Invoked
- **All queries go here first** — it's the primary model
- User speaks to robot → clawd-reachy-mini → OpenClaw → Ollama
- Wake word detection triggers listening mode
- Only escalates to Mistral Large for truly complex multi-step reasoning

### Configuration (openclaw.json)
```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "ollama/reachy-copilot",
        "fallbacks": ["mistral/mistral-large-latest"]
      }
    }
  }
}
```

### Edge Voice Model: Voxtral Mini 3B

Instead of a separate Whisper STT pipeline, we use **Voxtral Mini 3B** — Mistral's
native audio model that does ASR + audio understanding + function calling in one model.
This simplifies the pipeline from `mic → Whisper → text → LLM → tools` to `mic → Voxtral → tools`.

**Model swap strategy** (can't run both simultaneously on 8GB):
- **Text mode** (default): Ministral 3B Q4_K_M loaded, handles text tool-calling
- **Audio mode** (on demand): Voxtral Mini 3B Q4_K_M swapped in for voice tasks
- **Dual Q3 mode** (experimental): Both at Q3_K (~1.8 GB each), both loaded

### Memory Budget (Orin Nano 8GB)
```
Production mode (OpenClaw + Ollama + clawd-reachy-mini):
  Ministral 3B Q4_K_M      : ~2.0 GB   (Ollama)
  KV Cache (2048 ctx)       : ~0.5 GB
  CUDA runtime              : ~0.8 GB
  OS + JetPack              : ~1.0 GB
  OpenClaw Gateway (Node.js): ~0.3 GB
  clawd-reachy-mini (Python): ~0.4 GB   (Whisper + ElevenLabs)
  reachy-mini SDK           : ~0.2 GB
  ─────────────────────────────────
  Total                     : ~5.2 GB  ✅ (2.8 GB headroom)

With Voxtral swap (replace Ministral for voice tasks):
  Voxtral Mini 3B Q4_K_M   : ~2.5 GB   (swapped in via Ollama)
  KV Cache (2048 ctx)       : ~0.5 GB
  CUDA runtime              : ~0.8 GB
  OS + JetPack              : ~1.0 GB
  OpenClaw + clawd-reachy   : ~0.7 GB
  reachy-mini SDK           : ~0.2 GB
  ─────────────────────────────────
  Total                     : ~5.7 GB  ✅ (2.3 GB headroom)
```

---

## 3. OpenClaw Gateway (On Orin — Orchestrator)

**Model:** Routes to `ollama/reachy-copilot` (primary) or `mistral/mistral-large-latest` (fallback)
**Location:** Orin Nano (runs as daemon, port 18789)
**Role:** Session management, memory persistence, skill orchestration, multi-channel routing

### Key Insight: Built-in Ollama Provider
OpenClaw has **native Ollama support** — it auto-detects Ollama at `localhost:11434`.
No custom bridge server needed. Just set `"primary": "ollama/reachy-copilot"` in config.

### Capabilities
- **Built-in Ollama provider** — auto-detects local models, no auth needed
- **Built-in Mistral provider** — cloud fallback with API key
- Persistent memory across conversations (24/7)
- Multi-channel inbox (WhatsApp, Telegram, Discord, Slack, iMessage)
- Skill orchestration and tool routing
- Browser control (CDP) for web interactions
- Cron jobs, webhooks, proactive check-ins ("heartbeats")

### OpenClaw Configuration (`~/.openclaw/openclaw.json`)

**Verified working config (tested on Orin Nano, March 2026):**

```json
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
```

**Critical notes from actual deployment:**
- `"mode": "local"` in the `gateway` block is **required** — without it the gateway refuses to start with `Gateway start blocked`
- Do NOT add a `"bind"` key — it causes `Invalid input` errors; let OpenClaw use its loopback default
- Do NOT add a `"memory"` block — memory is built-in and not a configurable key (`Unrecognized key` error)
- Run `openclaw doctor --fix` after any manual config edit to validate the file
- Memory (short-term sliding window + long-term SQLite) is always on — no config needed

### Skills Registered
```
┌──────────────────────────────────────────────────┐
│              OpenClaw Skill Registry              │
├──────────────────────────────────────────────────┤
│ reachy-control   → Orin Bridge API (HTTP)        │
│ email-manager    → Gmail IMAP/SMTP               │
│ calendar         → Google Calendar API            │
│ web-search       → DuckDuckGo (ddgs)             │
│ browser          → CDP browser control            │
│ spotify          → Spotify Web API                │
│ smart-home       → Home Assistant / Hue           │
│ memory           → Built-in persistent memory     │
│ imessage         → macOS Messages (AppleScript)   │
│ whatsapp         → WhatsApp Business API          │
│ telegram         → Telegram Bot API               │
│ signal           → Signal CLI                     │
└──────────────────────────────────────────────────┘
```

### Smart Routing Logic
OpenClaw handles routing automatically via its model fallback chain:

```
User query → OpenClaw Gateway
                 ↓
         ollama/reachy-copilot (primary, local, <1s)
                 ↓ (if model returns "I need help with this")
         mistral/mistral-large-latest (fallback, cloud, 2-5s)
```

The fine-tuned model is trained to recognize when it needs help:
- Simple queries, robot actions, tool calls → handled locally
- Complex multi-step reasoning, healthcare analysis → model says
  "Let me think about that more carefully" → OpenClaw escalates to fallback

---

## 4. Reachy Robot Agent (Embodiment Layer)

**Hardware:** Reachy Mini (6-DOF head, antennas, camera, 4 mics, speaker)
**Location:** Connected to Orin Nano via gRPC
**Role:** Physical embodiment — translates tool calls into robot actions

### Available Actions
| Tool Call | Robot Action | Latency |
|-----------|-------------|---------|
| `robot_look_at(x, y, z)` | Head tracks to 3D point | ~100ms |
| `robot_express(emotion)` | Head + antenna animation | ~500ms |
| `robot_speak(text)` | TTS → speaker playback | ~1-2s |
| `robot_nod()` | Quick nod gesture | ~300ms |
| `robot_shake_no()` | Head shake gesture | ~400ms |
| `robot_see()` | Camera capture + vision | ~500ms |
| `robot_listen()` | Mic → Whisper STT | real-time |

### Emotion Expressions
```python
EMOTIONS = {
    "happy":     {"head": [0, -5, 0],   "antennas": [20, -20]},
    "sad":       {"head": [0, -20, 0],  "antennas": [-10, 10]},
    "curious":   {"head": [15, -5, 20], "antennas": [30, -5]},
    "surprised": {"head": [0, 5, 0],    "antennas": [40, -40]},
    "thinking":  {"head": [5, -10, 15], "antennas": [10, -10]},
    "nodding":   "special_sequence",
    "shaking":   "special_sequence",
}
```

---

## 5. Memory Architecture

### Three-Tier Memory System

```
┌────────────────────────────────────────────────────────┐
│                    MEMORY LAYERS                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  L1: Working Memory (In-Context)           ~2048 tok   │
│  ├── Current conversation turn                         │
│  ├── Active tool call results                          │
│  └── Robot state (pose, last action)                   │
│                                                        │
│  L2: Session Memory (Redis/In-Memory Cache) ~100 items │
│  ├── Recent conversation history (sliding window)      │
│  ├── User preferences learned this session             │
│  ├── Active reminders and scheduled tasks              │
│  └── Tool call cache (avoid re-fetching)               │
│                                                        │
│  L3: Long-Term Memory (SQLite / OpenClaw)   unlimited  │
│  ├── User profile and preferences                      │
│  ├── Conversation summaries (daily/weekly)             │
│  ├── Learned patterns and habits                       │
│  ├── Healthcare records (encrypted)                    │
│  └── Skill-specific persistent data                    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Memory Flow
```
User speaks → STT → Check L1 (in-context) → Check L2 (session cache)
                                                    ↓
                                           Need more context?
                                                    ↓ yes
                                           Query L3 (long-term DB)
                                                    ↓
                                           Inject into prompt
                                                    ↓
                                           LLM generates response
                                                    ↓
                                           Store new memories
                                           (L1 ← immediate, L2 ← session, L3 ← important)
```

---

## 6. Agent Communication Protocol

### Inter-Agent Messages
```json
{
  "from": "openclaw-gateway",
  "to": "reactive-agent",
  "type": "chat",
  "payload": {
    "user_message": "Hey Reachy, what's the weather?",
    "context": {
      "user_name": "Stefan",
      "location": "DC",
      "time": "2026-02-28T14:30:00Z"
    },
    "memory_context": [
      "User prefers Celsius",
      "User commutes to NYC by train"
    ]
  }
}
```

### Agent-to-Agent Escalation
```json
{
  "from": "reactive-agent",
  "to": "reasoning-agent",
  "type": "escalation",
  "reason": "complex_query",
  "payload": {
    "original_query": "Can you compare my blood test results from last month with this month and tell me if my cholesterol improved?",
    "local_attempt": "I need to look at multiple data points — let me think about this more carefully.",
    "user_context": { "patient_id": "PT-12345" }
  }
}
```

---

## 7. Training Pipeline (How Agents Are Built)

### Data Flow
```
Step 0: Download Models
    ↓ scripts/07_download_models.py (--edge-stack for full Orin stack)
    ↓ Ministral 3B (HF format for training)
    ↓ Voxtral Mini 3B GGUF (pre-quantized, for edge audio)

Mistral Large (Teacher)
    ↓ generates 500-4000 tool-calling conversations
    ↓ scripts/01_generate_training_data.py
    
Ministral 3B/8B (Student)
    ↓ SFT with QLoRA (scripts/02_sft_qlora.py)
    ↓   └─ Best-model checkpointing: 10% eval split, load_best_model_at_end
    ↓   └─ Auto-selects lowest eval_loss across hyperparameter sweep
    ↓ GRPO with reward functions (scripts/03_grpo_agent.py)
    ↓   └─ Checkpoint every 50 steps, save top 3
    ↓   └─ Auto-selects highest reward score across configs
    ↓ Quantize to Q4_K_M GGUF (scripts/04_quantize_deploy.py)
    
Deploy to Orin Nano via Ollama
    ↓ ollama create reachy-copilot -f Modelfile
    ↓ (Optional) ollama create reachy-voxtral -f Modelfile.voxtral
    
Reactive Agent is born! 🎉

Shortcut (no training, deploy pre-quantized):
    ↓ python 07_download_models.py --no-train
    ↓ scp models/ to Orin Nano
    ↓ ollama create reachy-copilot -f Modelfile
    ↓ Deploy in minutes, not hours
```

### Reward Functions (GRPO)
| Reward | What It Measures | Weight |
|--------|-----------------|--------|
| `format_correctness` | Valid `<tool_call>` JSON output | 0.25 |
| `tool_relevance` | Correct tool for the situation | 0.30 |
| `response_quality` | Empathy, completeness, professionalism | 0.25 |
| `thinking_quality` | Think-plan-act-reflect reasoning | 0.20 |

### Actual Training Results (Feb 28, 2026)

| Phase | Best Model | Metric | Value | Notes |
|-------|-----------|--------|-------|-------|
| SFT | `sft-r64-lr2e4` | eval_loss | **0.266** | LoRA r=64, lr=2e-4, 3 epochs |
| GRPO | `grpo-g4-test` | best_reward | **-0.5** | 4 generations, 1 epoch, on sft-r64-lr2e4 |
| Quantization | `model-q4_k_m.gguf` | size | **2.0 GB** | From 6.4 GB F16 → Q4_K_M |

**Key technical notes:**
- Base model is FP8-quantized on HuggingFace → requires `FineGrainedFP8Config(dequantize=True)` to load for training
- SFT outputs LoRA adapters (not full models) → GRPO must detect + merge adapter before training
- Quantize pipeline: dequant FP8 → BF16 → merge LoRA → save → convert_hf_to_gguf.py → llama-quantize Q4_K_M
- Verified tool calling works on quantized model: `search_web`, `calendar_list_events`, `look_at`, `speak`

### Experiment Tracking
All training tracked via **Weights & Biases**:
- Project: `reachy-copilot`
- Dashboard: https://wandb.ai/thalamus_ai/reachy-copilot
- Metrics: loss, learning rate, gradient norms, reward scores, GPU utilization

### Best-Model Checkpointing
The pipeline automatically selects the best model from each sweep:

| Phase | Metric | Strategy |
|-------|--------|----------|
| SFT | `eval_loss` (10% holdout) | `load_best_model_at_end=True`, lowest wins |
| GRPO | `best_reward` | Checkpoint every 50 steps, highest wins |
| Cross-run | `training_info.json` | `run_experiments.sh` compares all runs |

### Resilient Pipeline
- `set +e` after environment verification — individual failures don’t kill the pipeline
- Each phase checks prerequisites before running (no data → skip SFT/GRPO)
- Final summary reports total failure count alongside best models

---

## 8. Deployment Topology

### Hackathon Demo Setup
```
┌────────────────────────────────────────────────────────────────┐
│                    NETWORK (LAN / Tailscale)                    │
│                                                                │
│  ┌──────────────────────────┐     ┌────────────────┐           │
│  │  Orin Nano (10.0.0.232)  │     │  MacBook Pro   │           │
│  │  + Reachy Mini (gRPC)    │     │  (Control)     │           │
│  │                          │     │                │           │
│  │  Ollama :11434           │     │  VNC Viewer    │           │
│  │   └─ reachy-copilot      │     │  SSH Terminal  │           │
│  │  OpenClaw Gateway :18789 │     │  W&B Dashboard │           │
│  │   └─ Skills + Memory     │     │                │           │
│  │  clawd-reachy-mini       │     │  Monitor from  │           │
│  │   └─ Whisper + ElevenLabs│     │  anywhere      │           │
│  └──────────────────────────┘     └────────────────┘           │
│                                                                │
│  ── RTX 5090 Desktop (training only, not runtime) ────────── │
│  │  SFT + GRPO training, quantization, W&B logging           │ │
│  └────────────────────────────────────────────────────────────│ │
│                                                                │
│  ── Mistral API (Cloud Fallback) ────────────────────────────  │
│  │  mistral/mistral-large-latest (complex reasoning only)    │ │
│  └────────────────────────────────────────────────────────────  │
└────────────────────────────────────────────────────────────────┘
```

### Remote Access (Mac → Orin Nano from train)
- **VNC Server** on Orin Nano for GUI access
- **SSH tunnel** for terminal + port forwarding
- **Tailscale** for secure remote access over cellular
- **W&B Dashboard** for monitoring training from anywhere

### Deploying to Orin Nano — Step by Step

After training on the RTX 5090, deploy everything to the Orin Nano:

```bash
# ── On the RTX 5090 (training machine) ──────────────────────

# 1. Merge LoRA adapter + quantize to GGUF (produces ~2 GB file)
python scripts/04_quantize_deploy.py \
    --model models/sft-r64-lr2e4 \
    --output models/reachy-copilot-gguf \
    --llama-cpp ./llama.cpp

# 2. Copy model + Modelfile to the Orin Nano
scp models/reachy-copilot-gguf/model-q4_k_m.gguf slehman@10.0.0.232:~/reachy-model/
scp models/reachy-copilot-gguf/Modelfile slehman@10.0.0.232:~/reachy-model/

# ── On the Orin Nano (10.0.0.232) ───────────────────────────
ssh slehman@10.0.0.232

# IMPORTANT: Close Chrome and VS Code before continuing.
# The Orin uses unified CPU+GPU memory. Chrome alone eats ~1.5 GB.
pkill -f chromium; pkill -f code

# 3. Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh

# 4. Create the fine-tuned model in Ollama
# IMPORTANT: cd into the model directory — Modelfile uses a relative path
cd ~/reachy-model
ollama create reachy-copilot -f Modelfile
ollama run reachy-copilot "Hello! Search the web for weather."  # test

# 5. Install Node.js >= 22 (required for OpenClaw)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version  # should be >= 22

# 6. Install OpenClaw Gateway
# Use npm directly — the install.sh script can hang/crash on Orin
npm i -g openclaw
# Run wizard: select Mistral / your API key / mistral-large-latest
# Answer No to skills, No to channels, select "Hatch in TUI", then press q
openclaw onboard --install-daemon

# 7. Configure OpenClaw to use our Ollama model
# "mode": "local" is REQUIRED. Do NOT add "bind" or "memory" keys.
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
openclaw doctor --fix
systemctl --user start openclaw-gateway.service
# Verify: look for "[gateway] agent model: ollama/reachy-copilot"

# 8. Install uv (required for clawd-reachy-mini)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# 9. Install clawd-reachy-mini (voice + robot interface)
cd ~
git clone https://github.com/ArturSkowronski/clawd-reachy-mini.git
cd clawd-reachy-mini
# uv sync creates .venv automatically — do NOT use conda or pip for this
# Warning "'reachy-mini' does not have extra 'vision'" is harmless
# Note: downloads ~300 MB (torch, scipy, opencv, etc.) — allow 5-10 min
uv sync --extra dev --extra audio

# 10. Run clawd-reachy-mini (connects to local OpenClaw Gateway)
uv run clawd-reachy --gateway-host localhost --gateway-port 18789

# 11. Test the full pipeline (OpenClaw HTTP API)
curl -X POST http://127.0.0.1:18789/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer reachy-hackathon-2026" \
    -d '{"model": "ollama/reachy-copilot", "messages": [{"role": "user", "content": "Look at me and say hello!"}]}'
```

### Orin Nano Memory Budget (Verified)

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

### Troubleshooting on Orin

```bash
# Check Ollama is running
sudo systemctl status ollama

# Check GPU memory
tegrastats

# Check CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"

# If OOM: reduce context length in Modelfile
# Change: PARAMETER num_ctx 2048  →  PARAMETER num_ctx 1024

# List models loaded
ollama list
ollama ps   # shows currently loaded models + VRAM usage
```

---

## 9. clawd-reachy-mini Integration

The [clawd-reachy-mini](https://github.com/ArturSkowronski/clawd-reachy-mini) project provides
the voice and robot interface layer, connecting to our local OpenClaw Gateway:

### What It Provides
- WebSocket client for OpenClaw Gateway protocol
- STT backends (Whisper, Faster-Whisper, OpenAI)
- TTS via ElevenLabs
- `reachy-mini` SDK integration (head, antennas, emotions)
- Wake word detection
- Conversation loop: `mic → STT → OpenClaw Gateway → Ollama → TTS → robot`
- `action-skill/` directory with robot tool wrappers (SKILL.md format)

### How It Fits Our Architecture
1. **Connects to LOCAL OpenClaw Gateway** — `--gateway-host localhost` (not remote 5090)
2. **OpenClaw routes to our fine-tuned model** — `ollama/reachy-copilot` is the primary model
3. **Robot skills via action-skill/** — `reachy_connect`, `reachy_move_head`, `reachy_play_emotion`, `reachy_say`
4. **No custom bridge needed** — clawd-reachy-mini + OpenClaw + Ollama is the full stack

### How We Extend It
1. **Fine-tuned model** — OpenClaw uses our custom Ministral 3B instead of default model
2. **Additional OpenClaw skills** — web search, email, calendar, Spotify, etc.
3. **Memory via OpenClaw** — built-in persistent memory across sessions
4. **Vision pipeline** — Camera → Pixtral for visual understanding

### Installation (on Orin Nano)
```bash
git clone https://github.com/ArturSkowronski/clawd-reachy-mini.git
cd clawd-reachy-mini
uv sync --extra dev --extra audio

# Connect to LOCAL OpenClaw Gateway (running on same Orin)
uv run clawd-reachy --gateway-host localhost --gateway-port 18789
```

---

## 10. Architecture Inspirations

### VisionClaw Pattern
[VisionClaw](https://github.com/sseanliu/VisionClaw) (1.4k ⭐) — iOS/Android app for
Meta Ray-Ban glasses using Gemini Live + OpenClaw. Key insight we borrowed:

- **Single `execute(task: string)` tool** — the LLM gets ONE tool that delegates
  everything to OpenClaw. OpenClaw figures out which skill to use.
- **OpenAI-compatible endpoint** — `/v1/chat/completions` for seamless integration
- **Session key management** — multi-turn conversations via OpenClaw sessions

### What We Borrow vs. Build
| From **clawd-reachy-mini** | From **VisionClaw** | From **OpenClaw** | **Our additions** |
|---|---|---|---|
| reachy-mini SDK patterns | Single `execute` tool | Built-in Ollama provider | Fine-tuned Ministral 3B |
| ElevenLabs TTS | OpenAI-compat endpoint | Persistent memory | Custom tool-calling training |
| Wake word detection | Session key management | Skill system + registry | SFT + GRPO pipeline |
| Emotion animations | Multi-turn tool calling | Multi-channel messaging | Voxtral model swapping |

### Key Design Decisions
- **No custom bridge server** — OpenClaw's built-in Ollama provider replaces our `06_openclaw_bridge.py`
- **Inference is 5090-independent** — the 5090 trains models, Orin runs everything
- **All tools are OpenClaw skills** — robot, web search, email, calendar, etc.
- **Smart fallback** — local model handles 90%+, Mistral Large for the rest

---

## 11. Remote Monitoring (W&B from Laptop)

All training runs on the RTX 5090 are tracked via **Weights & Biases** and can be
monitored in real-time from any device (MacBook on the train, phone, etc.).

### Setup
```bash
# On the 5090 training machine:
export WANDB_API_KEY="your-key-here"
export WANDB_PROJECT="reachy-copilot"
export WANDB_ENTITY="tinytimor"  # your W&B username

# Training scripts auto-log to W&B:
python scripts/02_sft_qlora.py --data data/training_data.jsonl --wandb-run-name "sft-run-1"
python scripts/03_grpo_agent.py --model models/sft --wandb-run-name "grpo-run-1"
```

### Monitoring from Your Laptop
```bash
# Open in any browser — no VPN needed:
open https://wandb.ai/tinytimor/reachy-copilot

# Or use the W&B CLI:
pip install wandb
wandb login
wandb sync  # sync offline runs if needed
```

### What Gets Tracked
| Metric | Script | Description |
|--------|--------|-------------|
| Training loss | `02_sft_qlora.py` | SFT loss curve per epoch |
| Learning rate | `02_sft_qlora.py` | LR schedule visualization |
| Gradient norms | `02_sft_qlora.py` | Gradient health monitoring |
| Reward scores | `03_grpo_agent.py` | GRPO reward by category |
| GPU utilization | Both | VRAM, temp, utilization % |
| Hyperparameters | Both | LoRA rank, LR, batch size, etc. |

### Pipeline Completion Alerts
The `run_experiments.sh` script sends **W&B alerts** when the pipeline completes
or crashes — visible as push notifications on your phone via the W&B mobile app:
```bash
# Leave running unattended on 5090:
nohup ./run_experiments.sh > pipeline.log 2>&1 &
disown

# You'll get a W&B alert when it finishes ✅ or crashes ❌
# Check from anywhere: https://wandb.ai/tinytimor/reachy-copilot
```

---

## 12. Prize Strategy

| Prize | Agent Feature | Demo Moment |
|-------|---------------|-------------|
| **Local 1st-3rd** | Full multi-agent architecture, edge + cloud | Robot responds physically to voice |
| **Best Voice (ElevenLabs)** | Reactive Agent → ElevenLabs TTS → Robot speaker | Natural voice conversation |
| **Best Use of Mistral Vibe** | Used Vibe to build the project + Vibe skill | Show Vibe in action |
| **Best Architectural Modification** | Split-brain: edge model for reactions, cloud for reasoning | Show latency difference |

---

## Quick Reference

| Component | Port | Host | Protocol |
|-----------|------|------|----------|
| OpenClaw Gateway | 18789 | Orin Nano (10.0.0.232) | HTTP/WS |
| Ollama API | 11434 | Orin Nano (10.0.0.232) | HTTP |
| Reachy gRPC | 50051 | Reachy Mini | gRPC |
| VNC Server | 5901 | Orin Nano | VNC |
| W&B Dashboard | — | wandb.ai/thalamus_ai/reachy-copilot | HTTPS |
| Mistral API | — | api.mistral.ai | HTTPS |

### Key Files for Orin Deployment

| File | Purpose |
|------|---------|
| `models/reachy-copilot-gguf/model-q4_k_m.gguf` | Quantized model (2.0 GB) — SCP to Orin |
| `models/reachy-copilot-gguf/Modelfile` | Ollama config — SCP to Orin |
| `~/.openclaw/openclaw.json` | OpenClaw Gateway config — create on Orin |
| `clawd-reachy-mini/` | Voice + robot interface — clone on Orin |
## References & Credits

This project builds on the work of many open-source contributors and projects.
We gratefully acknowledge:

### Core Dependencies
| Project | Author(s) | License | How We Use It |
|---------|-----------|---------|---------------|
| [OpenClaw](https://github.com/openclaw/openclaw) | OpenClaw Team | MIT | Gateway orchestration, skill system, multi-channel messaging |
| [clawd-reachy-mini](https://github.com/ArturSkowronski/clawd-reachy-mini) | Artur Skowronski | MIT | Reachy Mini SDK integration patterns, ElevenLabs TTS, wake word detection |
| [VisionClaw](https://github.com/1rgs/VisionClaw) | 1rgs | MIT | OpenAI-compatible endpoint pattern, single `execute` tool design |
| [Reachy Mini SDK](https://github.com/pollen-robotics/reachy2-sdk) | Pollen Robotics | Apache 2.0 | Robot control (head, antennas, camera, microphone) |

### Models
| Model | Provider | License | Role |
|-------|----------|---------|------|
| [Mistral Large 3](https://mistral.ai/) | Mistral AI | Mistral Research License | Teacher model for data generation |
| [Ministral 3 3B Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Mistral AI | Apache 2.0 | Student model, fine-tuned for edge |
| [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Mistral AI | Apache 2.0 | Edge audio model (ASR + understanding) |
| [Ministral 3 3B GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF) | Mistral AI | Apache 2.0 | Pre-quantized for Ollama deployment |
| [Voxtral Mini 3B GGUF](https://huggingface.co/mradermacher/Voxtral-Mini-3B-2507-GGUF) | mradermacher (quantized) | Apache 2.0 | Pre-quantized for edge audio |

### Libraries & Frameworks
| Library | Use |
|---------|-----|
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Model loading, tokenization |
| [TRL](https://github.com/huggingface/trl) | SFT + GRPO training |
| [PEFT](https://github.com/huggingface/peft) | QLoRA parameter-efficient fine-tuning |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking and remote monitoring |
| [Ollama](https://ollama.ai/) | Edge model serving on Orin Nano |
| [FastAPI](https://fastapi.tiangolo.com/) | Bridge server HTTP API |
| [ElevenLabs](https://elevenlabs.io/) | Text-to-speech voice synthesis |
| [Piper TTS](https://github.com/rhasspy/piper) | Offline text-to-speech fallback |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF quantization engine |

### Hardware
| Device | Manufacturer | Role |
|--------|--------------|------|
| RTX 5090 (32GB) | NVIDIA | Training + cloud inference |
| Jetson Orin Nano Super (8GB) | NVIDIA | Edge deployment |
| Reachy Mini | Pollen Robotics | Embodied robot platform |

### Research & Inspiration
- DeepSeek-R1 — GRPO reinforcement learning approach for reasoning
- Mistral Agent Skills standard — tool-calling format
- Split-brain architecture pattern — edge reactions + cloud reasoning

---

## 13. Future Work — Fully On-Device Multimodal

> **The goal:** Replace ALL Mistral API calls with on-device models. Zero cloud dependency.

### Current State (Hackathon Demo)
| Modality | Model | Runs Where | Cloud? |
|----------|-------|------------|--------|
| Text + Tools | Ministral 3B Q4_K_M (fine-tuned) | Orin Nano (Ollama) | ❌ Local |
| Vision | Pixtral (via mistral-small) | Mistral API | ✅ Cloud |
| ASR | Voxtral Mini | Mistral API | ✅ Cloud |
| TTS | edge-tts | Orin Nano | ❌ Local |

### What's Now Available on Ollama
| Model | Size | Capabilities | Orin 8GB? |
|-------|------|-------------|-----------|
| `ministral-3:3b` | **3.0 GB** | Text + **Vision** + Tools, 256K ctx | ✅ Swap mode |
| `ministral-3:8b` | 6.0 GB | Text + **Vision** + Tools, 256K ctx | ⚠️ Tight |
| Voxtral Mini 3B GGUF | ~2.5 GB | Audio ASR + understanding | ✅ Swap mode |

### Near-Term Target
```
Text + Vision → ministral-3:3b on Ollama (3.0 GB, LOCAL)
ASR           → Voxtral Mini 3B GGUF (2.5 GB, LOCAL, swap mode)
TTS           → edge-tts or Piper (LOCAL)
Tools         → OpenClaw skills (LOCAL)

Result: ZERO cloud API calls. Fully offline embodied AI on a $249 board.
```

### Why Mistral + NVIDIA
- **Mistral** builds compact, capable models (3B with vision + tools + 256K context)
- **NVIDIA** builds the edge hardware that runs them (Orin Nano: 67 TOPS, 8GB, $249)
- Together: true embodied AI that works anywhere — no cloud, no latency, no cost per query
