# 🤖 AGENTS.md — Reachy Copilot Agent Architecture

> Multi-agent system combining edge inference, cloud intelligence, and physical robotics
> for the Mistral Worldwide Hackathon 2026.

---

## Agent Overview

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

---

## 1. Reasoning Agent (Cloud / RTX 5090)

**Model:** Mistral Large 3 (675B MoE, 41B active)
**Location:** Azure AI Foundry API or local on RTX 5090
**Role:** Complex reasoning, multi-step planning, tool orchestration

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
agent:
  name: reasoning-agent
  model: Mistral-Large-3
  provider: foundry  # or mistral API
  temperature: 0.7
  max_tokens: 2048
  tools:
    - search_web
    - send_email
    - calendar_list_events
    - calendar_create_event
    - get_patient_summary
    - memory_search
    - browser_action
  system_prompt: |
    You are the reasoning core of Reachy, an embodied AI assistant.
    Think carefully, plan your approach, execute tools, then reflect.
    Wrap your thinking in <think></think> tags.
    Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> for actions.
```

---

## 2. Reactive Agent (Orin Nano Edge)

**Model:** Ministral 3B Q4_K_M (fine-tuned via SFT + GRPO)
**Location:** Jetson Orin Nano Super (8GB, 67 TOPS)
**Role:** Real-time robot reactions, fast local responses, physical interaction

### Capabilities
- Sub-second response to user presence (head tracking, nodding, expressions)
- Quick verbal responses via Piper TTS / ElevenLabs
- Real-time speech transcription via Whisper.cpp
- Local tool calling for robot control
- Camera-based perception (object detection, face tracking)

### When Invoked
- User speaks directly to the robot (wake word detection)
- Real-time emotional reactions (user enters room, waves, etc.)
- Simple queries that don't need cloud reasoning
- Latency-critical robot movements

### Configuration
```yaml
agent:
  name: reactive-agent
  model: reachy-copilot  # Ollama model name
  provider: ollama
  endpoint: http://localhost:11434
  temperature: 0.6
  max_tokens: 512
  context_length: 2048
  tools:
    - robot_look_at
    - robot_express
    - robot_speak
    - set_reminder
  system_prompt: |
    You are Reachy, a friendly robot assistant.
    Respond quickly and naturally. Use your body to express emotions.
    For complex tasks, say "Let me think about that more carefully."
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
Text mode (default — Ministral only):
  Ministral 3B Q4_K_M     : ~2.0 GB
  KV Cache (2048 ctx)      : ~0.5 GB
  CUDA runtime             : ~0.8 GB
  OS + Python + Reachy SDK : ~1.5 GB
  Bridge server + FastAPI  : ~0.3 GB
  Piper TTS                : ~0.2 GB
  ─────────────────────────────────
  Total                    : ~5.3 GB  ✅ (2.7 GB headroom)

Audio mode (swap in Voxtral for voice tasks):
  Voxtral Mini 3B Q4_K_M   : ~2.5 GB
  KV Cache (2048 ctx)       : ~0.5 GB
  CUDA runtime              : ~0.8 GB
  OS + Python + Reachy SDK  : ~1.5 GB
  Bridge server + FastAPI   : ~0.3 GB
  ─────────────────────────────────
  Total                     : ~5.6 GB  ✅ (2.4 GB headroom)

Dual Q3 mode (both loaded simultaneously):
  Ministral 3B Q3_K         : ~1.8 GB
  Voxtral Mini 3B Q3_K      : ~1.8 GB
  KV Cache (2048 ctx × 2)   : ~1.0 GB
  CUDA runtime              : ~0.8 GB
  OS + Python + Reachy SDK  : ~1.5 GB
  ─────────────────────────────────
  Total                     : ~6.9 GB  ✅ (1.1 GB headroom)
```

---

## 3. OpenClaw Gateway Agent (Orchestrator)

**Model:** Model-agnostic (routes to Reasoning or Reactive agent)
**Location:** RTX 5090 Desktop (or Orin Nano in Option B)
**Role:** Session management, memory persistence, multi-channel routing

### Capabilities
- Persistent memory across conversations (24/7)
- Multi-channel inbox (WhatsApp, Telegram, Discord, Slack, iMessage)
- Skill orchestration and tool routing
- Browser control (CDP) for web interactions
- Cron jobs, webhooks, proactive check-ins ("heartbeats")
- Agent-to-agent sessions

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
```python
def route_request(user_input: str, context: dict) -> str:
    """Route to the appropriate agent based on complexity and latency needs."""

    # Latency-critical: always local
    if is_robot_action(user_input):  # "look at me", "nod", "wave"
        return "reactive-agent"  # Orin Nano, <200ms

    # Simple queries: local model
    if is_simple_query(user_input):  # greetings, time, weather
        return "reactive-agent"  # Orin Nano, <1s

    # Complex reasoning: cloud model
    if needs_reasoning(user_input):  # multi-step, healthcare, analysis
        return "reasoning-agent"  # Mistral Large, 2-5s

    # Default: try local first, fallback to cloud
    return "reactive-agent-with-fallback"
```

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
    ↓ GRPO with reward functions (scripts/03_grpo_agent.py)
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

### Experiment Tracking
All training tracked via **Weights & Biases**:
- Project: `reachy-copilot`
- Dashboard: https://wandb.ai/tinytimor/reachy-copilot
- Metrics: loss, learning rate, gradient norms, reward scores, GPU utilization

---

## 8. Deployment Topology

### Hackathon Demo Setup
```
┌────────────────────────────────────────────────────────────┐
│                    NETWORK (LAN / Tailscale)                │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  RTX 5090    │  │  Orin Nano   │  │  MacBook Pro   │   │
│  │  Desktop     │  │  + Reachy    │  │  (Control)     │   │
│  │              │  │              │  │                │   │
│  │  OpenClaw    │  │  Ollama      │  │  VNC Viewer    │   │
│  │  Gateway     │  │  3B model    │  │  SSH Terminal  │   │
│  │  :3000       │  │  Bridge API  │  │  W&B Dashboard │   │
│  │              │  │  :8000       │  │                │   │
│  │  W&B logging │  │  Whisper STT │  │  Monitor from  │   │
│  │              │  │  Piper TTS   │  │  train (NYC)   │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
│                                                            │
│  ── Azure AI Foundry (Cloud) ──────────────────────────── │
│  │  Mistral-Large-3 (fallback reasoning)                 │ │
│  └───────────────────────────────────────────────────────  │
└────────────────────────────────────────────────────────────┘
```

### Remote Access (Mac → Orin Nano from train)
- **VNC Server** on Orin Nano for GUI access
- **SSH tunnel** for terminal + port forwarding
- **Tailscale** for secure remote access over cellular
- **W&B Dashboard** for monitoring training from anywhere

---

## 9. clawd-reachy-mini Integration

The [clawd-reachy-mini](https://github.com/ArturSkowronski/clawd-reachy-mini) project provides
an existing OpenClaw ↔ Reachy Mini bridge:

### What It Provides
- WebSocket client for OpenClaw Gateway protocol
- STT backends (Whisper, Faster-Whisper, OpenAI)
- TTS via ElevenLabs
- Reachy Mini SDK integration (head, antennas, emotions)
- Wake word detection
- Conversation loop (mic → STT → Gateway → TTS → robot)

### How We Extend It
1. **Add fine-tuned model support** — Route to our custom Ministral 3B instead of default
2. **Add memory layer** — Inject conversation context from L2/L3 memory
3. **Add healthcare skills** — Patient summaries, medication reminders
4. **Add vision pipeline** — Camera → Pixtral for visual understanding
5. **Custom SKILL.md** — Our 18 tools registered as OpenClaw skills

### Installation
```bash
git clone https://github.com/ArturSkowronski/clawd-reachy-mini.git
cd clawd-reachy-mini
uv sync --extra dev --extra audio
uv run clawd-reachy --gateway-host <5090_IP>
```

---

## 10. Edge Bridge Architecture (VisionClaw Pattern)

The bridge server (`scripts/06_openclaw_bridge.py`) runs on Orin Nano and is the
central nervous system of the Reactive Agent. It exposes:

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main chat endpoint with smart routing |
| `/v1/chat/completions` | POST | OpenAI-compatible (VisionClaw pattern) |
| `/robot/{action}` | POST | Direct robot control |
| `/memory/store` | POST | Store to L2/L3 memory |
| `/memory/search` | POST | Search persistent memory |
| `/health` | GET | System status check |

### Key Design Decisions
- **Inference is 5090-independent** — the 5090 trains models, but Orin runs them
- **Local tool execution** — all robot + memory tools run on-device
- **OpenAI-compatible API** — VisionClaw pattern for seamless gateway integration
- **Smart routing** — simple queries stay local, complex queries escalate to cloud
- **Memory-augmented prompts** — L2/L3 memory injected into every LLM call

### What We Borrow vs. Build
| From **clawd-reachy-mini** | From **VisionClaw** | **Our additions** |
|---|---|---|
| Reachy SDK patterns | Single `execute` tool | Local Ollama inference |
| ElevenLabs TTS | OpenAI-compat endpoint | Memory-augmented prompts |
| Wake word detection | Session key management | Smart complexity routing |
| Emotion animations | Multi-turn tool calling | Voxtral model swapping |

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
| OpenClaw Gateway | 3000 | 5090 Desktop | HTTP/WS |
| Ollama API | 11434 | Orin Nano | HTTP |
| Reachy Bridge | 8000 | Orin Nano | HTTP/REST |
| Reachy gRPC | 50051 | Reachy Mini | gRPC |
| VNC Server | 5901 | Orin Nano | VNC |
| W&B Dashboard | — | wandb.ai | HTTPS |
| Azure Foundry | — | *.openai.azure.com | HTTPS |

---

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
