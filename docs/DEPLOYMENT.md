# 🤖 Deploying to Orin Nano

> Full step-by-step guide for deploying the fine-tuned Reachy Copilot model
> to the Jetson Orin Nano Super (8GB) via Ollama + OpenClaw Gateway + clawd-reachy-mini.

Back to [README](../README.md) · Hardware setup: [ORIN-REACHY-SETUP.md](ORIN-REACHY-SETUP.md)

---

## What Gets Deployed

| Component | Description |
|-----------|-------------|
| `model-q4_k_m.gguf` (~2.0 GB) | Quantized fine-tuned Ministral 3B (via Ollama) |
| `Modelfile` (~1 KB) | Ollama config: system prompt + Mistral v7 chat template |
| OpenClaw Gateway | Node.js daemon on port 18789, skill orchestration |
| clawd-reachy-mini | Python voice/robot interface, Whisper + ElevenLabs |

## Memory Budget (Orin Nano 8GB)

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

---

## Step-by-Step Setup

> **Before you start:** Close Chrome and VS Code on the Orin. The Orin uses unified memory
> (CPU + GPU share the same pool). Chrome alone consumes ~1.5 GB, leaving no room for the
> 2 GB model + CUDA runtime. Kill them first: `pkill -f chromium; pkill -f code`

```bash
# 1. Install Ollama on the Orin
curl -fsSL https://ollama.com/install.sh | sh

# 2. Copy model files from the 5090 (or they may already be on the Orin)
mkdir -p ~/reachy-model
# (scp from 5090 - see ORIN-REACHY-SETUP.md)

# 3. Create and test the model
# IMPORTANT: must cd into the model directory - Modelfile uses a relative path
cd ~/reachy-model   # (or wherever model-q4_k_m.gguf lives)
ollama create reachy-copilot -f Modelfile
ollama run reachy-copilot "Hello!"

# 4. Install Node.js >= 22 + OpenClaw Gateway
# Use npm directly - the install.sh script can hang/crash on the Orin
node --version  # must be >= 22 (already installed via nvm on this machine)
npm i -g openclaw

# 5. Run onboarding wizard (answer: Mistral / your API key / mistral-large-latest /
#    No to skills / No to channels / Hatch in TUI)
openclaw onboard --install-daemon
# When TUI opens, press q to exit - daemon keeps running

# 6. Write the openclaw.json config
# NOTE: "memory" and "bind" are NOT valid keys - omit them
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
# uv is NOT installed by default - install it first
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # reload PATH

cd ~
git clone https://github.com/ArturSkowronski/clawd-reachy-mini.git
cd clawd-reachy-mini
# uv sync creates .venv automatically - do NOT use conda or pip
# Warning "'reachy-mini' does not have extra 'vision'" is harmless
# Downloads ~300 MB (torch, scipy, opencv, etc.) - allow 5-10 min
uv sync --extra dev --extra audio
uv run clawd-reachy --gateway-host localhost --gateway-port 18789

# 9. Test the full pipeline (OpenClaw HTTP API)
curl -X POST http://127.0.0.1:18789/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer reachy-hackathon-2026" \
  -d '{"model": "ollama/reachy-copilot", "messages": [{"role": "user", "content": "Look at me and say hello!"}]}'
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `cudaMalloc failed: out of memory` | Close Chrome (`pkill -f chromium`) and VS Code, retry |
| `Gateway start blocked: set gateway.mode=local` | Add `"mode": "local"` to the `gateway` block in `openclaw.json` |
| `Unrecognized key: "memory"` | Remove the `memory` block - it's built-in, not configurable |
| `Invalid input` for `bind` | Remove the `bind` key entirely - let OpenClaw use its default |
| Gateway timed out / port not bound | Run `openclaw doctor --fix` then `systemctl --user restart openclaw-gateway.service` |
| `uv: command not found` | Run `curl -LsSf https://astral.sh/uv/install.sh \| sh && source ~/.bashrc` |

---

## Verified Tool Calling

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
