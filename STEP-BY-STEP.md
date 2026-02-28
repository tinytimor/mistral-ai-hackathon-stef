# 🚀 STEP-BY-STEP: Reachy Copilot — Hackathon Build Guide

> From `az login` to a talking robot in ~4 hours.
> Every step is explicit. No hand-waving.

---

## Prerequisites Checklist

- [ ] RTX 5090 machine with CUDA + Python 3.10+
- [ ] Orin Nano Super flashed with JetPack 6
- [ ] Reachy Mini powered on and on the network
- [ ] All three devices on the same Ethernet network
- [ ] Azure account with Mistral Large deployed (or any chat model)

---

## Phase 0: Environment Setup (15 min)

### Step 0.1 — Clone This Repo

```bash
git clone <this-repo-url> reachy-copilot
cd reachy-copilot
```

### Step 0.2 — Azure Login (on 5090 machine)

```bash
# Login to Azure
az login

# Verify your subscription
az account show --query "{name:name, id:id}"

# Set your Azure resource name
export AZURE_RESOURCE_NAME="your-azure-ai-resource"
export AZURE_MODEL_NAME="mistral-large-latest"  # your deployment name
```

### Step 0.3 — Install Python Dependencies (on 5090)

```bash
python3 -m venv .venv
source .venv/bin/activate

# Core training dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers trl peft bitsandbytes datasets accelerate
pip install openai azure-identity
pip install ddgs fastapi uvicorn httpx

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 0.4 — Set Up Orin Nano

```bash
# SSH into the Orin
ssh orin@<ORIN_IP>

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Create working directories
mkdir -p ~/reachy-model ~/reachy-bridge

# Install Python deps
python3 -m venv ~/reachy-env
source ~/reachy-env/bin/activate
pip install reachy2-sdk fastapi uvicorn httpx ddgs

# Test Reachy connection
python3 -c "
from reachy2_sdk import ReachySDK
reachy = ReachySDK(host='<REACHY_IP>')
print('Connected:', reachy.is_connected())
reachy.disconnect()
"
```

---

## Phase 1: Generate Training Data (30 min)

### Step 1.1 — Generate Synthetic Data with Mistral Large

On the 5090 machine:

```bash
source .venv/bin/activate

# Generate 200 samples (fast, ~15 min with Mistral Large)
python scripts/01_generate_training_data.py \
  --output data/training_data.jsonl \
  --num-samples 200 \
  --direct-only

# Or include HuggingFace datasets too (adds 4000 more samples):
python scripts/01_generate_training_data.py \
  --output data/training_data.jsonl \
  --num-samples 200 \
  --include-hf
```

### Step 1.2 — Verify the Data

```bash
# Count samples
wc -l data/training_data.jsonl

# Preview first sample
head -1 data/training_data.jsonl | python3 -m json.tool | head -30
```

**Expected output:** 200-4200 JSONL lines with tool-calling conversations.

---

## Phase 2: SFT with QLoRA (45 min)

### Step 2.1 — Run Supervised Fine-Tuning

```bash
python scripts/02_sft_qlora.py \
  --data data/training_data.jsonl \
  --output models/ministral-sft \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --max-seq-len 2048 \
  --lora-r 32
```

**What this does:**
- Loads Ministral-8B-Instruct with 4-bit QLoRA (~6GB VRAM)
- Trains LoRA adapters (r=32) on attention + MLP projections
- Uses packing for efficient sequence utilization
- Takes ~30-40 min on RTX 5090 with 4000 samples

### Step 2.2 — Verify SFT Output

```bash
ls models/ministral-sft/
# Should see: adapter_config.json, adapter_model.safetensors, training_info.json, tokenizer files
```

---

## Phase 3: GRPO Agent Training (30 min, OPTIONAL)

> ⚡ **Hackathon shortcut:** Skip this if time is tight. SFT alone gives you
> tool-calling ability. GRPO refines the quality but isn't strictly necessary.

### Step 3.1 — Run GRPO

```bash
python scripts/03_grpo_agent.py \
  --model models/ministral-sft \
  --output models/ministral-grpo \
  --epochs 1 \
  --batch-size 2 \
  --num-generations 4 \
  --lr 5e-6
```

**What this does:**
- Generates 4 completions per prompt
- Scores them with 4 reward functions (format, relevance, quality, thinking)
- Updates policy using GRPO (no separate critic model needed)
- Takes ~20-30 min on RTX 5090

---

## Phase 4: Quantize & Deploy (20 min)

### Step 4.1 — Get llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make -j$(nproc)
cd ..
```

### Step 4.2 — Merge + Convert + Quantize

```bash
# Use whichever model is your best (SFT or GRPO)
python scripts/04_quantize_deploy.py \
  --model models/ministral-sft \
  --output models/ministral-gguf \
  --llama-cpp ./llama.cpp \
  --quant Q4_K_M
```

### Step 4.3 — Deploy to Orin Nano

```bash
# Copy the quantized model to Orin
scp models/ministral-gguf/model-q4_k_m.gguf orin@<ORIN_IP>:~/reachy-model/
scp models/ministral-gguf/Modelfile orin@<ORIN_IP>:~/reachy-model/

# SSH to Orin and create the Ollama model
ssh orin@<ORIN_IP>
cd ~/reachy-model
ollama create reachy-copilot -f Modelfile

# Test it
ollama run reachy-copilot "Hello! Look at me and tell me a joke."
```

---

## Phase 5: Bridge Server + Robot (15 min)

### Step 5.1 — Copy Bridge Server to Orin

```bash
# From the 5090 machine
scp docs/ORIN-REACHY-SETUP.md orin@<ORIN_IP>:~/

# The bridge server code is in the ORIN-REACHY-SETUP.md
# Extract the server.py code, or create it directly:
```

### Step 5.2 — Create Bridge Server on Orin

SSH to the Orin and create `~/reachy-bridge/server.py` with the contents from
[docs/ORIN-REACHY-SETUP.md](docs/ORIN-REACHY-SETUP.md#bridge-server).

**Critical:** Update these two lines in the server:

```python
REACHY_IP = "192.168.1.XX"  # ← Your Reachy Mini's actual IP
MODEL_NAME = "reachy-copilot"  # ← Must match what you created in Ollama
```

### Step 5.3 — Start the Bridge

```bash
ssh orin@<ORIN_IP>
source ~/reachy-env/bin/activate
cd ~/reachy-bridge
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Step 5.4 — Test Everything End-to-End

```bash
# From any machine on the network:

# Health check
curl http://<ORIN_IP>:8000/health

# Make the robot nod
curl -X POST http://<ORIN_IP>:8000/robot/nod

# Full chat (LLM thinks → calls tools → robot moves)
curl -X POST http://<ORIN_IP>:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! Can you look at me, nod, and tell me about yourself?"}'
```

**Expected behavior:**
1. Robot looks at you (look_at tool)
2. Robot nods (express tool)
3. Robot responds with text (shown in terminal)

---

## Phase 6: Polish & Demo (remaining time)

### Demo Script

```bash
# Rehearse these interactions for the judges:

# 1. "Hey Reachy, what can you do?"
# 2. "Search for the latest news about AI in healthcare"
# 3. "I feel dizzy. Can you help?"
# 4. "Look at my colleague and say hello"
# 5. "Set a reminder in 5 minutes for my medication"
```

### Quick Improvements

- **Add ElevenLabs TTS:** Get an API key at elevenlabs.io, integrate into the `robot_speak` tool
- **Add camera vision:** Use Reachy's camera + Mistral Vision API for "what do you see?"
- **Add memory:** Store conversation history in a JSON file for context
- **Better expressions:** Fine-tune antenna movements for more personality

---

## File Reference

| File | Purpose | Runs On |
|------|---------|---------|
| [scripts/01_generate_training_data.py](scripts/01_generate_training_data.py) | Generate tool-calling data via Azure Mistral Large | 5090 |
| [scripts/02_sft_qlora.py](scripts/02_sft_qlora.py) | SFT with QLoRA on Ministral | 5090 |
| [scripts/03_grpo_agent.py](scripts/03_grpo_agent.py) | GRPO RL for think-plan-act-reflect | 5090 |
| [scripts/04_quantize_deploy.py](scripts/04_quantize_deploy.py) | Merge LoRA → GGUF → Q4_K_M | 5090 |
| [docs/ORIN-REACHY-SETUP.md](docs/ORIN-REACHY-SETUP.md) | Hardware connection guide + bridge server | Orin + Reachy |
| [BATTLE-PLAN.md](BATTLE-PLAN.md) | Strategic hackathon plan | Reference |
| [notes/deep-research-synthesis.md](notes/deep-research-synthesis.md) | Technical deep-dive notes | Reference |

## Datasets Used

| Dataset | Size | License | Purpose |
|---------|------|---------|---------|
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 11.5K | Apache-2.0 | Tool calling + agentic JSON |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113K | Apache-2.0 | Function calling conversations |
| Custom (Azure-generated) | ~200-500 | Yours | Robot-specific tool calling |

## Key Commands Cheat Sheet

```bash
# Azure login
az login

# Generate data
python scripts/01_generate_training_data.py --output data/training_data.jsonl --num-samples 200

# Train (SFT)
python scripts/02_sft_qlora.py --data data/training_data.jsonl --output models/ministral-sft

# Train (GRPO, optional)
python scripts/03_grpo_agent.py --model models/ministral-sft --output models/ministral-grpo

# Quantize
python scripts/04_quantize_deploy.py --model models/ministral-sft --output models/ministral-gguf

# Deploy to Orin
scp models/ministral-gguf/* orin@<ORIN_IP>:~/reachy-model/
ssh orin@<ORIN_IP> "cd ~/reachy-model && ollama create reachy-copilot -f Modelfile"

# Start bridge
ssh orin@<ORIN_IP> "source ~/reachy-env/bin/activate && cd ~/reachy-bridge && uvicorn server:app --host 0.0.0.0 --port 8000"

# Test
curl -X POST http://<ORIN_IP>:8000/chat -H "Content-Type: application/json" -d '{"message": "Hello!"}'
```

---

**Good luck at the hackathon! 🏆**
