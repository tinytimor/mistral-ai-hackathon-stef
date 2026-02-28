# 🔌 Orin Nano ↔ Reachy Mini — Hardware Connection Guide

> Complete step-by-step directions for connecting your NVIDIA Jetson Orin Nano Super
> to the Pollen Robotics Reachy Mini robot.

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [Physical Connections](#physical-connections)
3. [Orin Nano Setup](#orin-nano-setup)
4. [Reachy Mini SDK Installation](#reachy-mini-sdk-installation)
5. [Test the Connection](#test-the-connection)
6. [Ollama LLM Setup](#ollama-llm-setup)
7. [Bridge Server (LLM → Robot)](#bridge-server)
8. [OpenClaw Integration](#openclaw-integration)
9. [Troubleshooting](#troubleshooting)

---

## 1. Hardware Overview

| Component | Specs | Role |
|-----------|-------|------|
| **Orin Nano Super** | 8GB LPDDR5, 67 TOPS, JetPack 6 | Edge inference (runs Ministral Q4_K_M) |
| **Reachy Mini** | 6-DOF head, 2 antennas, camera, 4 mics, speaker | Physical embodiment |
| **RTX 5090** | 32GB GDDR7 | Training + gateway (OpenClaw) |
| **Network Switch/Router** | Gigabit | Connects all devices |

### Architecture

```
┌─────────────┐    Ethernet    ┌─────────────┐    gRPC     ┌─────────────┐
│  RTX 5090   │◄──────────────►│  Orin Nano   │◄──────────►│ Reachy Mini │
│  (Gateway)  │  192.168.1.x   │  (Inference) │ 192.168.x  │   (Robot)   │
│  OpenClaw   │                │  Ollama +    │            │   Head +    │
│  Port 3000  │                │  Bridge API  │            │   Cameras   │
└─────────────┘                └─────────────┘            └─────────────┘
```

---

## 2. Physical Connections

### Step 1: Connect Reachy Mini to Power

```
1. Plug in the Reachy Mini power adapter
2. Wait for the LED to turn solid (boot takes ~30 seconds)
3. The robot runs its own Linux computer internally with a gRPC server
```

### Step 2: Network the Reachy Mini

```
1. Connect an Ethernet cable from Reachy Mini to your router/switch
2. The robot will get an IP via DHCP
3. Find its IP address:

   Option A — mDNS (if supported):
   $ ping reachy.local

   Option B — Router admin page:
   Check your router's DHCP clients list

   Option C — nmap scan:
   $ nmap -sn 192.168.1.0/24 | grep -B2 "Pollen"
```

### Step 3: Connect Orin Nano to Network

```
1. Connect Ethernet cable from Orin Nano to the same router/switch
2. Verify the Orin has an IP:
   $ hostname -I
   # Example: 192.168.1.100

3. Verify Orin can reach Reachy:
   $ ping <REACHY_IP>
   # Should get responses
```

### Step 4: Connect RTX 5090 Machine to Network

```
1. Connect your 5090 machine to the same network
2. Verify all three can ping each other
```

> **⚠️ Important:** Do NOT connect Reachy directly to the Orin via Ethernet.
> Both must be on the same network via a router or switch.

---

## 3. Orin Nano Setup

### Verify JetPack is Installed

```bash
# Check JetPack version
cat /etc/nv_tegra_release
# Should show L4T r36.x for JetPack 6

# Check CUDA
nvcc --version
# Should show CUDA 12.x

# Check available memory
free -h
# Should show ~8GB total
```

### Install Python Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python essentials
sudo apt install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/reachy-env
source ~/reachy-env/bin/activate

# Install core packages
pip install --upgrade pip
pip install fastapi uvicorn httpx websockets pydantic
```

---

## 4. Reachy Mini SDK Installation

The Reachy Mini uses the **reachy2-sdk** (same SDK as Reachy 2 — the Mini
is essentially a head-only Reachy 2).

```bash
# Activate your environment
source ~/reachy-env/bin/activate

# Install the SDK
pip install reachy2-sdk

# This installs:
#   - reachy2_sdk (Python SDK)
#   - reachy2_sdk_api (gRPC protobuf definitions)
#   - grpcio (gRPC runtime)
#   - pyquaternion (for head orientation)
```

### Verify SDK Installation

```python
python3 -c "from reachy2_sdk import ReachySDK; print('SDK imported successfully')"
```

---

## 5. Test the Connection

### Step 1: Basic Connection

```python
#!/usr/bin/env python3
"""test_reachy_connection.py — Run this on the Orin Nano"""

from reachy2_sdk import ReachySDK
import time

REACHY_IP = "192.168.1.XX"  # ← Replace with your Reachy's IP

print(f"Connecting to Reachy at {REACHY_IP}...")
reachy = ReachySDK(host=REACHY_IP)

if reachy.is_connected():
    print("✅ Connected to Reachy!")
    print(f"   Info: {reachy.info}")
else:
    print("❌ Failed to connect!")
    exit(1)

# Turn on the robot
print("Turning on...")
reachy.turn_on()
time.sleep(1)

# Test head movement — look straight ahead
print("Looking straight ahead...")
reachy.head.look_at(x=0.5, y=0, z=0.2, duration=1.0, wait=True)
time.sleep(0.5)

# Nod
print("Nodding...")
reachy.head.goto([0, -20, 0], duration=0.5, wait=True)
reachy.head.goto([0, 10, 0], duration=0.5, wait=True)
reachy.head.goto([0, -10, 0], duration=0.5, wait=True)

# Look left, then right
print("Looking left...")
reachy.head.look_at(x=0.5, y=0.3, z=0.2, duration=1.0, wait=True)
print("Looking right...")
reachy.head.look_at(x=0.5, y=-0.3, z=0.2, duration=1.0, wait=True)

# Return to default
print("Returning to default posture...")
reachy.head.goto_posture("default", duration=1.0, wait=True)

# Antenna wave (if available)
if reachy.head.l_antenna:
    print("Waving antennas...")
    reachy.head.l_antenna.goto(30, duration=0.5)
    reachy.head.r_antenna.goto(-30, duration=0.5, wait=True)
    time.sleep(0.5)
    reachy.head.l_antenna.goto(0, duration=0.5)
    reachy.head.r_antenna.goto(0, duration=0.5, wait=True)

print("\n✅ All tests passed! Reachy is ready.")

# Clean up
reachy.turn_off()
reachy.disconnect()
```

Save this as `test_reachy_connection.py` on the Orin and run:

```bash
source ~/reachy-env/bin/activate
python3 test_reachy_connection.py
```

### Step 2: Test Audio (Microphone + Speaker)

```python
"""test_reachy_audio.py — Test Reachy's microphone and speaker"""

from reachy2_sdk import ReachySDK
import time

reachy = ReachySDK(host="192.168.1.XX")  # Replace with your IP

if reachy.audio:
    print("🎤 Audio system available!")

    # Record 3 seconds of audio
    print("Recording 3 seconds...")
    reachy.audio.start_recording()
    time.sleep(3)
    audio_data = reachy.audio.stop_recording()
    print(f"   Recorded {len(audio_data)} samples")

    # Play back (if speaker available)
    # Note: TTS would be handled separately (e.g., via ElevenLabs or piper)
else:
    print("⚠️ Audio not available on this Reachy configuration")

reachy.disconnect()
```

### Step 3: Test Camera

```python
"""test_reachy_camera.py — Test Reachy's camera"""

from reachy2_sdk import ReachySDK
import time

reachy = ReachySDK(host="192.168.1.XX")  # Replace with your IP

if reachy.cameras:
    print("📷 Camera system available!")
    # Access camera feed
    # The camera API depends on your Reachy configuration
    print(f"   Cameras: {reachy.cameras}")
else:
    print("⚠️ Camera not available")

reachy.disconnect()
```

---

## 6. Ollama LLM Setup

Install and configure Ollama on the Orin Nano for local inference.

### Install Ollama

```bash
# Native install (recommended — supports CUDA on Jetson)
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# The installer creates a systemd service, so Ollama starts automatically
# Check status:
sudo systemctl status ollama
```

### Load a Base Model (Quick Start)

```bash
# While your fine-tuned model is training on the 5090,
# test with a small model first:
ollama pull mistral:7b-instruct-q4_K_M    # ~4.4GB — tight fit
# OR use a smaller model:
ollama pull phi3:mini-4k-instruct-q4_K_M   # ~2.3GB — comfortable fit
ollama pull qwen2.5:3b-instruct-q4_K_M     # ~2.0GB — most headroom

# Test it
ollama run qwen2.5:3b-instruct-q4_K_M "Hello! What can you help me with?"
```

### Deploy Your Fine-Tuned Model

After running `04_quantize_deploy.py` on the 5090:

```bash
# On the 5090, copy the model to Orin:
scp models/ministral-3b-gguf/model-q4_k_m.gguf orin@<ORIN_IP>:~/reachy-model/
scp models/ministral-3b-gguf/Modelfile orin@<ORIN_IP>:~/reachy-model/

# On the Orin:
cd ~/reachy-model/
ollama create reachy-copilot -f Modelfile
ollama run reachy-copilot "Hello, I'm Reachy!"
```

### Verify Ollama API

```bash
# Ollama serves an OpenAI-compatible API on port 11434
curl http://localhost:11434/api/generate -d '{
  "model": "reachy-copilot",
  "prompt": "What are the symptoms of diabetes?",
  "stream": false
}'
```

---

## 7. Bridge Server (LLM → Robot) {#bridge-server}

This FastAPI server runs on the Orin, connecting the LLM (Ollama) to the robot (Reachy SDK).

Save as `~/reachy-bridge/server.py` on the Orin:

```python
#!/usr/bin/env python3
"""
reachy_bridge_server.py — FastAPI bridge connecting Ollama LLM to Reachy Mini.
Runs on the Orin Nano.

Usage:
    cd ~/reachy-bridge
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from reachy2_sdk import ReachySDK

# ─── Configuration ────────────────────────────────────────────────────────────
REACHY_IP = "192.168.1.XX"       # ← Replace with your Reachy's IP
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "reachy-copilot"     # Or whatever you named it in Ollama

# ─── Global state ─────────────────────────────────────────────────────────────
reachy: Optional[ReachySDK] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to Reachy on startup, disconnect on shutdown."""
    global reachy
    print(f"🤖 Connecting to Reachy at {REACHY_IP}...")
    reachy = ReachySDK(host=REACHY_IP)
    if reachy.is_connected():
        reachy.turn_on()
        reachy.head.goto_posture("default", wait=True)
        print("✅ Reachy connected and ready!")
    else:
        print("⚠️ Reachy not connected — running in LLM-only mode")
        reachy = None
    yield
    if reachy:
        reachy.turn_off()
        reachy.disconnect()
        print("🔌 Reachy disconnected")


app = FastAPI(title="Reachy Bridge", lifespan=lifespan)


# ─── Request/Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tool_calls_executed: list[dict] = []
    thinking: Optional[str] = None


# ─── Tool Execution ──────────────────────────────────────────────────────────

EMOTION_MOVEMENTS = {
    "happy": {"head": [0, -5, 0], "antennas": [20, -20]},
    "sad": {"head": [0, -20, 0], "antennas": [-10, 10]},
    "curious": {"head": [15, -5, 20], "antennas": [30, -5]},
    "surprised": {"head": [0, 5, 0], "antennas": [40, -40]},
    "thinking": {"head": [5, -10, 15], "antennas": [10, -10]},
    "nodding": None,  # Special handling
    "shaking_no": None,  # Special handling
}


async def execute_tool(name: str, args: dict) -> dict:
    """Execute a tool call on the robot."""
    global reachy

    if name == "robot_look_at":
        if reachy:
            x = args.get("x", 0.5)
            y = args.get("y", 0)
            z = args.get("z", 0.2)
            duration = args.get("duration", 1.0)
            reachy.head.look_at(x=x, y=y, z=z, duration=duration, wait=True)
        return {"status": "success", "action": f"Looking at ({args.get('x')}, {args.get('y')}, {args.get('z')})"}

    elif name == "robot_express":
        emotion = args.get("emotion", "happy")
        if reachy:
            if emotion == "nodding":
                reachy.head.goto([0, -20, 0], duration=0.3, wait=True)
                reachy.head.goto([0, 5, 0], duration=0.3, wait=True)
                reachy.head.goto([0, -15, 0], duration=0.3, wait=True)
                reachy.head.goto_posture("default", duration=0.5, wait=True)
            elif emotion == "shaking_no":
                reachy.head.goto([0, -10, 20], duration=0.3, wait=True)
                reachy.head.goto([0, -10, -20], duration=0.3, wait=True)
                reachy.head.goto([0, -10, 15], duration=0.3, wait=True)
                reachy.head.goto_posture("default", duration=0.5, wait=True)
            elif emotion in EMOTION_MOVEMENTS:
                mv = EMOTION_MOVEMENTS[emotion]
                reachy.head.goto(mv["head"], duration=0.5, wait=True)
                if reachy.head.l_antenna and mv.get("antennas"):
                    reachy.head.l_antenna.goto(mv["antennas"][0], duration=0.3)
                    reachy.head.r_antenna.goto(mv["antennas"][1], duration=0.3, wait=True)
                await asyncio.sleep(1.0)
                reachy.head.goto_posture("default", duration=0.5, wait=True)
        return {"status": "success", "emotion": emotion}

    elif name == "robot_speak":
        text = args.get("text", "")
        # TODO: Integrate ElevenLabs or Piper TTS here
        # For now, just log it
        print(f"🔊 SPEAK: {text}")
        return {"status": "success", "text": text, "note": "TTS not yet connected"}

    elif name == "search_web":
        query = args.get("query", "")
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=args.get("max_results", 3))
            return {"results": results}
        except ImportError:
            return {"error": "ddgs not installed. pip install ddgs"}
        except Exception as e:
            return {"error": str(e)}

    elif name == "get_patient_summary":
        # Mock patient data for demo
        return {
            "patient_id": args.get("patient_id", "UNKNOWN"),
            "name": "Demo Patient",
            "conditions": ["Type 2 Diabetes"],
            "medications": ["Metformin 500mg"],
            "vitals": {"bp": "120/80", "hr": 72},
        }

    elif name == "set_reminder":
        return {
            "status": "success",
            "message": args.get("message", ""),
            "minutes": args.get("minutes", 0),
        }

    return {"status": "unknown_tool", "name": name}


def parse_tool_calls(text: str) -> list[dict]:
    """Parse <tool_call> tags from LLM output."""
    calls = []
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            calls.append(parsed)
        except json.JSONDecodeError:
            continue
    return calls


def extract_thinking(text: str) -> Optional[str]:
    """Extract <think> content from LLM output."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else None


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the LLM and execute any tool calls on the robot."""

    # Build the prompt
    system_prompt = """You are Reachy, an embodied AI assistant. Use tools when needed.
Wrap your thinking in <think></think> tags.
Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> to call tools."""

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if request.context:
        messages.append({"role": "system", "content": f"Context: {request.context}"})
    messages.append({"role": "user", "content": request.message})

    # Call Ollama
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

    result = response.json()
    llm_response = result.get("message", {}).get("content", "")

    # Extract thinking
    thinking = extract_thinking(llm_response)

    # Parse and execute tool calls
    tool_calls = parse_tool_calls(llm_response)
    executed = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        print(f"🔧 Executing tool: {name}({args})")
        result = await execute_tool(name, args)
        executed.append({"name": name, "arguments": args, "result": result})

    # Clean up the response (remove tool_call tags for the user)
    clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', llm_response, flags=re.DOTALL)
    clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL)
    clean_response = clean_response.strip()

    return ChatResponse(
        response=clean_response,
        tool_calls_executed=executed,
        thinking=thinking,
    )


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "reachy_connected": reachy is not None and reachy.is_connected(),
        "ollama_model": MODEL_NAME,
    }


@app.post("/robot/look_at")
async def robot_look_at(x: float = 0.5, y: float = 0, z: float = 0.2, duration: float = 1.0):
    """Direct robot control — make Reachy look at a point."""
    result = await execute_tool("robot_look_at", {"x": x, "y": y, "z": z, "duration": duration})
    return result


@app.post("/robot/express/{emotion}")
async def robot_express(emotion: str, intensity: float = 0.7):
    """Direct robot control — make Reachy express an emotion."""
    result = await execute_tool("robot_express", {"emotion": emotion, "intensity": intensity})
    return result


@app.post("/robot/nod")
async def robot_nod():
    """Quick nod gesture."""
    result = await execute_tool("robot_express", {"emotion": "nodding"})
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run the Bridge Server

```bash
source ~/reachy-env/bin/activate
cd ~/reachy-bridge
pip install fastapi uvicorn httpx ddgs
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Test the Bridge

```bash
# Health check
curl http://localhost:8000/health

# Chat (triggers LLM + robot)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! Can you look at me and nod?"}'

# Direct robot control
curl -X POST "http://localhost:8000/robot/look_at?x=0.5&y=0&z=0.2"
curl -X POST "http://localhost:8000/robot/express/happy"
curl -X POST "http://localhost:8000/robot/nod"
```

---

## 8. OpenClaw Integration

On the RTX 5090 machine, OpenClaw can connect to the Orin's bridge server.

### OpenClaw Skill Configuration

In your OpenClaw setup, add a custom skill that calls the Orin bridge:

```javascript
// openclaw-skills/reachy-control.js
const ORIN_BRIDGE = "http://<ORIN_IP>:8000";

module.exports = {
  name: "reachy-control",
  description: "Control the Reachy Mini robot",

  async chat(message) {
    const response = await fetch(`${ORIN_BRIDGE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    return await response.json();
  },

  async lookAt(x, y, z) {
    const response = await fetch(
      `${ORIN_BRIDGE}/robot/look_at?x=${x}&y=${y}&z=${z}`,
      { method: "POST" }
    );
    return await response.json();
  },

  async express(emotion) {
    const response = await fetch(
      `${ORIN_BRIDGE}/robot/express/${emotion}`,
      { method: "POST" }
    );
    return await response.json();
  },
};
```

---

## 9. Troubleshooting

### Can't connect to Reachy

```bash
# Check if Reachy's gRPC port is open (default: 50051)
nc -zv <REACHY_IP> 50051

# Check if Reachy is on the network
nmap -sn 192.168.1.0/24

# Check Orin's network
ip addr show
ping <REACHY_IP>
```

### Ollama out of memory

```bash
# Check memory usage
tegrastats

# Use a smaller quantization
ollama pull qwen2.5:3b-instruct-q4_K_S   # Smaller than Q4_K_M

# Reduce context length in Modelfile
PARAMETER num_ctx 1024   # Instead of 2048

# Kill other processes
sudo systemctl stop gdm   # Stop desktop if running headless
```

### Slow inference

```bash
# Verify CUDA is being used by Ollama
ollama run reachy-copilot "test" --verbose
# Look for "gpu" in the output

# Check GPU usage
tegrastats | head -5
# Look for GR3D_FREQ (GPU frequency) — should be non-zero during inference
```

### SDK import error

```bash
# Make sure grpcio is compatible with ARM64
pip install --force-reinstall grpcio grpcio-tools

# If reachy2-sdk fails, try installing from source:
pip install git+https://github.com/pollen-robotics/reachy2-sdk.git
```

### Network latency

```bash
# Test latency between Orin and Reachy
ping -c 10 <REACHY_IP>
# Should be <2ms on local Ethernet

# If high latency, check for WiFi — use Ethernet instead
```
