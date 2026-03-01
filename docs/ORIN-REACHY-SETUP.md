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

The bridge server connects the LLM (Ollama) to the robot (Reachy SDK) and runs
entirely on the Orin Nano. Rather than copying scripts manually, clone the repo:

### Clone the Repo on the Orin

```bash
# On the Orin Nano:
source ~/reachy-env/bin/activate
cd ~

git clone https://github.com/tinytimor/mistral-ai-hackathon-stef.git
cd mistral-ai-hackathon-stef

# Install dependencies
pip install -r requirements.txt

# Copy your .env (or create one)
cp .env.example .env
# Edit REACHY_IP, OLLAMA_MODEL, etc.
```

### Start the Bridge

```bash
cd ~/mistral-ai-hackathon-stef

# Standalone mode (no 5090 / OpenClaw needed):
python scripts/06_openclaw_bridge.py --standalone --reachy-ip 192.168.1.42

# With memory service (optional — run in a separate terminal):
python scripts/05_memory_manager.py --serve --port 8100 &
python scripts/06_openclaw_bridge.py --standalone --reachy-ip 192.168.1.42 --memory-url http://localhost:8100

# With OpenClaw Gateway (if running on 5090):
python scripts/06_openclaw_bridge.py --gateway-host 192.168.1.XX --reachy-ip 192.168.1.42
```

### Pull Updates Later

When you push new code from your Mac or 5090:

```bash
cd ~/mistral-ai-hackathon-stef
git pull origin main
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
