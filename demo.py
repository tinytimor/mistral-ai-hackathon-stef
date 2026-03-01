#!/usr/bin/env python3
"""Reachy Copilot - Embodied AI Assistant Demo.

┌─────────────────────────────────────────────────────────────────┐
│                    WHAT RUNS WHERE                               │
│                                                                  │
│  🟢 LOCAL (Orin Nano)              🔵 MISTRAL API (Cloud)       │
│  ├─ Ministral 3B (Ollama)          ├─ Vision: mistral-small     │
│  │  Chat + tool-calling            │  (Pixtral built-in)        │
│  ├─ Robot control (reachy SDK)     ├─ ASR: voxtral-mini-2602    │
│  │  Head, antennas, emotions       │  (speech-to-text)          │
│  ├─ OpenClaw Gateway (:18789)      ├─ Fallback: mistral-large   │
│  │  Memory, sessions, routing      │  (complex reasoning)       │
│  └─ Camera capture (SSH→Reachy)    └─ Web: Brave Search API     │
│                                                                  │
│  🟡 EDGE-TTS (Microsoft, free)                                  │
│  └─ Text-to-speech → Reachy speaker                             │
└─────────────────────────────────────────────────────────────────┘

Usage:
  python demo.py                # text input mode (type to chat)
  python demo.py --voice        # voice input mode (speak to Reachy)

Commands (in text mode):
  Type anything → chat with Reachy
  'see' / 'what do you see?' → camera + vision
  'voice' → one-shot voice input
  'quit' → exit
"""

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx
from reachy_mini.reachy_mini import ReachyMini

# ── Configuration ──────────────────────────────────────────────────────────────
REACHY_IP = os.environ.get("REACHY_IP", "10.0.0.129")
REACHY_SSH_USER = os.environ.get("REACHY_SSH_USER", "pollen")
MISTRAL_API_KEY = os.environ.get(
    "MISTRAL_API_KEY", "79Sg6zwrHQourAzu3IZFDmbM2zRQl4SZ"
)
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "BSABjj8611JRSvqpp-kEeEfWx5MEdM9")
OLLAMA_URL = "http://localhost:11434"
OPENCLAW_URL = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "reachy-hackathon-2026"
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-GuyNeural")
VOXTRAL_MODEL = "voxtral-mini-2602"
MIC_RECORD_SECONDS = int(os.environ.get("MIC_SECONDS", "5"))

# Vision keywords that trigger camera capture
VISION_KEYWORDS = [
    "what do you see", "what can you see", "look at", "describe what",
    "what's in front", "what is in front", "take a photo", "take a picture",
    "capture", "camera", "see anything", "what's around", "surroundings",
    "show me", "who is", "who do you see", "what's that", "identify", "recognize",
]

# Conversation history for multi-turn
conversation_history: list[dict] = []
MAX_HISTORY = 10

# ── Helpers ────────────────────────────────────────────────────────────────────
def log_local(msg: str) -> None:
    print(f"  🟢 [LOCAL] {msg}")

def log_api(msg: str) -> None:
    print(f"  🔵 [API]   {msg}")


# ── Microphone + ASR (Voxtral) ────────────────────────────────────────────────
def record_from_reachy(seconds: int = MIC_RECORD_SECONDS) -> str | None:
    """Record audio from Reachy's mic via SSH. Returns local wav path or None."""
    remote_path = "/tmp/reachy_mic.wav"
    local_path = "/tmp/reachy_mic.wav"
    log_local(f"Recording {seconds}s from Reachy mic...")
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=5",
                f"{REACHY_SSH_USER}@{REACHY_IP}",
                f"arecord -D reachymini_audio_src -f S16_LE -r 16000 -c 2 "
                f"-d {seconds} {remote_path} 2>/dev/null",
            ],
            capture_output=True,
            timeout=seconds + 10,
        )
        if result.returncode != 0:
            print(f"  ❌ Mic record failed: {result.stderr.decode()[:150]}")
            return None

        subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=accept-new",
                f"{REACHY_SSH_USER}@{REACHY_IP}:{remote_path}",
                local_path,
            ],
            capture_output=True,
            timeout=10,
        )
        size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        if size < 1000:
            print("  ❌ Recording too small, mic may not be working")
            return None
        log_local(f"Recorded {size} bytes")
        return local_path
    except Exception as e:
        print(f"  ❌ Mic error: {e}")
        return None


def transcribe_audio(wav_path: str) -> str:
    """Transcribe audio using Mistral Voxtral ASR API."""
    log_api(f"Transcribing with Voxtral ({VOXTRAL_MODEL})...")
    try:
        with open(wav_path, "rb") as f:
            resp = httpx.post(
                "https://api.mistral.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"model": VOXTRAL_MODEL},
                timeout=30,
            )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        if text:
            log_api(f"Heard: \"{text}\"")
        else:
            log_api("(silence - no speech detected)")
        return text
    except Exception as e:
        print(f"  ❌ ASR error: {e}")
        return ""


def listen() -> str:
    """Record from Reachy mic → transcribe with Voxtral. Returns text."""
    print("🎤 Listening... (speak now)")
    wav = record_from_reachy()
    if not wav:
        return ""
    return transcribe_audio(wav)


# ── Camera ─────────────────────────────────────────────────────────────────────
def capture_frame() -> bytes | None:
    """Capture a camera frame from Reachy via SSH + SDK GStreamer backend."""
    log_local("Capturing camera frame via SSH→Reachy RPi...")
    capture_script = (
        "import cv2, time, sys; "
        "from reachy_mini.media.media_manager import MediaManager, MediaBackend; "
        "mm = MediaManager(backend=MediaBackend.GSTREAMER, log_level='WARNING', signalling_host='localhost'); "
        "time.sleep(0.5); "
        "frame = mm.get_frame(); "
        "mm.close(); "
        "cv2.imwrite('/tmp/reachy_snap.jpg', frame) if frame is not None else sys.exit(1)"
    )
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=5",
                f"{REACHY_SSH_USER}@{REACHY_IP}",
                f"/venvs/mini_daemon/bin/python3 -c \"{capture_script}\"",
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            print(f"  ❌ Camera capture failed: {result.stderr.decode()[:200]}")
            return None

        local_path = "/tmp/reachy_snap.jpg"
        subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=accept-new",
                f"{REACHY_SSH_USER}@{REACHY_IP}:/tmp/reachy_snap.jpg",
                local_path,
            ],
            capture_output=True,
            timeout=10,
        )
        with open(local_path, "rb") as f:
            data = f.read()
        log_local(f"Captured frame ({len(data)} bytes)")
        return data

    except subprocess.TimeoutExpired:
        print("  ❌ Camera capture timed out")
        return None
    except Exception as e:
        print(f"  ❌ Camera error: {e}")
        return None


# ── Vision (Mistral API) ──────────────────────────────────────────────────────
def describe_image(image_bytes: bytes, question: str = "Describe what you see") -> str:
    """Send image to Mistral API for visual understanding."""
    img_b64 = base64.b64encode(image_bytes).decode()
    system_prompt = (
        "You are Reachy, a friendly robot assistant. Describe what you see "
        "from your camera in 2-3 conversational sentences. Be warm "
        "and specific about what you observe. No markdown formatting."
    )
    for model in ["mistral-small-latest", "pixtral-large-latest"]:
        try:
            log_api(f"Vision → {model}")
            resp = httpx.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{question}\n\n{system_prompt}"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                            }},
                        ],
                    }],
                    "max_tokens": 200,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠️  Vision model {model} failed: {e}")
            continue
    return "I tried to look but my vision system had trouble. Could you try again?"


# ── TTS (edge-tts → Reachy speaker) ───────────────────────────────────────────
def speak(text: str) -> None:
    """Speak text using edge-tts → play on Reachy's speaker via SSH+dmix."""
    clean = _strip_markdown(text)
    if not clean:
        return
    print(f"  🟡 [TTS]   Speaking: {clean[:90]}{'...' if len(clean) > 90 else ''}")

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        asyncio.run(_generate_tts(clean[:500], mp3_path))

        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            print("  ⚠️  TTS empty, falling back to espeak")
            _speak_espeak(clean)
            return

        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ar", "16000", "-ac", "2", "-acodec", "pcm_s16le", wav_path],
            capture_output=True, timeout=10,
        )

        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            _play_on_reachy(wav_path)
        else:
            print("  ⚠️  wav conversion failed, playing locally")
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", mp3_path],
                capture_output=True, timeout=15,
            )

        for p in [mp3_path, wav_path]:
            try:
                os.unlink(p)
            except OSError:
                pass
    except Exception as e:
        print(f"  ⚠️  edge-tts failed ({e}), falling back to espeak")
        _speak_espeak(clean)


def _play_on_reachy(wav_path: str) -> None:
    """SCP wav to Reachy and play via dmix shared audio device."""
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=accept-new",
         wav_path, f"{REACHY_SSH_USER}@{REACHY_IP}:/tmp/reachy_tts.wav"],
        capture_output=True, timeout=10,
    )
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new",
         f"{REACHY_SSH_USER}@{REACHY_IP}",
         "aplay -D reachymini_audio_sink /tmp/reachy_tts.wav 2>/dev/null"],
        capture_output=True, timeout=30,
    )


async def _generate_tts(text: str, output_path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(output_path)


def _speak_espeak(text: str) -> None:
    """Fallback TTS via espeak-ng."""
    try:
        subprocess.run(
            ["espeak-ng", "-s", "145", "-w", "/tmp/espeak_out.wav", text[:200]],
            capture_output=True, timeout=15,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", "/tmp/espeak_out.wav",
             "-ar", "16000", "-ac", "2", "-acodec", "pcm_s16le",
             "/tmp/espeak_out_16k.wav"],
            capture_output=True, timeout=10,
        )
        _play_on_reachy("/tmp/espeak_out_16k.wav")
    except Exception:
        try:
            subprocess.run(["espeak-ng", "-s", "145", text[:200]],
                           capture_output=True, timeout=15)
        except Exception:
            pass


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"#+\s+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[\U0001f300-\U0001f9ff]", "", text)
    return text.strip()


# ── LLM (Ollama - LOCAL) ──────────────────────────────────────────────────────
def ask_model(text: str) -> str:
    """Send text to reachy-copilot via Ollama (LOCAL on Orin Nano)."""
    conversation_history.append({"role": "user", "content": text})
    messages = conversation_history[-MAX_HISTORY:]

    now = datetime.now()
    system_context = (
        f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}. "
        "When searching the web, include today's date for current results. "
        "Give concise spoken answers - no markdown, no bullet points."
    )

    log_local("Thinking with Ministral 3B (Ollama)...")
    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "reachy-copilot",
            "messages": [
                {"role": "system", "content": system_context},
                *messages,
            ],
            "stream": False,
        },
        timeout=90,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    conversation_history.append({"role": "assistant", "content": content})
    return content


def ask_openclaw(text: str) -> str:
    """Send text through OpenClaw Gateway (LOCAL gateway → Ollama)."""
    log_local("Routing through OpenClaw Gateway...")
    try:
        resp = httpx.post(
            f"{OPENCLAW_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENCLAW_TOKEN}",
            },
            json={
                "model": "ollama/reachy-copilot",
                "messages": [{"role": "user", "content": text}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ⚠️  OpenClaw failed ({e}), falling back to direct Ollama")
        return ask_model(text)


# ── Tool Parsing ──────────────────────────────────────────────────────────────
def parse_tool_calls(content: str) -> list[dict]:
    """Extract tool calls from model output like [TOOL_CALLS]name[ARGS]{...}"""
    calls = []
    parts = content.split("[TOOL_CALLS]")
    for part in parts[1:]:
        try:
            name_rest = part.split("[ARGS]", 1)
            if len(name_rest) == 2:
                name = name_rest[0].strip()
                args_str = (
                    name_rest[1]
                    .split("[TOOL_CALLS]")[0]
                    .split("[/INST]")[0]
                    .strip()
                )
                args = json.loads(args_str)
                if name == "tool_call" and "name" in args:
                    calls.append({
                        "name": args["name"],
                        "args": args.get("arguments", args.get("args", {})),
                    })
                else:
                    calls.append({"name": name, "args": args})
        except Exception:
            pass
    return calls


# ── Web Search (Brave API) ────────────────────────────────────────────────────
def search_web(query: str, max_results: int = 3) -> str:
    """Search using Brave Search API."""
    log_api(f"Brave Search: \"{query}\"")
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results, "extra_snippets": "true"},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        parts: list[str] = []

        infobox = data.get("infobox", {})
        if infobox and infobox.get("results"):
            for item in infobox["results"][:2]:
                if item.get("description"):
                    parts.append(item["description"][:300])

        for news in data.get("news", {}).get("results", [])[:2]:
            if news.get("description"):
                age = news.get("age", "")
                parts.append(f"[{age}] {news['title']}: {news['description'][:200]}")

        for disc in data.get("discussions", {}).get("results", [])[:1]:
            if disc.get("description"):
                parts.append(f"Discussion: {disc['description'][:200]}")

        for r in data.get("web", {}).get("results", [])[:max_results]:
            title = r.get("title", "")
            desc = r.get("description", "")
            extra = r.get("extra_snippets", [])
            entry = f"{title}: {desc}"
            if extra:
                entry += " | " + " | ".join(extra[:2])
            parts.append(entry[:500])

        return "\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# ── Tool Executor ─────────────────────────────────────────────────────────────
def execute_tool(reachy: ReachyMini, name: str, args: dict) -> str | None:
    """Execute a tool call on Reachy. Returns result string for info tools."""
    if name in ("look_at", "robot_look_at"):
        x = float(args.get("x", 0.5))
        y = float(args.get("y", 0.0))
        z = float(args.get("z", 0.0))
        log_local(f"look_at({x}, {y}, {z})")
        reachy.look_at_world(x, y, z)

    elif name in ("speak", "robot_speak"):
        speak(args.get("text", ""))

    elif name in ("nod", "robot_nod"):
        log_local("nodding...")
        reachy.look_at_world(0.5, 0.0, 0.15)
        time.sleep(0.4)
        reachy.look_at_world(0.5, 0.0, -0.05)
        time.sleep(0.4)
        reachy.look_at_world(0.5, 0.0, 0.0)

    elif name in ("shake_head", "robot_shake_no"):
        log_local("shaking head...")
        reachy.look_at_world(0.5, 0.3, 0.0)
        time.sleep(0.4)
        reachy.look_at_world(0.5, -0.3, 0.0)
        time.sleep(0.4)
        reachy.look_at_world(0.5, 0.0, 0.0)

    elif name in ("robot_express", "express"):
        emotion = args.get("emotion", "happy")
        log_local(f"expressing: {emotion}")
        EMOTIONS = {
            "happy":    {"head": [0.5, 0.0, 0.05]},
            "sad":      {"head": [0.5, 0.0, -0.15]},
            "curious":  {"head": [0.5, 0.2, 0.05]},
            "surprised":{"head": [0.5, 0.0, 0.1]},
            "thinking": {"head": [0.5, 0.15, 0.0]},
            "excited":  {"head": [0.5, 0.0, 0.08]},
        }
        e = EMOTIONS.get(emotion, EMOTIONS["happy"])
        reachy.look_at_world(*e["head"])

    elif name in ("search_web",):
        query = args.get("query", "")
        max_r = int(args.get("max_results", 3))
        result = search_web(query, max_r)
        print(f"  📋 Results: {result[:150]}{'...' if len(result) > 150 else ''}")
        return result

    elif name in ("robot_see", "see", "capture_image"):
        log_local("Capturing camera frame...")
        img = capture_frame()
        if img:
            desc = describe_image(img, args.get("question", "What do you see?"))
            print(f"  👁  Vision: {desc[:120]}...")
            return desc
        return "I couldn't capture an image right now."

    else:
        print(f"  ⚙️  {name}({args})  [not implemented - would go via OpenClaw]")

    return None


# ── Vision Query Detection ────────────────────────────────────────────────────
def is_vision_query(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in VISION_KEYWORDS)


# ── Main Loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy Copilot Demo")
    parser.add_argument("--voice", action="store_true",
                        help="Use Reachy's mic + Voxtral ASR for voice input")
    args = parser.parse_args()
    voice_mode = args.voice

    now = datetime.now()
    print("=" * 62)
    print("  🤖 REACHY COPILOT - Mistral AI Hackathon 2026")
    print(f"  📅 {now.strftime('%A, %B %d, %Y at %I:%M %p')}")
    print("=" * 62)
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  🟢 LOCAL        │  🔵 MISTRAL API          │")
    print("  │  Ministral 3B    │  Vision (Pixtral)        │")
    print("  │  Robot control   │  ASR (Voxtral)           │")
    print("  │  OpenClaw GW     │  Web (Brave Search)      │")
    print("  │  Camera (SSH)    │  Fallback (Large)        │")
    print("  │  🟡 TTS (edge)   │                          │")
    print("  └─────────────────────────────────────────────┘")
    print()

    print("🤖 Connecting to Reachy Mini...")
    reachy = ReachyMini(connection_mode="network", media_backend="no_media")
    reachy.wake_up()
    time.sleep(1)
    print("✅ Connected! Reachy is awake.\n")

    # Quick nod
    reachy.look_at_world(0.5, 0.0, 0.1)
    time.sleep(0.3)
    reachy.look_at_world(0.5, 0.0, 0.0)

    if voice_mode:
        print("🎤 VOICE MODE - Speak to Reachy! Say 'quit' or 'stop' to exit.")
        print(f"   Recording {MIC_RECORD_SECONDS}s per turn (set MIC_SECONDS env to change)\n")
    else:
        print("⌨️  TEXT MODE - Type to chat. Commands:")
        print("   'see' / 'what do you see?' → camera + vision")
        print("   'voice' → switch to voice mode for one turn")
        print("   'quit' → exit\n")

    # Greet
    speak("Hello! I'm Reachy, your AI assistant. How can I help you today?")

    while True:
        # ── Get user input ─────────────────────────────────────────────
        if voice_mode:
            user_input = listen()
            if not user_input:
                print("  (no speech detected, try again)")
                continue
            print(f"\n🧑 You said: \"{user_input}\"")
        else:
            try:
                user_input = input("\n🧑 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "stop"):
            break
        if user_input.lower() == "voice":
            user_input = listen()
            if not user_input:
                print("  (no speech detected)")
                continue
            print(f"\n🧑 You said: \"{user_input}\"")

        # ── Vision queries ─────────────────────────────────────────────
        if is_vision_query(user_input):
            print("📷 Looking around...")
            try:
                reachy.look_at_world(0.5, 0.0, 0.05)
            except Exception:
                pass

            img = capture_frame()
            if img:
                description = describe_image(img, user_input)
                print(f"\n🤖 Reachy: {description}")
                conversation_history.append(
                    {"role": "assistant", "content": description}
                )
                speak(description)
            else:
                msg = "Sorry, I couldn't get a picture from my camera right now."
                print(f"\n🤖 Reachy: {msg}")
                speak(msg)

            try:
                reachy.look_at_world(0.5, 0.0, 0.0)
            except Exception:
                pass
            continue

        # ── Regular queries: LOCAL Ollama reachy-copilot ────────────────
        print("🤔 Thinking...")
        try:
            response = ask_model(user_input)
        except Exception as e:
            print(f"❌ Model error: {e}")
            continue

        # Check for tool calls in response
        calls = parse_tool_calls(response)

        if calls:
            display = response.split("[TOOL_CALLS]")[0].strip()
            after_text = ""
            last_part = response.split("[TOOL_CALLS]")[-1]
            if "[/INST]" in last_part:
                after_text = last_part.split("[/INST]", 1)[1].strip()
            elif "[ARGS]" in last_part:
                try:
                    args_part = last_part.split("[ARGS]", 1)[1]
                    brace_count = 0
                    end_idx = 0
                    for i, c in enumerate(args_part):
                        if c == "{":
                            brace_count += 1
                        elif c == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    remaining = args_part[end_idx:].strip()
                    if remaining and not remaining.startswith("["):
                        after_text = remaining
                except Exception:
                    pass

            full_display = (display + " " + after_text).strip()
            full_display = re.sub(r"\[/?INST\]", "", full_display).strip()
            full_display = re.sub(r"\[/?THINK\]", "", full_display).strip()
            speak_text = re.sub(r"[*_#`]", "", full_display)
            speak_text = re.sub(r"^-\s+", "", speak_text, flags=re.MULTILINE)
            speak_text = re.sub(r"\s+", " ", speak_text).strip()

            if full_display:
                print(f"\n🤖 Reachy: {full_display}")

            tool_results: list[str] = []
            for call in calls:
                result = execute_tool(reachy, call["name"], call["args"])
                if result:
                    tool_results.append(f"{call['name']}: {result}")
                time.sleep(0.3)

            if tool_results:
                summary_prompt = (
                    "Tool results:\n"
                    + "\n".join(tool_results)
                    + "\n\nGive a short, friendly spoken answer based on the above. "
                    "No markdown, no bullet points, just plain spoken sentences."
                )
                try:
                    spoken = ask_model(summary_prompt)
                    print(f"\n🤖 Reachy: {spoken}")
                    speak(spoken[:500])
                except Exception as e:
                    print(f"  Summary error: {e}")
                    speak(tool_results[0][:200])
            elif speak_text:
                speak(speak_text[:500])

            conversation_history.append(
                {"role": "assistant", "content": full_display or response}
            )
        else:
            print(f"\n🤖 Reachy: {response}")
            speak(response[:500])

    print("\n👋 Shutting down...")
    speak("Goodbye! It was nice chatting with you.")
    try:
        reachy.look_at_world(0.5, 0.0, 0.0)
        time.sleep(0.5)
        reachy.goto_sleep()
    except Exception:
        pass
    reachy.client.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()
