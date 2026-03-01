#!/usr/bin/env python3
"""
01_generate_training_data.py - Generate tool-calling + agentic training data
using Mistral models via Microsoft Foundry, Mistral La Plateforme API, or locally.

Three providers:
    - Microsoft Foundry: Mistral-Large-3 (deployed - sold directly by Azure)
    - Mistral API:       All models incl. Voxtral, Magistral, Ministral 3
    - Local RTX 5090:    Open-source models via vLLM (Apache 2.0)

Open-source models (Apache 2.0, Dec 2025):
    - Ministral 3 3B/8B/14B (v25.12) - vision + agentic, 256k ctx, edge-optimized
    - Mistral Small 3.2 (v25.06) - 24B, vision, function calling
    - Devstral 2 (v25.12) - code agent specialist
    - Magistral Small 1.2 (v25.09) - reasoning model (OPEN)
    - Voxtral Mini 4B Realtime - streaming ASR, Apache 2.0

Distillation strategy:
    Teacher (Mistral-Large-3 via Foundry) → generates high-quality training data
    Student (Ministral 3 8B or 3B) → fine-tune with SFT/GRPO → quantize → deploy on Orin Nano

Prerequisites:
    pip install openai azure-identity datasets python-dotenv

    Create a .env file (see .env.example) with your credentials:
    - For Foundry: set FOUNDRY_ENDPOINT + FOUNDRY_API_KEY (from ai.azure.com portal)
    - For Mistral API: set MISTRAL_API_KEY (get at console.mistral.ai)

Usage:
    # Use Mistral Large 3 as teacher via Foundry (RECOMMENDED):
    python scripts/01_generate_training_data.py --provider foundry --model Mistral-Large-3 --num-samples 500

    # Use Mistral Large 3 via La Plateforme API (alternative):
    python scripts/01_generate_training_data.py --provider mistral --model mistral-large-latest --num-samples 500

    # List all available models and distillation strategy:
    python scripts/01_generate_training_data.py --list-models
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Load .env file FIRST before any Azure imports
from dotenv import load_dotenv
load_dotenv()  # loads from .env in current dir or parent dirs

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI, AzureOpenAI

# ─── Provider config ──────────────────────────────────────────────────────────
# Supports three providers:
#   1. "foundry" - Microsoft Foundry (Mistral-Large-3 only - sold directly by Azure)
#   2. "mistral" - Mistral La Plateforme API (all models incl. Voxtral, no marketplace needed)
#   3. "local"   - Open-source models on RTX 5090 (32GB) via local inference server
#
# NOTE: Foundry partner models (Ministral-3B, Mistral-small-2503, etc.) require
# Azure Marketplace subscription. Use "mistral" provider or "local" instead.
# Only Mistral-Large-3 is sold directly by Azure (no marketplace needed).
PROVIDER = os.getenv("PROVIDER", "foundry")  # "foundry", "mistral", or "local"

# ─── Microsoft Foundry Configuration ─────────────────────────────────────────
# Mistral-Large-3 deployed at: ai.azure.com → Build → Models → Deployments
# Your deployment: stefanlehman2-7147 / Mistral-Large-3 (Global Standard)
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "")

# Fallback: construct from resource name
AZURE_RESOURCE = os.getenv("AZURE_RESOURCE_NAME", "")
if not FOUNDRY_ENDPOINT and AZURE_RESOURCE:
    FOUNDRY_ENDPOINT = f"https://{AZURE_RESOURCE}.cognitiveservices.azure.com/"

# ─── Mistral La Plateforme API (api.mistral.ai) ─────────────────────────────
# Sign up at https://console.mistral.ai/ - free tier includes API credits
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")  # from console.mistral.ai
MISTRAL_BASE_URL = "https://api.mistral.ai/v1/"  # OpenAI-compatible

# ─── Local inference (RTX 5090, 32GB VRAM) ───────────────────────────────────
# Run open-source Mistral models locally via vLLM, Ollama, or llama.cpp
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1/")

# ─── Model selection ─────────────────────────────────────────────────────────
# For distillation: use a LARGE teacher model to generate data,
# then fine-tune a SMALL student model to mimic it.
# Default teacher: Mistral-Large-3 (via Foundry) or mistral-large-latest (via Mistral API)
MODEL = os.getenv("FOUNDRY_MODEL_NAME") or os.getenv("MISTRAL_MODEL_NAME", "Mistral-Large-3")

# If using Mistral API provider, use the La Plateforme model name
if PROVIDER == "mistral" and MODEL == "Mistral-Large-3":
    MODEL = os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest")

# ─── Full Mistral model catalog ─────────────────────────────────────────────
#
# THREE TIERS:
#   1. FOUNDRY  - Mistral-Large-3 ONLY (sold directly by Azure, no marketplace)
#   2. MISTRAL  - La Plateforme API (all models, pay-per-token, incl. Voxtral)
#   3. LOCAL    - Open-source on RTX 5090 (32GB) - free, QLoRA/LoRA fine-tunable
#
# ⚠️  Foundry partner models (Ministral-3B, Mistral-small-2503, Mistral-medium-2505,
#     Codestral-2501) require Azure Marketplace subscription - NOT usable with
#     a direct Azure subscription. Use "mistral" or "local" provider instead.
#
MISTRAL_MODELS = {
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: MICROSOFT FOUNDRY (direct from Azure, no marketplace)
    # ══════════════════════════════════════════════════════════════════════════
    "Mistral-Large-3": {
        "description": "🏆 Mistral Large 3 - 675B / 41B active (MoE). BEST teacher. Deployed on Foundry!",
        "context": 256000,
        "tool_calling": True,
        "params": "675B total / 41B active",
        "provider": "foundry",
        "role": "teacher",
        "pricing": "Azure pay-per-token (sold directly by Azure - NO marketplace needed)",
        "fine_tunable": True,
        "api_name": "Mistral-Large-3",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2: MISTRAL LA PLATEFORME API (all models, pay-per-token)
    # ══════════════════════════════════════════════════════════════════════════
    "mistral-large-latest": {
        "description": "Mistral Large 3 via La Plateforme - same model as Foundry, alternative endpoint",
        "context": 256000,
        "tool_calling": True,
        "params": "675B total / 41B active",
        "provider": "mistral",
        "role": "teacher",
        "pricing": "$0.50 / $1.50 per M tokens",
        "fine_tunable": True,
        "api_name": "mistral-large-latest",
    },
    "mistral-medium-latest": {
        "description": "Mistral Medium 3.1 - PREMIER (closed), frontier multimodal (text+image)",
        "context": 128000,
        "tool_calling": True,
        "params": "undisclosed",
        "provider": "mistral",
        "role": "teacher",
        "pricing": "varies",
        "fine_tunable": False,
        "api_name": "mistral-medium-latest",
    },
    "mistral-small-latest": {
        "description": "Mistral Small 3.2 (v25.06) - OPEN, 24B, vision + function calling ✅",
        "context": 128000,
        "tool_calling": True,
        "params": "~24B",
        "provider": "mistral",
        "role": "student",
        "pricing": "$0.10 / $0.30 per M tokens",
        "fine_tunable": True,
        "api_name": "mistral-small-latest",
    },
    "ministral-8b-latest": {
        "description": "🆕 Ministral 3 8B (v25.12) - vision + agentic, 256k ctx, edge-optimized",
        "context": 256000,
        "tool_calling": True,
        "params": "8.4B LM + 0.4B vision encoder",
        "provider": "mistral",
        "role": "student",
        "pricing": "$0.15 / $0.15 per M tokens",
        "fine_tunable": True,
        "api_name": "ministral-8b-latest",
    },
    "ministral-3b-latest": {
        "description": "🎯 Ministral 3 3B (v25.12) - vision + agentic, edge student, fits 8GB VRAM",
        "context": 256000,
        "tool_calling": True,
        "params": "3.4B LM + 0.4B vision encoder",
        "provider": "mistral",
        "role": "student",
        "pricing": "$0.10 / $0.10 per M tokens",
        "fine_tunable": True,
        "api_name": "ministral-3b-latest",
    },
    "open-mistral-nemo": {
        "description": "Mistral Nemo 12B - open-source, good mid-size student",
        "context": 128000,
        "tool_calling": True,
        "params": "12B",
        "provider": "mistral",
        "role": "student",
        "pricing": "$0.15 / $0.15 per M tokens",
        "fine_tunable": True,
        "api_name": "open-mistral-nemo",
    },
    "codestral-latest": {
        "description": "Codestral (v25.08) - PREMIER (closed), code specialist (via API)",
        "context": 262144,
        "tool_calling": False,
        "params": "22B",
        "provider": "mistral",
        "role": "specialist",
        "pricing": "$0.30 / $0.90 per M tokens",
        "fine_tunable": True,
        "api_name": "codestral-latest",
    },
    "magistral-small-latest": {
        "description": "🧠 Magistral Small 1.2 (v25.09) - OPEN reasoning model, great for complex planning",
        "context": 40960,
        "tool_calling": True,
        "params": "~24B",
        "provider": "mistral",
        "role": "specialist",
        "pricing": "$0.10 / $0.30 per M tokens",
        "fine_tunable": False,
        "api_name": "magistral-small-latest",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3: LOCAL - Open-source models for RTX 5090 (32GB VRAM)
    #         QLoRA (4-bit) training or full fine-tune for smaller models
    #         ALL Apache 2.0 - fully open weights from HuggingFace
    # ══════════════════════════════════════════════════════════════════════════
    "Ministral-3-8B-Local": {
        "description": "🆕📦 Ministral 3 8B (Dec 2025) - vision + agentic, 256k ctx, Apache 2.0",
        "context": 256000,
        "tool_calling": True,
        "params": "8.4B LM + 0.4B vision encoder",
        "provider": "local",
        "role": "student",
        "pricing": "FREE - Apache 2.0",
        "fine_tunable": True,
        "api_name": "mistralai/Ministral-3-8B-Instruct-2512",
        "hf_model": "mistralai/Ministral-3-8B-Instruct-2512",
        "vram_bf16": "~17GB",
        "vram_4bit": "~5GB",
        "training": "QLoRA (fits easily on 5090) or LoRA",
    },
    "Ministral-3-3B-Local": {
        "description": "🎯📦 Ministral 3 3B (Dec 2025) - vision + agentic, fits 8GB, perfect for Orin Nano",
        "context": 256000,
        "tool_calling": True,
        "params": "3.4B LM + 0.4B vision encoder",
        "provider": "local",
        "role": "student",
        "pricing": "FREE - Apache 2.0",
        "fine_tunable": True,
        "api_name": "mistralai/Ministral-3-3B-Instruct-2512",
        "hf_model": "mistralai/Ministral-3-3B-Instruct-2512",
        "vram_bf16": "~8GB",
        "vram_4bit": "~2.5GB",
        "training": "QLoRA or full fine-tune (fits entirely in 5090 BF16)",
    },
    "Ministral-3-14B-Local": {
        "description": "📦 Ministral 3 14B (Dec 2025) - vision + agentic, highest quality edge model",
        "context": 256000,
        "tool_calling": True,
        "params": "14B+ LM + 0.4B vision encoder",
        "provider": "local",
        "role": "student",
        "pricing": "FREE - Apache 2.0",
        "fine_tunable": True,
        "api_name": "mistralai/Ministral-3-14B-Instruct-2512",
        "hf_model": "mistralai/Ministral-3-14B-Instruct-2512",
        "vram_bf16": "~28GB",
        "vram_4bit": "~8GB",
        "training": "QLoRA only (4-bit base ~8GB + adapter + optimizer ≈ 16GB)",
    },
    "Mistral-Small-3.2-Local": {
        "description": "📦 Mistral Small 3.2 (Jun 2025) - 24B, vision, OPEN Apache 2.0",
        "context": 128000,
        "tool_calling": True,
        "params": "~24B",
        "provider": "local",
        "role": "student",
        "pricing": "FREE - Apache 2.0",
        "fine_tunable": True,
        "api_name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "hf_model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "vram_bf16": "~48GB (too big for BF16)",
        "vram_4bit": "~13GB",
        "training": "QLoRA only (4-bit base ~13GB + adapter + optimizer ≈ 20GB)",
    },
    "Devstral-2-Local": {
        "description": "🔧📦 Devstral 2 (Dec 2025) - OPEN, code agent specialist, Apache 2.0",
        "context": 262144,
        "tool_calling": True,
        "params": "~24B",
        "provider": "local",
        "role": "specialist",
        "pricing": "FREE - Apache 2.0",
        "fine_tunable": True,
        "api_name": "mistralai/Devstral-2-2512",
        "hf_model": "mistralai/Devstral-2-2512",
        "vram_bf16": "~48GB (too big for BF16)",
        "vram_4bit": "~13GB",
        "training": "QLoRA only",
    },
    "Mistral-Nemo-12B-Local": {
        "description": "📦 Mistral Nemo 12B Instruct - QLoRA on 5090, Apache 2.0",
        "context": 128000,
        "tool_calling": True,
        "params": "12B",
        "provider": "local",
        "role": "student",
        "pricing": "FREE - Apache 2.0",
        "fine_tunable": True,
        "api_name": "mistralai/Mistral-Nemo-Instruct-2407",
        "hf_model": "mistralai/Mistral-Nemo-Instruct-2407",
        "vram_bf16": "~24GB",
        "vram_4bit": "~7GB",
        "training": "QLoRA (fits on 5090, ~24GB BF16 + adapter overhead)",
    },

    # ── VOICE / ASR MODELS (for speech → text → tool-calling pipeline) ───────
    "Voxtral-Mini-4B-Realtime": {
        "description": "🎤 Voxtral Mini 4B Realtime - streaming ASR, Apache 2.0, runs on RTX 5090",
        "context": 131072,  # ~3 hours of audio
        "tool_calling": False,
        "params": "4B (3.4B LM + 970M audio encoder)",
        "provider": "local",
        "role": "specialist",
        "pricing": "FREE - Apache 2.0, self-hosted",
        "fine_tunable": False,
        "api_name": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "hf_model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "vram_bf16": "~8GB",
        "vram_4bit": "~3GB",
        "gpu_req": "≥16GB (RTX 5090 ✅, Orin Nano ✅ with INT4)",
        "latency": "sub-200ms configurable (80ms–2.4s)",
        "languages": "en, fr, de, es, it, pt, nl, ru, uk, ja, ko, zh, ar",
        "deployment": "vLLM /v1/realtime WebSocket or Transformers VoxtralRealtimeForConditionalGeneration",
    },
}

# ─── Tool definitions - OpenClaw-style personal AI assistant tools ────────────
# These mirror the actual capabilities of OpenClaw (openclaw.ai) skills:
# gog (Gmail/Calendar/Drive), imsg (iMessage), wacli (WhatsApp), xurl (Twitter/X),
# bluebubbles, signal, browser, spotify, smart home, etc.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for real-time information using Brave Search or DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email via Gmail using the gog CLI (Google Workspace integration).",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body text (plain text or HTML)"},
                    "cc": {"type": "string", "description": "CC recipients (comma-separated)", "default": ""},
                    "is_html": {"type": "boolean", "description": "Whether body is HTML", "default": False},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_email",
            "description": "Search Gmail inbox using Gmail search syntax via gog CLI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Gmail search query (e.g. 'newer_than:7d from:boss@company.com')"},
                    "max_results": {"type": "integer", "description": "Max emails to return", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_imessage",
            "description": "Send an iMessage or SMS via the imsg CLI (macOS Messages.app bridge).",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Phone number (E.164) or Apple ID email"},
                    "text": {"type": "string", "description": "Message text to send"},
                    "service": {"type": "string", "enum": ["imessage", "sms"], "description": "Messaging service", "default": "imessage"},
                },
                "required": ["to", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp",
            "description": "Send a WhatsApp message via wacli or the OpenClaw message tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Phone number in E.164 format or group JID"},
                    "message": {"type": "string", "description": "Message text"},
                },
                "required": ["to", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_signal",
            "description": "Send a Signal message via signal-cli integration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Phone number in E.164 format"},
                    "message": {"type": "string", "description": "Message text"},
                },
                "required": ["to", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram",
            "description": "Send a Telegram message to a chat or user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Chat ID or @username"},
                    "message": {"type": "string", "description": "Message text"},
                },
                "required": ["to", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_events",
            "description": "List upcoming calendar events from Google Calendar via gog CLI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: primary)", "default": "primary"},
                    "from_date": {"type": "string", "description": "Start date (ISO 8601)"},
                    "to_date": {"type": "string", "description": "End date (ISO 8601)"},
                    "max_results": {"type": "integer", "description": "Max events to return", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Create a new calendar event in Google Calendar via gog CLI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "from_date": {"type": "string", "description": "Start datetime (ISO 8601)"},
                    "to_date": {"type": "string", "description": "End datetime (ISO 8601)"},
                    "description": {"type": "string", "description": "Event description", "default": ""},
                    "attendees": {"type": "string", "description": "Comma-separated attendee emails", "default": ""},
                },
                "required": ["summary", "from_date", "to_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "Control a browser - navigate, screenshot, click, type, extract content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "screenshot", "click", "type", "snapshot", "evaluate"],
                        "description": "Browser action to perform",
                    },
                    "url": {"type": "string", "description": "URL to navigate to (for navigate action)"},
                    "node": {"type": "string", "description": "DOM node selector (for click/type)"},
                    "text": {"type": "string", "description": "Text to type (for type action)"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_look_at",
            "description": "Make the robot look at a point in 3D space (meters). X=forward, Y=left, Z=up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Forward distance in meters"},
                    "y": {"type": "number", "description": "Left-right offset (positive=left)"},
                    "z": {"type": "number", "description": "Up-down offset (positive=up)"},
                    "duration": {"type": "number", "description": "Movement duration in seconds", "default": 1.0},
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_express",
            "description": "Make the robot express an emotion via head/antenna movements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": ["happy", "sad", "curious", "surprised", "thinking", "nodding", "shaking_no"],
                        "description": "The emotion to express",
                    },
                    "intensity": {"type": "number", "description": "Intensity 0.0-1.0", "default": 0.7},
                },
                "required": ["emotion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_speak",
            "description": "Make the robot speak text aloud using text-to-speech (ElevenLabs/system TTS).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                    "language": {"type": "string", "description": "Language code", "default": "en"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a timed reminder via cron job or Apple Reminders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The reminder message"},
                    "minutes": {"type": "integer", "description": "Minutes from now"},
                    "channel": {"type": "string", "description": "Where to deliver (whatsapp/telegram/imessage)", "default": "imessage"},
                },
                "required": ["message", "minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smart_home",
            "description": "Control smart home devices - Philips Hue lights, 8Sleep mattress, Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_type": {"type": "string", "enum": ["light", "thermostat", "lock", "speaker"], "description": "Device type"},
                    "action": {"type": "string", "enum": ["on", "off", "set", "status"], "description": "Action to perform"},
                    "device_name": {"type": "string", "description": "Device name or room"},
                    "value": {"type": "string", "description": "Value to set (brightness %, temperature, etc.)", "default": ""},
                },
                "required": ["device_type", "action", "device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_control",
            "description": "Control Spotify playback - play, pause, skip, search, queue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["play", "pause", "skip", "previous", "search", "queue", "now_playing"], "description": "Spotify action"},
                    "query": {"type": "string", "description": "Search query or track/playlist name"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_tweet",
            "description": "Post a tweet or reply on Twitter/X via the xurl CLI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Tweet text (max 280 chars)"},
                    "reply_to": {"type": "string", "description": "Post ID to reply to (optional)"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search the agent's persistent memory for past conversations and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

# ─── Scenario templates - Tiered by complexity ──────────────────────────────
# SHORT: 1 tool, instant reaction, <200ms target on edge
# MEDIUM: 2-3 tools, requires planning, some conditional logic
# LONG: 4+ tools, multi-step reasoning, conditional branching, reflection

SCENARIOS_SHORT = [
    # ── Single tool, immediate response ──
    "User says 'turn off the living room lights.'",
    "User says 'what song is playing right now?'",
    "User says 'play some chill lo-fi beats.'",
    "User says 'look at me' - robot should look straight ahead at face level.",
    "User says 'nod if you understand' - robot should nod.",
    "User says 'what's on my calendar today?'",
    "User says 'set the bedroom lights to 30% brightness and warm white.'",
    "User says 'what's the temperature in the house right now?'",
    "User says 'skip this track and queue up Bohemian Rhapsody.'",
    "User says 'play my Discover Weekly playlist.'",
    "User says 'send a WhatsApp to +15559876543 saying I'm running 15 minutes late.'",
    "User says 'text my mom that I'll be home for dinner at 7' - number is +15551234567.",
    "User says 'send an iMessage to john@icloud.com saying the meeting is confirmed.'",
    "User says 'send a Signal message to +15551112222 - tell them the documents are encrypted and ready.'",
    "User says 'send a Telegram message to @devteam_chat saying the deployment succeeded.'",
    "User says 'schedule a meeting with the AI team tomorrow from 2-3pm titled Sprint Planning.'",
    "User says 'set a reminder in 30 minutes to take my medication.'",
    "User says 'tweet: Just shipped our hackathon project - an embodied AI assistant on @ReachyRobot! 🤖🦞'",
    "User says 'what can you do?' - robot should explain its OpenClaw capabilities.",
    "User says something the robot can't help with: 'can you order me an Uber?'",
    "User gives a vague request: 'do something useful.'",
    "User says 'you seem excited!' - robot should express happy.",
    "User says 'send an email to sarah@company.com about rescheduling our 3pm meeting to 4pm.'",
    "User says 'block out Friday afternoon for deep work - no meetings.'",
    "User says 'take a screenshot of the current page and describe what you see.'",
]

SCENARIOS_MEDIUM = [
    # ── 2-3 tools, requires thinking about order and dependencies ──
    "User says 'check my email for anything important from the last 24 hours' - search email, then look at user and speak the summary aloud.",
    "User says 'find all emails from Amazon in the last week and summarize my recent orders' - search email, then speak the summary.",
    "User says 'draft a reply to the last email from my boss - tell them the report will be ready by Friday' - search for boss's email, then send reply.",
    "User says 'lock the front door and turn off all downstairs lights.' - two smart home actions.",
    "User says 'move my 10am to 11am and let the attendees know' - list events to find the 10am, then create/modify event.",
    "User asks 'do I have any conflicts next week?' - list events, analyze for overlaps, speak result.",
    "User says 'go to Hacker News and tell me the top 5 stories right now' - browse website, then speak summary.",
    "User says 'open Amazon and check the price of the Sony WH-1000XM5 headphones' - browse, then speak price.",
    "User says 'check the weather, then if it's nice, message the group chat on Signal about meeting at the park at 4pm.' - search web for weather, conditionally send Signal message.",
    "User says 'WhatsApp the group about tonight's dinner reservation at 8pm' - check calendar for the reservation details, then send WhatsApp.",
    "User says 'check my mentions on Twitter and summarize any replies' - search web for mentions, speak summary.",
    "User says 'read me my emails while I eat breakfast' - search email, look at user, speak the summaries aloud.",
    "User says 'use Signal to tell Alex the meeting room changed to B204, and set a reminder for the meeting in 1 hour.'",
    "User says 'turn on the porch lights and play jazz music.' - smart home + spotify.",
    "User says 'I have a meeting in 30 minutes - what is it about?' - check calendar, then search email for context, speak the brief.",
    "User says 'message everyone on WhatsApp that the hackathon demo starts in 30 minutes' - send WhatsApp, express excited, speak confirmation.",
    "User says 'unsubscribe me from marketing emails' - search recent promo emails, explain what was found, offer to help.",
    "User says 'check my recent iMessage conversations and tell me if anyone needs a reply' - memory search + speak summary.",
]

SCENARIOS_LONG = [
    # ── 4+ tools, multi-step reasoning, conditional branching, reflection ──
    "User says 'check my email for any meeting invites, add them to my calendar, and text me a summary on WhatsApp' - search email → parse invites → create calendar events → send WhatsApp summary → reflect on what was done.",
    "User asks 'what time is my flight tomorrow?' - search email for booking confirmations → check calendar → cross-reference → speak the answer → set a reminder for departure.",
    "User says 'search for the best pizza places near me, send the top 3 to my wife on iMessage, and remind me to make a reservation in 2 hours' - web search → filter results → send iMessage → set reminder → confirm.",
    "User says 'turn on the porch lights, play jazz music, and send a WhatsApp to the dinner guests that I'm ready for them' - smart home → spotify → WhatsApp → express happy → speak confirmation.",
    "User says 'look up today's Hacker News top stories, draft an email summarizing the AI ones, and post a tweet about the most interesting one' - browse HN → filter for AI → draft email → send email → compose tweet → post tweet → reflect.",
    "User says 'help me plan a productive morning routine' - think about components → search web for best practices → create 5 calendar events → set 3 reminders → speak the plan → express encouraging.",
    "User says 'I have a presentation in 2 hours, help me prepare' - check calendar for presentation details → search web for the topic → search memory for past prep notes → express encouragement → set reminder at T-15min → speak the brief.",
    "User says 'I want to disconnect this weekend - help me set up an auto-reply on email and let my close contacts know on WhatsApp and iMessage' - draft auto-reply email → search memory for close contacts → send WhatsApp to each → send iMessage to each → create calendar block → reflect on coverage.",
    "User says 'my mom's birthday is next week - help me plan something' - search web for gift ideas → check calendar for conflicts → search memory for mom's preferences → create calendar event for party → draft email invite → send WhatsApp to family → speak the plan.",
    "User says 'check my messages across all platforms and give me a briefing' - search email → search memory for WhatsApp context → search memory for iMessage context → look at user → speak comprehensive briefing → express appropriate emotion.",
    "User asks 'email my boss, text my wife, and play some music - oh and turn off the kitchen lights' - search memory for boss email + wife number → send email → send iMessage → play spotify → smart home lights off → speak confirmation of all 4.",
    "User says 'I'm hosting a dinner party tonight - help me get ready' - check calendar for guest list → search web for recipe ideas → turn on ambient lights → play dinner playlist → send WhatsApp to guests with arrival time → set reminder for oven → speak the plan.",
    "User says 'I think I double-booked myself tomorrow - can you check and fix it?' - list calendar events → identify conflicts → search email for context on each → decide which to move → reschedule the less important one → email affected attendees → speak what was done.",
    "User says 'give me a full morning briefing' - check calendar for today → search email for overnight messages → search web for news + weather → look at user → speak comprehensive briefing → express appropriate emotion based on schedule density.",
    "User says 'help me prepare for my job interview at Google next Tuesday' - check calendar to confirm → search web for Google interview tips → search memory for relevant experience notes → create study calendar events for the weekend → set daily reminders → draft a thank-you email template → speak the prep plan.",
]

# Combined flat list for backward compatibility
SCENARIOS = SCENARIOS_SHORT + SCENARIOS_MEDIUM + SCENARIOS_LONG

# Complexity metadata for each tier
SCENARIO_TIERS = (
    [(s, "short") for s in SCENARIOS_SHORT]
    + [(s, "medium") for s in SCENARIOS_MEDIUM]
    + [(s, "long") for s in SCENARIOS_LONG]
)

# ─── System prompt (what we train the model to internalize) ──────────────────
SYSTEM_PROMPT = """You are Reachy, an embodied AI personal assistant running on a Reachy Mini robot with OpenClaw-style capabilities. You can:
- Search the web for real-time information
- Send and read emails (Gmail via gog CLI)
- Send messages on iMessage, WhatsApp, Signal, Telegram
- Manage Google Calendar (list events, create events, check conflicts)
- Control a browser (navigate, screenshot, fill forms, extract content)
- Control smart home devices (lights, thermostat, locks)
- Control Spotify playback (play, pause, skip, search, queue)
- Post tweets on Twitter/X via xurl CLI
- Search your persistent memory for past context
- Control your robotic head (look at things, express emotions)
- Speak aloud to the user via TTS
- Set reminders and cron jobs

BEFORE every response, wrap your reasoning in <think>...</think> tags. Inside, follow this process:

1. THINK: What is the user asking? What's the intent behind their words?
2. PLAN: Which tools do I need? In what order? Are there dependencies between steps?
   - For simple tasks (1 tool): State the tool and call it.
   - For medium tasks (2-3 tools): Plan the sequence and note any dependencies.
   - For complex tasks (4+ tools): Break into numbered steps. Note which steps depend on results from earlier steps. Identify any conditional branches ("if X then Y, else Z").
3. ACT: Call the tools in the planned order.
4. REFLECT: Did I fully address the user's needs? Did anything unexpected happen? Should I do anything proactively?

Example reasoning:
<think>
The user wants to know their flight time tomorrow. This requires:
1. Search email for booking confirmations (flight info is usually emailed)
2. Check calendar for any flight-related events
3. Cross-reference both sources to give a confident answer
4. Optionally set a reminder for departure
Step 2 can run in parallel with step 1. Step 3 depends on both. Step 4 is proactive.
</think>

You run as a 24/7 personal AI assistant. Be proactive, helpful, and natural.
Express appropriate emotions through your robotic head while communicating.
If you're unsure, ask for clarification. Respect privacy - confirm before
sending messages on behalf of the user."""


def create_client(provider: str = None) -> OpenAI:
    """Create an OpenAI-compatible client for Foundry, Mistral API, or local inference.

    Args:
        provider: "foundry", "mistral", or "local". Defaults to PROVIDER env var.

    Foundry auth:
        1. FOUNDRY_API_KEY env var (grab from ai.azure.com → Build → Models → Deployments → Keys)
        2. Entra ID via DefaultAzureCredential (run `az login` first)

    Mistral auth:
        - MISTRAL_API_KEY env var (from console.mistral.ai)

    Local:
        - Expects vLLM / Ollama / llama.cpp server at LOCAL_BASE_URL
    """
    provider = provider or PROVIDER

    if provider == "mistral":
        # Mistral La Plateforme - all models, no marketplace
        if not MISTRAL_API_KEY:
            print("❌ MISTRAL_API_KEY not set. Get one at https://console.mistral.ai/")
            sys.exit(1)
        return OpenAI(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)

    if provider == "local":
        # Local inference server (vLLM, Ollama, llama.cpp)
        return OpenAI(base_url=LOCAL_BASE_URL, api_key="not-needed")

    # Microsoft Foundry - Mistral-Large-3 (sold directly by Azure)
    if not FOUNDRY_ENDPOINT:
        print("❌ FOUNDRY_ENDPOINT not set. Options:")
        print("   1. Set FOUNDRY_ENDPOINT to your deployment URL from ai.azure.com")
        print("   2. Set AZURE_RESOURCE_NAME for legacy endpoint format")
        print("   Your deployment: stefanlehman2-7147 → Build → Models → Mistral-Large-3")
        sys.exit(1)

    api_key = os.getenv("FOUNDRY_API_KEY") or os.getenv("AZURE_API_KEY")
    if api_key:
        return AzureOpenAI(
            azure_endpoint=FOUNDRY_ENDPOINT,
            api_key=api_key,
            api_version="2024-12-01-preview",
        )
    else:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=FOUNDRY_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )


def list_mistral_models():
    """Print all available Mistral models - Foundry, La Plateforme, and local."""
    print("\n🔍 Mistral AI model catalog for Reachy Copilot:")
    print("=" * 85)

    # Group by provider tier
    for tier_label, tier_key in [
        ("☁️  TIER 1: MICROSOFT FOUNDRY (direct from Azure - no marketplace needed)", "foundry"),
        ("🌐 TIER 2: MISTRAL LA PLATEFORME API (all models, pay-per-token)", "mistral"),
        ("🖥️  TIER 3: LOCAL on RTX 5090 (32GB) - open-source, free", "local"),
    ]:
        print(f"\n  {tier_label}")
        print("  " + "-" * 81)

        for name, info in MISTRAL_MODELS.items():
            if info["provider"] != tier_key:
                continue
            tc = "✅" if info["tool_calling"] else "❌"
            ft = "✅" if info["fine_tunable"] else "❌"
            role_icon = {"teacher": "🏫", "student": "🎓", "specialist": "🔧"}
            role = role_icon.get(info["role"], "📦") + " " + info["role"].upper()
            print(f"\n    📦 {name}")
            print(f"       {info['description']}")
            print(f"       {role} | Params: {info['params']} | Context: {info['context']:,}")
            print(f"       Tool calling: {tc} | Fine-tunable: {ft} | {info['pricing']}")
            if "vram_bf16" in info:
                print(f"       VRAM: {info['vram_bf16']} BF16 / {info['vram_4bit']} 4-bit | Training: {info['training']}")
            if "gpu_req" in info:
                print(f"       GPU: {info['gpu_req']} | Latency: {info['latency']}")
                print(f"       Languages: {info['languages']}")

    print("\n" + "=" * 85)
    print("  ⚠️  IMPORTANT: Foundry partner models (Ministral-3B, Mistral-small-2503,")
    print("     Mistral-medium-2505, Codestral-2501) require Azure Marketplace subscription.")
    print("     Only Mistral-Large-3 is sold directly by Azure (no marketplace).")
    print()
    print("  💡 DISTILLATION STRATEGY:")
    print("     1. Teacher: Mistral-Large-3 via Foundry (your deployment) or mistral-large-latest via API")
    print("        → Generates highest-quality training data with tool calling")
    print("     2. Student: Fine-tune Ministral 3 8B or 3B locally on RTX 5090 with QLoRA")
    print("        → Quantize to Q4_K_M GGUF → Deploy on Orin Nano (8GB)")
    print()
    print("  🎤 VOICE PIPELINE (Reachy Copilot):")
    print("     Voxtral Mini 4B (local) → streaming ASR → tool-calling LLM → Reachy actions")
    print("     Apache 2.0 - fully self-hosted, no API costs!")
    print()
    print("  💡 USAGE:")
    print("     Teacher (Foundry):      --provider foundry --model Mistral-Large-3")
    print("     Teacher (API):          --provider mistral --model mistral-large-latest")
    print("     Student (API):          --provider mistral --model ministral-8b-latest")
    print("     Student (Local 5090):   --provider local   --model mistralai/Ministral-3-8B-Instruct-2512")
    print("     Orin Nano target:       --provider local   --model mistralai/Ministral-3-3B-Instruct-2512")
    print()


def _api_call_with_retry(func, *args, max_retries=5, **kwargs):
    """Wrapper that retries API calls with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower() or "RateLimitReached" in err_str:
                wait = (2 ** attempt) * 3 + random.uniform(1, 3)
                print(f"  ⏳ Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise


def generate_conversation(client: OpenAI, scenario: str, model: str = "mistral-large-latest") -> dict | None:
    """Use Mistral to generate a training conversation for the given scenario."""
    meta_prompt = f"""You are generating training data for a small language model that will run
on an embodied robot. Given the scenario below, generate a realistic multi-turn conversation.

The conversation MUST follow this format:
1. A system message setting up the assistant's capabilities
2. A user message based on the scenario
3. An assistant response that uses tool calls (using the provided tools)
4. Tool response messages with realistic mock data
5. A final assistant response that synthesizes the tool results

The assistant should demonstrate think-plan-act-reflect reasoning in its responses.
Include inner reasoning in <think>...</think> tags before tool calls.

SCENARIO: {scenario}

Generate the conversation as a JSON array of message objects with "role" and "content" fields.
For tool calls, use the standard OpenAI format with "tool_calls" array.
For tool responses, use role "tool" with "tool_call_id" and "content".

IMPORTANT: Return ONLY the JSON array, no other text."""

    try:
        response = _api_call_with_retry(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "You generate high-quality training data for AI models. Output only valid JSON."},
                {"role": "user", "content": meta_prompt},
            ],
            tools=TOOLS,
            temperature=0.8,
            max_tokens=4096,
        )

        content = response.choices[0].message.content
        if not content:
            return None

        # Try to parse the JSON from the response
        # Handle cases where the model wraps in ```json ... ```
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        messages = json.loads(content)

        return {
            "scenario": scenario,
            "messages": messages,
            "tools": [t["function"] for t in TOOLS],
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠ Failed to generate for scenario: {e}", file=sys.stderr)
        return None


def generate_direct_tool_call(client: OpenAI, scenario: str, model: str = "mistral-large-latest",
                              complexity: str = "short") -> dict | None:
    """Generate a direct tool-calling example by letting the model actually call tools.

    Args:
        complexity: "short" (1 tool), "medium" (2-3 tools), "long" (4+ tools)
    """
    # Complexity-aware user message that forces the teacher to show its reasoning
    cot_prefix = {
        "short": "Respond with a brief <think> tag showing your intent, then call the right tool.",
        "medium": "This task needs 2-3 steps. In your <think> tags, plan the sequence and note dependencies between steps before acting.",
        "long": "This is a complex multi-step task. In your <think> tags, break it into numbered steps, note dependencies and conditional branches, then execute step by step. After all tool results, reflect on completeness.",
    }
    augmented_scenario = f"{scenario}\n\n[Instruction: {cot_prefix.get(complexity, cot_prefix['short'])}]"

    try:
        response = _api_call_with_retry(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": augmented_scenario},
            ],
            tools=TOOLS,
            temperature=0.7,
            max_tokens=2048,
        )

        msg = response.choices[0].message
        # Store the original scenario (without augmentation) in the training data
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario},
        ]

        # If the model made tool calls, record them
        if msg.tool_calls:
            assistant_msg = {"role": "assistant", "content": msg.content or "", "tool_calls": []}
            for tc in msg.tool_calls:
                assistant_msg["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })
            messages.append(assistant_msg)

            # Generate mock tool responses
            for tc in msg.tool_calls:
                mock_result = _generate_mock_tool_result(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(mock_result),
                })

            # Get the final response after tool results
            final = _api_call_with_retry(
                client.chat.completions.create,
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=0.7,
                max_tokens=1024,
            )
            final_msg = final.choices[0].message
            messages.append({"role": "assistant", "content": final_msg.content or ""})
        else:
            messages.append({"role": "assistant", "content": msg.content or ""})

        return {
            "scenario": scenario,
            "complexity": complexity,
            "messages": messages,
            "tools": [t["function"] for t in TOOLS],
        }
    except Exception as e:
        print(f"  ⚠ Failed direct generation: {e}", file=sys.stderr)
        return None


def _generate_mock_tool_result(func_name: str, arguments_str: str) -> dict:
    """Generate realistic mock results for each OpenClaw-style tool."""
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError:
        args = {}

    if func_name == "search_web":
        query = args.get("query", "unknown")
        return {
            "results": [
                {"title": f"Result 1 for: {query}", "snippet": f"Here is relevant information about {query}...", "url": "https://example.com/1"},
                {"title": f"Result 2 for: {query}", "snippet": f"Additional details regarding {query}...", "url": "https://example.com/2"},
            ]
        }
    elif func_name == "send_email":
        return {"status": "sent", "message_id": f"msg-{random.randint(10000, 99999)}", "to": args.get("to", ""), "subject": args.get("subject", "")}
    elif func_name == "search_email":
        return {
            "results": [
                {"from": "boss@company.com", "subject": "Q1 Review", "date": "2026-02-27", "snippet": "Please review the attached Q1 report..."},
                {"from": "hr@company.com", "subject": "Benefits Enrollment", "date": "2026-02-26", "snippet": "Open enrollment ends March 15..."},
            ],
            "total": 2,
        }
    elif func_name == "send_imessage":
        return {"status": "delivered", "to": args.get("to", ""), "service": args.get("service", "imessage")}
    elif func_name == "send_whatsapp":
        return {"status": "sent", "to": args.get("to", ""), "message_id": f"wa-{random.randint(10000, 99999)}"}
    elif func_name == "send_signal":
        return {"status": "sent", "to": args.get("to", ""), "timestamp": "2026-02-28T14:30:00Z"}
    elif func_name == "send_telegram":
        return {"status": "sent", "chat_id": args.get("to", ""), "message_id": random.randint(1000, 9999)}
    elif func_name == "calendar_list_events":
        return {
            "events": [
                {"summary": "Team Standup", "start": "2026-02-28T09:00:00", "end": "2026-02-28T09:30:00", "location": "Zoom"},
                {"summary": "Lunch with Sarah", "start": "2026-02-28T12:00:00", "end": "2026-02-28T13:00:00", "location": "Café Roma"},
                {"summary": "Hackathon Demo", "start": "2026-02-28T16:00:00", "end": "2026-02-28T17:00:00", "location": "Main Stage"},
            ]
        }
    elif func_name == "calendar_create_event":
        return {"status": "created", "event_id": f"evt-{random.randint(10000, 99999)}", "summary": args.get("summary", ""), "link": "https://calendar.google.com/event/abc123"}
    elif func_name == "browser_action":
        action = args.get("action", "navigate")
        if action == "screenshot":
            return {"status": "success", "description": "Screenshot captured: a webpage showing search results with multiple product listings"}
        elif action == "snapshot":
            return {"status": "success", "content": "Page title: Hacker News\n1. Show HN: I built an AI robot assistant (342 points)\n2. GPT-5 released (891 points)"}
        else:
            return {"status": "success", "action": action, "url": args.get("url", "")}
    elif func_name == "robot_look_at":
        return {"status": "success", "message": f"Looking at ({args.get('x', 0)}, {args.get('y', 0)}, {args.get('z', 0)})"}
    elif func_name == "robot_express":
        return {"status": "success", "emotion": args.get("emotion", "neutral"), "message": "Expression displayed"}
    elif func_name == "robot_speak":
        return {"status": "success", "message": "Speech completed", "duration_seconds": len(args.get("text", "")) * 0.05}
    elif func_name == "set_reminder":
        return {"status": "success", "reminder_id": "REM-" + str(random.randint(1000, 9999)), "message": args.get("message", ""), "trigger_in_minutes": args.get("minutes", 0)}
    elif func_name == "smart_home":
        return {"status": "success", "device": args.get("device_name", ""), "action": args.get("action", ""), "state": "on" if args.get("action") in ("on", "set") else "off"}
    elif func_name == "spotify_control":
        action = args.get("action", "play")
        if action == "now_playing":
            return {"track": "Midnight City", "artist": "M83", "album": "Hurry Up, We're Dreaming", "progress": "2:14/4:03"}
        elif action == "search":
            return {"results": [{"name": args.get("query", ""), "artist": "Various", "type": "playlist"}]}
        else:
            return {"status": "success", "action": action}
    elif func_name == "post_tweet":
        return {"status": "posted", "tweet_id": str(random.randint(10**17, 10**18)), "url": "https://x.com/user/status/123456"}
    elif func_name == "memory_search":
        return {"results": [{"content": "User mentioned they prefer morning meetings", "timestamp": "2026-02-25", "relevance": 0.92}]}
    else:
        return {"status": "success", "message": f"{func_name} executed"}


def load_hf_datasets():
    """Load and format existing HuggingFace tool-calling datasets."""
    from datasets import load_dataset

    formatted = []

    # ─── NousResearch Hermes Function Calling (11.5K rows, Apache-2.0) ────
    print("📦 Loading NousResearch/hermes-function-calling-v1...")
    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling", split="train")
        for i, row in enumerate(ds):
            if i >= 2000:  # Take a subset for hackathon speed
                break
            formatted.append({
                "source": "hermes-function-calling-v1",
                "messages": row["conversations"],
                "category": row.get("category", "unknown"),
            })
        print(f"  ✅ Loaded {len(formatted)} from Hermes")
    except Exception as e:
        print(f"  ⚠ Could not load Hermes: {e}")

    # ─── Glaive Function Calling v2 (113K rows, Apache-2.0) ───────────────
    print("📦 Loading glaiveai/glaive-function-calling-v2...")
    try:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        count = 0
        for i, row in enumerate(ds):
            if count >= 2000:
                break
            if random.random() < 0.02:  # Sample ~2% of 113K
                formatted.append({
                    "source": "glaive-function-calling-v2",
                    "messages": row.get("conversations", row.get("chat", "")),
                })
                count += 1
        print(f"  ✅ Loaded {count} from Glaive")
    except Exception as e:
        print(f"  ⚠ Could not load Glaive: {e}")

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Generate training data using Mistral models (Foundry or Mistral API)")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl", help="Output JSONL file")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of synthetic samples to generate")
    parser.add_argument("--include-hf", action="store_true", help="Also include HuggingFace datasets")
    parser.add_argument("--direct-only", action="store_true", help="Use direct tool-calling generation only")
    parser.add_argument("--list-models", action="store_true", help="List all Mistral models (Foundry + API + local) and exit")
    parser.add_argument("--provider", type=str, choices=["foundry", "mistral", "local"], default=None,
                        help="'foundry' (Large-3 only), 'mistral' (La Plateforme API), or 'local' (RTX 5090)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. 'Mistral-Large-3', 'ministral-8b-latest', or HF model ID)")
    args = parser.parse_args()

    # List models and exit if requested
    if args.list_models:
        list_mistral_models()
        sys.exit(0)

    # Override provider/model from CLI args
    provider = args.provider or PROVIDER
    model = args.model or MODEL

    # Auto-detect provider from model name if not explicitly set
    if not args.provider and model in MISTRAL_MODELS:
        provider = MISTRAL_MODELS[model]["provider"]

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    provider_labels = {"foundry": "Microsoft Foundry", "mistral": "Mistral La Plateforme", "local": "Local (RTX 5090)"}
    provider_label = provider_labels.get(provider, provider)
    base_urls = {"foundry": FOUNDRY_ENDPOINT, "mistral": MISTRAL_BASE_URL, "local": LOCAL_BASE_URL}
    base_url = base_urls.get(provider, "")
    print(f"🔐 Connecting to {provider_label}...")
    print(f"   Model: {model}")
    print(f"   Base URL: {base_url}")
    if model in MISTRAL_MODELS:
        info = MISTRAL_MODELS[model]
        print(f"   Role: {info['role'].upper()} | Params: {info['params']} | Context: {info['context']:,}")
    print()

    client = create_client(provider)

    # Test connection
    print("🧪 Testing connection...")
    try:
        test = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'connected' in one word."}],
            max_tokens=10,
        )
        print(f"   ✅ Connected! Response: {test.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        if provider == "foundry":
            print("\n   Make sure you've deployed Mistral-Large-3 at https://ai.azure.com")
            print("   Set FOUNDRY_ENDPOINT or AZURE_RESOURCE_NAME in .env")
            print("   Or run: az login (for Entra ID auth)")
        elif provider == "local":
            print(f"\n   Make sure a local inference server is running at {LOCAL_BASE_URL}")
            print("   Start with: vllm serve mistralai/Ministral-8B-Instruct-2410")
            print("   Or: ollama run ministral:8b")
        else:
            print("\n   Make sure MISTRAL_API_KEY is set in .env")
            print("   Get one at: https://console.mistral.ai/")
        sys.exit(1)

    # Generate synthetic data with complexity tiers
    # Distribution: ~40% short, ~35% medium, ~25% long
    tier_weights = {"short": 0.40, "medium": 0.35, "long": 0.25}
    tier_pools = {
        "short": list(SCENARIOS_SHORT),
        "medium": list(SCENARIOS_MEDIUM),
        "long": list(SCENARIOS_LONG),
    }
    tier_counts = {
        "short": max(1, int(args.num_samples * tier_weights["short"])),
        "medium": max(1, int(args.num_samples * tier_weights["medium"])),
        "long": args.num_samples,  # will be clamped below
    }
    tier_counts["long"] = args.num_samples - tier_counts["short"] - tier_counts["medium"]

    # Build the ordered sample list: short → medium → long
    sample_plan = []
    for tier in ["short", "medium", "long"]:
        pool = tier_pools[tier]
        for j in range(tier_counts[tier]):
            scenario = pool[j % len(pool)]
            # Add contextual variation for second+ passes through the pool
            if j >= len(pool):
                variations = [
                    "The user sounds tired and wants minimal interaction. ",
                    "The user is in a hurry and wants things done fast. ",
                    "The user is a software developer working from home. ",
                    "The user is multitasking while cooking dinner. ",
                    "The user is walking and talking to the robot via voice. ",
                    "The user is at a hackathon and needs help organizing. ",
                    "It's late at night and the user is winding down. ",
                    "The robot is in a home office setup. ",
                    "The user is planning a party this weekend. ",
                    "The user is stressed about a deadline. ",
                    "The user is on vacation and wants remote help. ",
                    "The user is a parent juggling work and kids. ",
                ]
                scenario = random.choice(variations) + scenario
            sample_plan.append((scenario, tier))

    # Shuffle to interleave tiers (prevents all short first)
    random.shuffle(sample_plan)

    print(f"\n🔄 Generating {args.num_samples} training samples...")
    print(f"   Distribution: {tier_counts['short']} short | {tier_counts['medium']} medium | {tier_counts['long']} long")
    samples = []

    for i, (scenario, complexity) in enumerate(sample_plan):
        tier_icon = {"short": "⚡", "medium": "🔧", "long": "🧠"}[complexity]
        print(f"  [{i + 1}/{args.num_samples}] {tier_icon} [{complexity.upper()}] {scenario[:55]}...", flush=True)

        if args.direct_only:
            result = generate_direct_tool_call(client, scenario, model, complexity=complexity)
        else:
            # Alternate between methods for diversity
            if i % 3 == 0:
                result = generate_direct_tool_call(client, scenario, model, complexity=complexity)
            else:
                result = generate_conversation(client, scenario, model)

        if result:
            # Ensure complexity is tagged even for conversation-style samples
            if "complexity" not in result:
                result["complexity"] = complexity
            samples.append(result)

        # Delay between requests to respect rate limits
        time.sleep(0.5)

    # Optionally load HF datasets
    hf_samples = []
    if args.include_hf:
        print("\n📦 Loading HuggingFace datasets...")
        hf_samples = load_hf_datasets()

    # Write output
    print(f"\n💾 Writing {len(samples)} synthetic + {len(hf_samples)} HF samples to {args.output}...")
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
        for sample in hf_samples:
            f.write(json.dumps(sample) + "\n")

    total = len(samples) + len(hf_samples)
    print(f"\n✅ Done! Generated {total} training samples.")
    print(f"   Output: {args.output}")
    print(f"   Synthetic: {len(samples)}")
    print(f"   HuggingFace: {len(hf_samples)}")
    print(f"\n   Next step: python scripts/02_sft_qlora.py --data {args.output}")


if __name__ == "__main__":
    main()
