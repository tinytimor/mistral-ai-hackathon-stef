#!/usr/bin/env python3
"""
00_quickstart.py — Test your Mistral API connection and verify everything works.

Run this FIRST before generating training data!

Usage:
    # Test Mistral La Plateforme API (recommended first step):
    python scripts/00_quickstart.py --provider mistral

    # Test Microsoft Foundry (Mistral-Large-3):
    python scripts/00_quickstart.py --provider foundry

    # Test local vLLM server:
    python scripts/00_quickstart.py --provider local
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ─── Provider config (same as 01_generate_training_data.py) ──────────────────
PROVIDER = os.getenv("PROVIDER", "foundry")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1/"
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1/")
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "")
AZURE_RESOURCE = os.getenv("AZURE_RESOURCE_NAME", "")
if not FOUNDRY_ENDPOINT and AZURE_RESOURCE:
    FOUNDRY_ENDPOINT = f"https://{AZURE_RESOURCE}.openai.azure.com/openai/v1/"


def test_connection(provider: str):
    """Test the API connection and run a simple tool-calling example."""

    print(f"\n{'='*70}")
    print(f"  🚀 Reachy Copilot — Quickstart Test")
    print(f"  Provider: {provider}")
    print(f"{'='*70}\n")

    # ── Step 1: Import & create client ────────────────────────────────────
    print("📦 Step 1: Creating API client...")

    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai package not installed. Run:")
        print("   pip install openai azure-identity python-dotenv")
        sys.exit(1)

    if provider == "mistral":
        if not MISTRAL_API_KEY:
            print("❌ MISTRAL_API_KEY not set in .env")
            print("   Get your key at: https://console.mistral.ai/")
            sys.exit(1)
        client = OpenAI(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)
        model = os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest")
        print(f"   ✅ Mistral La Plateforme client ready")
        print(f"   🔑 API key: {MISTRAL_API_KEY[:8]}...{MISTRAL_API_KEY[-4:]}")
        print(f"   📡 Endpoint: {MISTRAL_BASE_URL}")

    elif provider == "local":
        client = OpenAI(base_url=LOCAL_BASE_URL, api_key="not-needed")
        model = os.getenv("LOCAL_MODEL", "mistralai/Ministral-3-8B-Instruct-2512")
        print(f"   ✅ Local client ready")
        print(f"   📡 Endpoint: {LOCAL_BASE_URL}")

    else:  # foundry
        if not FOUNDRY_ENDPOINT:
            print("❌ FOUNDRY_ENDPOINT not set.")
            print("   Set AZURE_RESOURCE_NAME or FOUNDRY_ENDPOINT in .env")
            print("   Your deployment: ai.azure.com → stefanlehman2-7147 → Mistral-Large-3")
            sys.exit(1)

        api_key = os.getenv("FOUNDRY_API_KEY") or os.getenv("AZURE_API_KEY")
        if api_key:
            client = OpenAI(base_url=FOUNDRY_ENDPOINT, api_key=api_key)
            print(f"   ✅ Foundry client ready (API key auth)")
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                client = OpenAI(base_url=FOUNDRY_ENDPOINT, api_key=token_provider())
                print(f"   ✅ Foundry client ready (Entra ID auth via az login)")
            except Exception as e:
                print(f"❌ Foundry auth failed: {e}")
                print("   Options:")
                print("   1. Set FOUNDRY_API_KEY in .env (from ai.azure.com portal)")
                print("   2. Run 'az login' for Entra ID auth")
                sys.exit(1)

        model = os.getenv("FOUNDRY_MODEL_NAME", "Mistral-Large-3")
        print(f"   📡 Endpoint: {FOUNDRY_ENDPOINT}")

    print(f"   🤖 Model: {model}")

    # ── Step 2: Simple completion test ────────────────────────────────────
    print(f"\n💬 Step 2: Testing basic chat completion...")
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Reachy, a friendly robot assistant. Be brief."},
                {"role": "user", "content": "Hello! What can you help me with?"},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        dt = time.time() - t0
        reply = response.choices[0].message.content
        print(f"   ✅ Response in {dt:.1f}s:")
        print(f"   💬 {reply[:200]}")
        print(f"   📊 Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
    except Exception as e:
        print(f"   ❌ Chat completion failed: {e}")
        print(f"\n   Debug: Check your endpoint, API key, and model name.")
        sys.exit(1)

    # ── Step 3: Tool-calling test ─────────────────────────────────────────
    print(f"\n🔧 Step 3: Testing tool calling (function calling)...")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email via Gmail.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "robot_speak",
                "description": "Make Reachy Mini speak out loud using TTS.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to speak"},
                        "emotion": {"type": "string", "enum": ["neutral", "happy", "sad", "excited"], "default": "neutral"},
                    },
                    "required": ["text"],
                },
            },
        },
    ]

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Reachy, an embodied AI robot assistant with tool-calling capabilities. Use the available tools to help users."},
                {"role": "user", "content": "Can you search for the weather in NYC and then tell me about it out loud?"},
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=500,
            temperature=0.1,
        )
        dt = time.time() - t0
        msg = response.choices[0].message

        if msg.tool_calls:
            print(f"   ✅ Tool calls received in {dt:.1f}s:")
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                print(f"      🔧 {tc.function.name}({json.dumps(args, indent=2) if isinstance(args, dict) else args})")
        else:
            print(f"   ⚠️  No tool calls returned (model responded with text instead):")
            print(f"   💬 {msg.content[:200]}")

        print(f"   📊 Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")

    except Exception as e:
        print(f"   ❌ Tool calling failed: {e}")
        print(f"   This might be expected for some models. The data gen script handles this.")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ CONNECTION VERIFIED! You're ready to generate training data.")
    print(f"{'='*70}")
    print(f"\n  Next steps:")
    print(f"  1. Generate training data (teacher model creates examples):")
    print(f"     python scripts/01_generate_training_data.py --provider {provider} --model {model} --num-samples 50")
    print(f"")
    print(f"  2. Check the output:")
    print(f"     ls -la data/training_data_*.jsonl")
    print(f"")
    print(f"  3. Fine-tune student model on RTX 5090:")
    print(f"     python scripts/02_sft_qlora.py")
    print(f"")
    print(f"  4. Quantize & deploy on Orin Nano:")
    print(f"     python scripts/04_quantize_deploy.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Mistral API connection for Reachy Copilot")
    parser.add_argument("--provider", choices=["foundry", "mistral", "local"],
                        default=PROVIDER,
                        help="API provider (default: from .env PROVIDER)")
    args = parser.parse_args()
    test_connection(args.provider)
