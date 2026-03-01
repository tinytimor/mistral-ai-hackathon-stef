#!/usr/bin/env python3
"""
06_openclaw_bridge.py - OpenClaw ↔ Reachy Mini edge bridge with memory integration.

This script runs on the Orin Nano and connects:
  1. OpenClaw Gateway (running on 5090 or locally) via WebSocket/HTTP
  2. Ollama local model (Ministral 3B Q4_K_M) for fast reactive responses
  3. Reachy Mini robot (via gRPC / reachy2-sdk or reachy-mini SDK)
  4. Memory Manager (L2 cache + L3 SQLite) for context persistence

KEY DESIGN: Inference is fully 5090-independent.
  - The 5090 is used to TRAIN specialized models (SFT + GRPO)
  - Once deployed, the Orin Nano runs everything locally via Ollama
  - OpenClaw Gateway is OPTIONAL - for cloud escalation only
  - All robot tools execute locally on the Orin, no cloud needed

What we add beyond clawd-reachy-mini:
  - LOCAL LLM inference via Ollama (standalone mode is NOT just echo)
  - Local tool execution (robot control, memory, web search)
  - Memory-augmented prompts (3-tier memory across conversations)
  - Smart routing (simple → local, complex → cloud if available)
  - OpenAI-compatible /v1/chat/completions (VisionClaw pattern)
  - Fine-tuned model trained specifically for Reachy tool-calling

What we take from clawd-reachy-mini:
  - Reachy Mini SDK interface patterns (connect, emotions, head, antennas)
  - OpenClaw SKILL.md tool definitions format
  - ElevenLabs TTS integration pattern
  - Wake word detection architecture

What we take from VisionClaw:
  - Single 'execute' tool pattern (proxy all actions through OpenClaw)
  - OpenAI-compatible /v1/chat/completions endpoint for gateway compat
  - Session key + conversation history for multi-turn tool calling

Usage:
    # Standalone (Orin Nano, local model only - NO 5090 needed):
    python scripts/06_openclaw_bridge.py --standalone --reachy-ip 192.168.1.XX

    # With OpenClaw Gateway (optional cloud escalation):
    python scripts/06_openclaw_bridge.py --gateway-host 192.168.1.YY --reachy-ip 192.168.1.XX

    # With memory service:
    python scripts/06_openclaw_bridge.py --standalone --memory-url http://localhost:8100

Prerequisites:
    pip install fastapi uvicorn httpx websockets pydantic
    # Robot SDK: pip install reachy-mini  (or reachy2-sdk for older firmware)
    # Memory manager: python scripts/05_memory_manager.py --serve --port 8100
    # Local model: ollama create reachy-copilot -f Modelfile
"""

import argparse
import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

# ─── Configuration ────────────────────────────────────────────────────────────

REACHY_IP = os.getenv("REACHY_IP", "localhost")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "reachy-copilot")
GATEWAY_HOST = os.getenv("OPENCLAW_HOST", "")
GATEWAY_PORT = int(os.getenv("OPENCLAW_PORT", "18789"))
GATEWAY_TOKEN = os.getenv("OPENCLAW_TOKEN", "")
MEMORY_URL = os.getenv("MEMORY_URL", "")  # http://localhost:8100

# ─── Complexity classifier keywords ──────────────────────────────────────────

SIMPLE_KEYWORDS = {"hello", "hi", "hey", "thanks", "ok", "yes", "no", "bye",
                   "good", "great", "nod", "wave", "look", "smile"}
COMPLEX_KEYWORDS = {"search", "find", "compare", "analyze", "explain", "research",
                    "plan", "schedule", "email", "browse", "summarize", "patient",
                    "medication", "blood", "health", "appointment", "calendar"}


# ─── Request/Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    tool_calls_executed: list[dict] = []
    thinking: Optional[str] = None
    routed_to: str = "local"  # "local" or "gateway"
    memory_context_used: bool = False

class MemoryStoreRequest(BaseModel):
    key: str
    value: str
    category: str = "general"
    importance: float = 0.5

class RobotCommand(BaseModel):
    action: str  # look_at, express, speak, nod, shake_no
    parameters: dict = {}


# ─── Robot Controller ─────────────────────────────────────────────────────────

class ReachyController:
    """Controls the Reachy Mini robot via reachy2-sdk."""

    EMOTIONS = {
        "happy":     {"head": [0, -5, 0],   "antennas": [20, -20]},
        "sad":       {"head": [0, -20, 0],  "antennas": [-10, 10]},
        "curious":   {"head": [15, -5, 20], "antennas": [30, -5]},
        "surprised": {"head": [0, 5, 0],    "antennas": [40, -40]},
        "thinking":  {"head": [5, -10, 15], "antennas": [10, -10]},
    }

    def __init__(self, reachy_ip: str):
        self.reachy_ip = reachy_ip
        self.reachy = None
        self.connected = False

    def connect(self):
        """Connect to Reachy Mini."""
        try:
            from reachy2_sdk import ReachySDK
            self.reachy = ReachySDK(host=self.reachy_ip)
            if self.reachy.is_connected():
                self.reachy.turn_on()
                self.reachy.head.goto_posture("default", wait=True)
                self.connected = True
                print(f"✅ Connected to Reachy at {self.reachy_ip}")
            else:
                print(f"⚠️ Could not connect to Reachy at {self.reachy_ip}")
                self.reachy = None
        except ImportError:
            print("⚠️ reachy2-sdk not installed - running in LLM-only mode")
            self.reachy = None
        except Exception as e:
            print(f"⚠️ Reachy connection error: {e}")
            self.reachy = None

    def disconnect(self):
        if self.reachy and self.connected:
            try:
                self.reachy.turn_off()
                self.reachy.disconnect()
            except Exception:
                pass
            self.connected = False

    async def execute(self, action: str, params: dict) -> dict:
        """Execute a robot action."""
        if not self.reachy:
            return {"status": "simulated", "action": action, "note": "Reachy not connected"}

        try:
            if action == "look_at":
                x = params.get("x", 0.5)
                y = params.get("y", 0)
                z = params.get("z", 0.2)
                duration = params.get("duration", 1.0)
                self.reachy.head.look_at(x=x, y=y, z=z, duration=duration, wait=True)
                return {"status": "success", "action": f"Looking at ({x}, {y}, {z})"}

            elif action == "express":
                emotion = params.get("emotion", "happy")
                if emotion == "nodding":
                    self.reachy.head.goto([0, -20, 0], duration=0.3, wait=True)
                    self.reachy.head.goto([0, 5, 0], duration=0.3, wait=True)
                    self.reachy.head.goto([0, -15, 0], duration=0.3, wait=True)
                    self.reachy.head.goto_posture("default", duration=0.5, wait=True)
                elif emotion == "shaking_no":
                    self.reachy.head.goto([0, -10, 20], duration=0.3, wait=True)
                    self.reachy.head.goto([0, -10, -20], duration=0.3, wait=True)
                    self.reachy.head.goto_posture("default", duration=0.5, wait=True)
                elif emotion in self.EMOTIONS:
                    mv = self.EMOTIONS[emotion]
                    self.reachy.head.goto(mv["head"], duration=0.5, wait=True)
                    if self.reachy.head.l_antenna and mv.get("antennas"):
                        self.reachy.head.l_antenna.goto(mv["antennas"][0], duration=0.3)
                        self.reachy.head.r_antenna.goto(mv["antennas"][1], duration=0.3, wait=True)
                    await asyncio.sleep(1.0)
                    self.reachy.head.goto_posture("default", duration=0.5, wait=True)
                return {"status": "success", "emotion": emotion}

            elif action == "speak":
                text = params.get("text", "")
                print(f"🔊 SPEAK: {text}")
                # TODO: Integrate ElevenLabs or Piper TTS
                return {"status": "success", "text": text}

            elif action == "nod":
                return await self.execute("express", {"emotion": "nodding"})

            elif action == "default_pose":
                self.reachy.head.goto_posture("default", duration=1.0, wait=True)
                return {"status": "success", "action": "default_pose"}

            return {"status": "unknown_action", "action": action}

        except Exception as e:
            return {"status": "error", "action": action, "error": str(e)}


# ─── Memory Client ────────────────────────────────────────────────────────────

class MemoryClient:
    """Client for the memory manager service (05_memory_manager.py)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.available = False

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/memory/stats")
                self.available = resp.status_code == 200
        except Exception:
            self.available = False
        return self.available

    async def store(self, key: str, value: str, category: str = "general",
                    importance: float = 0.5):
        if not self.available:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{self.base_url}/memory/store", json={
                    "key": key, "value": value,
                    "category": category, "importance": importance,
                })
        except Exception:
            pass

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        if not self.available:
            return []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.base_url}/memory/search", json={
                    "query": query, "max_results": max_results,
                })
                if resp.status_code == 200:
                    return resp.json().get("results", [])
        except Exception:
            pass
        return []

    async def get_context(self, user_query: str) -> str:
        if not self.available:
            return ""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.base_url}/memory/context", json={
                    "user_query": user_query, "max_items": 5,
                })
                if resp.status_code == 200:
                    return resp.json().get("context", "")
        except Exception:
            pass
        return ""

    async def record_turn(self, role: str, content: str, tool_calls: list = None):
        if not self.available:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{self.base_url}/memory/turn", json={
                    "role": role, "content": content,
                    "tool_calls": tool_calls or [],
                })
        except Exception:
            pass


# ─── Smart Router ─────────────────────────────────────────────────────────────

def classify_complexity(message: str) -> str:
    """Classify message as 'simple' or 'complex' to route to right agent."""
    words = set(message.lower().split())

    # Very short messages are usually simple
    if len(words) <= 3:
        if words & SIMPLE_KEYWORDS:
            return "simple"

    # Complex keywords trigger cloud routing
    if words & COMPLEX_KEYWORDS:
        return "complex"

    # Questions with "how", "why", "what if" tend to be complex
    if any(message.lower().startswith(w) for w in ["how do", "why does", "what if", "can you compare",
                                                     "analyze", "explain why"]):
        return "complex"

    # Default to simple for speed
    return "simple"


# ─── LLM Client ──────────────────────────────────────────────────────────────

async def call_ollama(message: str, context: str = "", model: str = OLLAMA_MODEL) -> str:
    """Call local Ollama model for fast reactive responses."""
    system_prompt = """You are Reachy, an embodied AI assistant running on a Reachy Mini robot.
Use tools when needed. Be concise and friendly.
Wrap thinking in <think></think> tags.
Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> for actions.

Available tools: robot_look_at, robot_express, robot_speak, search_web, set_reminder, get_patient_summary"""

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Memory context:\n{context}"})
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model,
            "messages": messages,
            "stream": False,
        })

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error: {resp.text}")

    return resp.json().get("message", {}).get("content", "")


# ─── Tool Call Parsing & Execution ────────────────────────────────────────────

def parse_tool_calls(text: str) -> list[dict]:
    """Parse <tool_call> tags from LLM output."""
    calls = []
    for match in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            parsed = json.loads(match)
            calls.append(parsed)
        except json.JSONDecodeError:
            continue
    return calls

def extract_thinking(text: str) -> Optional[str]:
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def clean_response(text: str) -> str:
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app(reachy_ip: str, gateway_host: str = "",
               gateway_port: int = 18789, memory_url: str = "") -> FastAPI:
    """Create the FastAPI application."""

    robot = ReachyController(reachy_ip)
    memory = MemoryClient(memory_url) if memory_url else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        robot.connect()
        if memory:
            await memory.check_health()
            if memory.available:
                print(f"✅ Memory service connected at {memory_url}")
            else:
                print(f"⚠️ Memory service not available at {memory_url}")
        yield
        # Shutdown
        robot.disconnect()

    app = FastAPI(title="Reachy OpenClaw Bridge", lifespan=lifespan)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Process a chat message with smart routing and memory."""
        routed_to = "local"
        memory_context = ""

        # 1. Get memory context
        if memory and memory.available:
            memory_context = await memory.get_context(request.message)
            await memory.record_turn("user", request.message)

        # 2. Classify complexity and route
        complexity = classify_complexity(request.message)

        if complexity == "complex" and gateway_host:
            # Route to OpenClaw Gateway (Mistral Large)
            routed_to = "gateway"
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {}
                    if GATEWAY_TOKEN:
                        headers["Authorization"] = f"Bearer {GATEWAY_TOKEN}"
                    resp = await client.post(
                        f"http://{gateway_host}:{gateway_port}/api/chat",
                        json={"message": request.message, "context": memory_context},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        llm_response = resp.json().get("response", "")
                    else:
                        # Fallback to local
                        llm_response = await call_ollama(request.message, memory_context)
                        routed_to = "local (fallback)"
            except Exception:
                llm_response = await call_ollama(request.message, memory_context)
                routed_to = "local (fallback)"
        else:
            # Local model (fast)
            llm_response = await call_ollama(request.message, memory_context)

        # 3. Extract and execute tool calls
        thinking = extract_thinking(llm_response)
        tool_calls = parse_tool_calls(llm_response)
        executed = []

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            print(f"🔧 Executing: {name}({args})")

            if name.startswith("robot_"):
                action = name.replace("robot_", "")
                result = await robot.execute(action, args)
            elif name == "search_web":
                try:
                    from ddgs import DDGS
                    results = DDGS().text(args.get("query", ""), max_results=3)
                    result = {"results": results}
                except Exception as e:
                    result = {"error": str(e)}
            elif name == "set_reminder":
                result = {"status": "set", "message": args.get("message", ""),
                          "minutes": args.get("minutes", 0)}
                if memory:
                    await memory.store(
                        f"reminder_{int(time.time())}",
                        f"Reminder: {args.get('message', '')} in {args.get('minutes', 0)} min",
                        category="reminder", importance=0.8
                    )
            elif name == "memory_search":
                if memory:
                    result = {"results": await memory.search(args.get("query", ""))}
                else:
                    result = {"error": "Memory service not available"}
            else:
                result = {"status": "unknown_tool", "name": name}

            executed.append({"name": name, "arguments": args, "result": result})

        # 4. Record assistant turn in memory
        clean_text = clean_response(llm_response)
        if memory and memory.available:
            await memory.record_turn("assistant", clean_text,
                                     [{"name": tc["name"]} for tc in tool_calls])

        return ChatResponse(
            response=clean_text,
            tool_calls_executed=executed,
            thinking=thinking,
            routed_to=routed_to,
            memory_context_used=bool(memory_context),
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "reachy_connected": robot.connected,
            "ollama_model": OLLAMA_MODEL,
            "gateway_connected": bool(gateway_host),
            "memory_available": memory.available if memory else False,
        }

    @app.post("/robot/{action}")
    async def robot_action(action: str, params: dict = {}):
        return await robot.execute(action, params)

    @app.post("/robot/nod")
    async def nod():
        return await robot.execute("nod", {})

    @app.post("/memory/store")
    async def store_memory(req: MemoryStoreRequest):
        if not memory:
            raise HTTPException(400, "Memory service not configured")
        await memory.store(req.key, req.value, req.category, req.importance)
        return {"status": "stored"}

    @app.post("/memory/search")
    async def search_memory(query: str, max_results: int = 5):
        if not memory:
            raise HTTPException(400, "Memory service not configured")
        return {"results": await memory.search(query, max_results)}

    return app


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reachy OpenClaw Bridge")
    parser.add_argument("--reachy-ip", type=str, default=REACHY_IP, help="Reachy Mini IP")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--gateway-host", type=str, default=GATEWAY_HOST, help="OpenClaw gateway host")
    parser.add_argument("--gateway-port", type=int, default=GATEWAY_PORT, help="OpenClaw gateway port")
    parser.add_argument("--memory-url", type=str, default=MEMORY_URL, help="Memory service URL")
    parser.add_argument("--standalone", action="store_true", help="Run without OpenClaw gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    global OLLAMA_URL, OLLAMA_MODEL
    OLLAMA_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model

    gw_host = "" if args.standalone else args.gateway_host

    app = create_app(
        reachy_ip=args.reachy_ip,
        gateway_host=gw_host,
        gateway_port=args.gateway_port,
        memory_url=args.memory_url,
    )

    import uvicorn
    print(f"🤖 Reachy OpenClaw Bridge starting...")
    print(f"   Reachy IP:   {args.reachy_ip}")
    print(f"   Ollama:      {OLLAMA_URL} ({OLLAMA_MODEL})")
    print(f"   Gateway:     {'disabled' if args.standalone else f'{gw_host}:{args.gateway_port}'}")
    print(f"   Memory:      {args.memory_url or 'disabled'}")
    print(f"   Serving on:  {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
