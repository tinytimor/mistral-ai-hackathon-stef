#!/usr/bin/env python3
"""
05_memory_manager.py - Three-tier memory system for Reachy Copilot edge deployment.

Manages:
  L1: Working memory (in-context, current turn)
  L2: Session cache (Redis-like in-memory with TTL)
  L3: Long-term persistent memory (SQLite)

Designed to run on Orin Nano (8GB) with minimal overhead (~50MB RAM).

Usage:
    # As a standalone FastAPI service (recommended for edge):
    python scripts/05_memory_manager.py --serve --port 8100

    # As an importable module in the bridge server:
    from scripts.memory_manager import MemoryManager
    mem = MemoryManager(db_path="data/memory.db")
    mem.store("user_preference", "Prefers Celsius", ttl=None)  # L3 permanent
    results = mem.search("temperature preference")
"""

import argparse
import hashlib
import json
import os
import sqlite3
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memory.db")
L2_MAX_ITEMS = int(os.getenv("MEMORY_L2_MAX_ITEMS", "200"))
L2_DEFAULT_TTL = int(os.getenv("MEMORY_L2_TTL_SECONDS", "3600"))  # 1 hour
L1_MAX_TOKENS = int(os.getenv("MEMORY_L1_MAX_TOKENS", "2048"))
SUMMARY_INTERVAL = int(os.getenv("MEMORY_SUMMARY_INTERVAL", "20"))  # summarize every N turns


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory item across any tier."""
    key: str
    value: str
    category: str = "general"           # general, preference, health, conversation, tool_result
    importance: float = 0.5             # 0.0 = trivial, 1.0 = critical
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None         # None = permanent (L3), seconds for L2
    source: str = "user"                # user, system, agent, tool
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> dict:
        return asdict(self)


# ─── L2: Session Cache (In-Memory with LRU + TTL) ────────────────────────────

class SessionCache:
    """
    Fast in-memory cache for session-level data.
    LRU eviction + TTL expiration. Thread-safe.

    Memory footprint: ~50KB for 200 items (typical).
    """

    def __init__(self, max_items: int = L2_MAX_ITEMS, default_ttl: int = L2_DEFAULT_TTL):
        self.max_items = max_items
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[MemoryEntry]:
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry

    def put(self, key: str, value: str, category: str = "general",
            importance: float = 0.5, ttl: Optional[int] = None,
            source: str = "user", metadata: dict = None) -> MemoryEntry:
        with self._lock:
            entry = MemoryEntry(
                key=key,
                value=value,
                category=category,
                importance=importance,
                ttl=ttl or self.default_ttl,
                source=source,
                metadata=metadata or {},
            )
            self._cache[key] = entry
            self._cache.move_to_end(key)

            # Evict oldest if over capacity (but keep high-importance items)
            while len(self._cache) > self.max_items:
                # Find lowest importance item to evict
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)

            return entry

    def search(self, query: str, max_results: int = 5) -> list[MemoryEntry]:
        """Simple keyword search across cache values."""
        query_lower = query.lower()
        results = []
        with self._lock:
            for entry in self._cache.values():
                if entry.is_expired:
                    continue
                # Score by keyword match + importance + recency
                score = 0.0
                if query_lower in entry.value.lower():
                    score += 0.5
                if query_lower in entry.key.lower():
                    score += 0.3
                if query_lower in entry.category.lower():
                    score += 0.2
                score += entry.importance * 0.3
                score += max(0, 1.0 - entry.age_seconds / 3600) * 0.2  # recency bonus

                if score > 0.1:
                    results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:max_results]]

    def get_recent(self, n: int = 10, category: str = None) -> list[MemoryEntry]:
        """Get the N most recent non-expired entries."""
        with self._lock:
            entries = []
            for entry in reversed(self._cache.values()):
                if entry.is_expired:
                    continue
                if category and entry.category != category:
                    continue
                entries.append(entry)
                if len(entries) >= n:
                    break
            return entries

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired]
            for k in expired_keys:
                del self._cache[k]
            return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._cache)


# ─── L3: Long-Term Memory (SQLite) ───────────────────────────────────────────

class LongTermMemory:
    """
    Persistent memory store using SQLite.
    Optimized for low-memory environments (Orin Nano).

    Features:
    - Full-text search via FTS5
    - Category-based filtering
    - Importance-weighted retrieval
    - Automatic conversation summarization
    - Encrypted health data support (future)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Main memory table
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                timestamp REAL NOT NULL,
                source TEXT DEFAULT 'user',
                metadata TEXT DEFAULT '{}',
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        """)

        # Full-text search index
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                key, value, category,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Triggers to keep FTS in sync
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, key, value, category)
                VALUES (new.rowid, new.key, new.value, new.category);
            END
        """)

        c.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, value, category)
                VALUES ('delete', old.rowid, old.key, old.value, old.category);
            END
        """)

        c.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, value, category)
                VALUES ('delete', old.rowid, old.key, old.value, old.category);
                INSERT INTO memories_fts(rowid, key, value, category)
                VALUES (new.rowid, new.key, new.value, new.category);
            END
        """)

        # Conversation summaries table
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                summary TEXT NOT NULL,
                turn_start INTEGER,
                turn_end INTEGER,
                timestamp REAL NOT NULL,
                key_topics TEXT DEFAULT '[]',
                sentiment TEXT DEFAULT 'neutral'
            )
        """)

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_session ON conversation_summaries(session_id)")

        conn.commit()
        conn.close()

    def store(self, key: str, value: str, category: str = "general",
              importance: float = 0.5, source: str = "user",
              metadata: dict = None) -> str:
        """Store a memory permanently."""
        memory_id = hashlib.md5(f"{key}:{value}:{time.time()}".encode()).hexdigest()[:12]
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT OR REPLACE INTO memories (id, key, value, category, importance, timestamp, source, metadata, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, key, value, category, importance, time.time(), source,
              json.dumps(metadata or {}), time.time()))

        conn.commit()
        conn.close()
        return memory_id

    def search(self, query: str, max_results: int = 10,
               category: str = None, min_importance: float = 0.0) -> list[dict]:
        """Full-text search across long-term memories."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Use FTS5 for text search
        fts_query = query.replace('"', '""')  # Escape quotes

        if category:
            c.execute("""
                SELECT m.id, m.key, m.value, m.category, m.importance, m.timestamp,
                       m.source, m.metadata, m.access_count,
                       rank
                FROM memories_fts fts
                JOIN memories m ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ? AND m.category = ? AND m.importance >= ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, category, min_importance, max_results))
        else:
            c.execute("""
                SELECT m.id, m.key, m.value, m.category, m.importance, m.timestamp,
                       m.source, m.metadata, m.access_count,
                       rank
                FROM memories_fts fts
                JOIN memories m ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ? AND m.importance >= ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, min_importance, max_results))

        rows = c.fetchall()

        # Update access counts
        for row in rows:
            c.execute("UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                      (time.time(), row[0]))

        conn.commit()
        conn.close()

        return [
            {
                "id": row[0], "key": row[1], "value": row[2],
                "category": row[3], "importance": row[4],
                "timestamp": row[5], "source": row[6],
                "metadata": json.loads(row[7]), "access_count": row[8],
                "relevance_score": -row[9],  # FTS rank is negative (lower = better)
            }
            for row in rows
        ]

    def get_by_category(self, category: str, limit: int = 20) -> list[dict]:
        """Get memories by category, ordered by importance then recency."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            SELECT id, key, value, category, importance, timestamp, source, metadata
            FROM memories
            WHERE category = ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, (category, limit))

        rows = c.fetchall()
        conn.close()

        return [
            {"id": r[0], "key": r[1], "value": r[2], "category": r[3],
             "importance": r[4], "timestamp": r[5], "source": r[6],
             "metadata": json.loads(r[7])}
            for r in rows
        ]

    def store_conversation_summary(self, session_id: str, summary: str,
                                    turn_start: int, turn_end: int,
                                    key_topics: list = None,
                                    sentiment: str = "neutral") -> str:
        """Store a conversation summary for long-term context."""
        summary_id = hashlib.md5(f"{session_id}:{turn_start}:{turn_end}".encode()).hexdigest()[:12]
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT OR REPLACE INTO conversation_summaries
            (id, session_id, summary, turn_start, turn_end, timestamp, key_topics, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (summary_id, session_id, summary, turn_start, turn_end,
              time.time(), json.dumps(key_topics or []), sentiment))

        conn.commit()
        conn.close()
        return summary_id

    def get_conversation_context(self, session_id: str = None,
                                  max_summaries: int = 5) -> list[dict]:
        """Get recent conversation summaries for context injection."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        if session_id:
            c.execute("""
                SELECT id, session_id, summary, turn_start, turn_end,
                       timestamp, key_topics, sentiment
                FROM conversation_summaries
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, max_summaries))
        else:
            c.execute("""
                SELECT id, session_id, summary, turn_start, turn_end,
                       timestamp, key_topics, sentiment
                FROM conversation_summaries
                ORDER BY timestamp DESC
                LIMIT ?
            """, (max_summaries,))

        rows = c.fetchall()
        conn.close()

        return [
            {"id": r[0], "session_id": r[1], "summary": r[2],
             "turn_start": r[3], "turn_end": r[4], "timestamp": r[5],
             "key_topics": json.loads(r[6]), "sentiment": r[7]}
            for r in rows
        ]

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*), AVG(importance) FROM memories")
        total, avg_importance = c.fetchone()

        c.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
        categories = dict(c.fetchall())

        c.execute("SELECT COUNT(*) FROM conversation_summaries")
        summaries = c.fetchone()[0]

        # DB file size
        db_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024) if Path(self.db_path).exists() else 0

        conn.close()

        return {
            "total_memories": total or 0,
            "avg_importance": round(avg_importance or 0, 3),
            "categories": categories,
            "conversation_summaries": summaries,
            "db_size_mb": round(db_size_mb, 2),
        }


# ─── Unified Memory Manager ──────────────────────────────────────────────────

class MemoryManager:
    """
    Unified interface across all three memory tiers.

    Automatically promotes important session memories to long-term storage.
    Provides context injection for LLM prompts.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH,
                 l2_max_items: int = L2_MAX_ITEMS,
                 l2_ttl: int = L2_DEFAULT_TTL):
        self.l2_cache = SessionCache(max_items=l2_max_items, default_ttl=l2_ttl)
        self.l3_store = LongTermMemory(db_path=db_path)
        self._conversation_turns = []
        self._turn_counter = 0
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    def store(self, key: str, value: str, category: str = "general",
              importance: float = 0.5, ttl: Optional[int] = None,
              source: str = "user", metadata: dict = None):
        """
        Store a memory. Automatically routes to appropriate tier:
        - importance >= 0.7 or ttl=None → L3 (permanent)
        - otherwise → L2 (session cache)
        """
        # Always cache in L2 for fast access
        self.l2_cache.put(key, value, category, importance, ttl, source, metadata)

        # Promote to L3 if important or explicitly permanent
        if importance >= 0.7 or ttl is None:
            self.l3_store.store(key, value, category, importance, source, metadata)

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Search across all memory tiers. L2 first (fast), then L3."""
        results = []

        # Search L2 (in-memory, instant)
        l2_results = self.l2_cache.search(query, max_results=max_results)
        for entry in l2_results:
            results.append({
                "tier": "L2",
                "key": entry.key,
                "value": entry.value,
                "category": entry.category,
                "importance": entry.importance,
                "age_seconds": entry.age_seconds,
            })

        # Search L3 (SQLite FTS, ~1-5ms)
        l3_results = self.l3_store.search(query, max_results=max_results)
        for entry in l3_results:
            # Deduplicate with L2
            if not any(r["key"] == entry["key"] and r["value"] == entry["value"] for r in results):
                results.append({
                    "tier": "L3",
                    **entry,
                })

        # Sort by importance, then recency
        results.sort(key=lambda x: (x.get("importance", 0), -x.get("age_seconds", float("inf"))), reverse=True)
        return results[:max_results]

    def record_turn(self, role: str, content: str, tool_calls: list = None):
        """Record a conversation turn for context tracking and summarization."""
        self._turn_counter += 1
        turn = {
            "turn": self._turn_counter,
            "role": role,
            "content": content[:500],  # Truncate for memory efficiency
            "tool_calls": tool_calls or [],
            "timestamp": time.time(),
        }
        self._conversation_turns.append(turn)

        # Auto-summarize every N turns
        if len(self._conversation_turns) >= SUMMARY_INTERVAL:
            self._auto_summarize()

    def _auto_summarize(self):
        """Create a summary of recent conversation turns and store in L3."""
        if not self._conversation_turns:
            return

        # Simple extractive summary (for edge, avoid calling LLM for summaries)
        user_messages = [t["content"] for t in self._conversation_turns if t["role"] == "user"]
        assistant_messages = [t["content"] for t in self._conversation_turns if t["role"] == "assistant"]
        tools_used = set()
        for t in self._conversation_turns:
            for tc in t.get("tool_calls", []):
                tools_used.add(tc.get("name", "unknown"))

        summary = f"User discussed: {'; '.join(user_messages[:5])}. "
        if tools_used:
            summary += f"Tools used: {', '.join(tools_used)}. "
        summary += f"({len(self._conversation_turns)} turns)"

        # Extract key topics (simple keyword extraction)
        all_text = " ".join(user_messages).lower()
        topic_keywords = ["health", "weather", "medication", "reminder", "search",
                          "email", "calendar", "exercise", "food", "appointment"]
        key_topics = [kw for kw in topic_keywords if kw in all_text]

        turn_start = self._conversation_turns[0]["turn"]
        turn_end = self._conversation_turns[-1]["turn"]

        self.l3_store.store_conversation_summary(
            session_id=self._session_id,
            summary=summary,
            turn_start=turn_start,
            turn_end=turn_end,
            key_topics=key_topics,
        )

        # Keep last 5 turns as overlap, clear the rest
        self._conversation_turns = self._conversation_turns[-5:]

    def get_context_for_prompt(self, user_query: str, max_context_items: int = 5) -> str:
        """
        Build a context string to inject into the LLM prompt.
        Combines relevant memories from L2 and L3.
        """
        parts = []

        # 1. Recent conversation summaries
        summaries = self.l3_store.get_conversation_context(
            session_id=self._session_id, max_summaries=3
        )
        if summaries:
            parts.append("## Previous Conversations")
            for s in summaries:
                parts.append(f"- {s['summary']}")

        # 2. Relevant memories
        memories = self.search(user_query, max_results=max_context_items)
        if memories:
            parts.append("\n## Relevant Memories")
            for m in memories:
                parts.append(f"- [{m['category']}] {m['value']}")

        # 3. User preferences
        prefs = self.l3_store.get_by_category("preference", limit=5)
        if prefs:
            parts.append("\n## User Preferences")
            for p in prefs:
                parts.append(f"- {p['value']}")

        return "\n".join(parts) if parts else ""

    def get_stats(self) -> dict:
        """Get unified memory statistics."""
        l3_stats = self.l3_store.get_stats()
        return {
            "session_id": self._session_id,
            "l1_current_turns": len(self._conversation_turns),
            "l2_cached_items": self.l2_cache.size,
            "l3_total_memories": l3_stats["total_memories"],
            "l3_categories": l3_stats["categories"],
            "l3_summaries": l3_stats["conversation_summaries"],
            "l3_db_size_mb": l3_stats["db_size_mb"],
            "turn_counter": self._turn_counter,
        }


# ─── FastAPI Server (standalone mode) ────────────────────────────────────────

def create_app(db_path: str = DEFAULT_DB_PATH) -> "FastAPI":
    """Create a FastAPI app for the memory service."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Reachy Memory Service", version="1.0")
    manager = MemoryManager(db_path=db_path)

    class StoreRequest(BaseModel):
        key: str
        value: str
        category: str = "general"
        importance: float = 0.5
        ttl: Optional[int] = None
        source: str = "user"
        metadata: dict = {}

    class SearchRequest(BaseModel):
        query: str
        max_results: int = 10

    class TurnRequest(BaseModel):
        role: str
        content: str
        tool_calls: list = []

    class ContextRequest(BaseModel):
        user_query: str
        max_items: int = 5

    @app.post("/memory/store")
    async def store_memory(req: StoreRequest):
        manager.store(req.key, req.value, req.category,
                      req.importance, req.ttl, req.source, req.metadata)
        return {"status": "stored", "key": req.key}

    @app.post("/memory/search")
    async def search_memory(req: SearchRequest):
        results = manager.search(req.query, req.max_results)
        return {"results": results, "count": len(results)}

    @app.post("/memory/turn")
    async def record_turn(req: TurnRequest):
        manager.record_turn(req.role, req.content, req.tool_calls)
        return {"status": "recorded", "turn": manager._turn_counter}

    @app.post("/memory/context")
    async def get_context(req: ContextRequest):
        context = manager.get_context_for_prompt(req.user_query, req.max_items)
        return {"context": context}

    @app.get("/memory/stats")
    async def memory_stats():
        return manager.get_stats()

    @app.delete("/memory/cache")
    async def clear_cache():
        removed = manager.l2_cache.clear_expired()
        return {"expired_removed": removed, "remaining": manager.l2_cache.size}

    return app


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reachy Copilot Memory Manager")
    parser.add_argument("--serve", action="store_true", help="Run as FastAPI server")
    parser.add_argument("--port", type=int, default=8100, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument("--test", action="store_true", help="Run a quick self-test")
    args = parser.parse_args()

    if args.test:
        print("🧪 Running memory manager self-test...")
        mem = MemoryManager(db_path=":memory:")  # In-memory for testing

        # Store some test data
        mem.store("user_name", "Stefan", "preference", 0.9, source="onboarding")
        mem.store("location", "Washington DC", "preference", 0.8, source="user")
        mem.store("temperature_pref", "Prefers Celsius", "preference", 0.7)
        mem.store("morning_routine", "Coffee first, then emails", "preference", 0.6)
        mem.store("bp_reading", "120/80 on 2026-02-28", "health", 0.9, source="tool")

        # Record some turns
        mem.record_turn("user", "What's the weather in NYC?")
        mem.record_turn("assistant", "Let me search for that.",
                        [{"name": "search_web", "arguments": {"query": "NYC weather"}}])
        mem.record_turn("user", "Thanks! Also remind me about my pills in 30 minutes.")
        mem.record_turn("assistant", "I've set a reminder for your medication.",
                        [{"name": "set_reminder", "arguments": {"message": "pills", "minutes": 30}}])

        # Search
        print("\n🔍 Search for 'temperature':")
        for r in mem.search("temperature"):
            print(f"   [{r['tier']}] {r['key']}: {r['value']}")

        print("\n🔍 Search for 'health':")
        for r in mem.search("health"):
            print(f"   [{r['tier']}] {r.get('key', 'N/A')}: {r.get('value', 'N/A')}")

        # Context injection
        print("\n📝 Context for 'What temperature is it?':")
        ctx = mem.get_context_for_prompt("What temperature is it?")
        print(ctx or "   (no relevant context)")

        # Stats
        print("\n📊 Memory stats:")
        stats = mem.get_stats()
        for k, v in stats.items():
            print(f"   {k}: {v}")

        print("\n✅ Self-test passed!")
        return

    if args.serve:
        import uvicorn
        app = create_app(db_path=args.db)
        print(f"🧠 Memory service starting on {args.host}:{args.port}")
        print(f"   DB: {args.db}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
