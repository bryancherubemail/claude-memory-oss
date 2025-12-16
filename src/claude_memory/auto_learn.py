#!/usr/bin/env python3
"""
Automatic Learning Extraction using Ollama

Analyzes Claude's conversation logs and extracts learnings automatically.
Triggered via Claude Code hooks (PreCompact, SessionEnd).
"""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration - override with environment variables
DATA_DIR = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
MEMORY_DB = DATA_DIR / "data" / "memory.db"
SESSIONS_DIR = DATA_DIR / "sessions"
QUEUE_FILE = DATA_DIR / "data" / ".extraction_queue.json"
PROCESSED_TRACKER = DATA_DIR / "data" / ".processed_messages.json"

# Claude's conversation logs location
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Ollama model - smaller models are faster, larger models extract better
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

# Timeout tracking
_start_time: Optional[float] = None
_max_time: Optional[int] = None

# Category mapping
CATEGORY_MAP = {
    "DECISION": "decisions",
    "CONVENTION": "conventions",
    "TECH_STACK": "tech_stack",
    "TECH": "tech_stack",
    "GOTCHA": "gotchas",
    "BUG": "gotchas",
    "ISSUE": "gotchas",
    "SOLUTION": "solutions",
    "FIX": "solutions",
}

EXTRACTION_PROMPT = """You extract HIGH-VALUE technical learnings from development conversations.

OUTPUT FORMAT - Each learning on its own line, starting with the category prefix:
CATEGORY: learning content

Categories: DECISION, CONVENTION, TECH_STACK, GOTCHA, SOLUTION

RULES:
- Maximum 5 learnings, only high-value items worth remembering months later
- Include project name and specific details (versions, file paths, reasoning)
- REJECT: obvious facts, generic advice, trivial commands
- If nothing valuable, output "NONE"

EXAMPLES:
DECISION: Project uses PostgreSQL instead of MySQL for better JSON support
GOTCHA: GORM .Find() without .Limit() causes memory exhaustion on large datasets
CONVENTION: All API endpoints must validate input before processing

CONVERSATION:
{conversation_text}

Output learnings (one per line, starting with category):"""

SESSION_STATE_PROMPT = """Summarize the current working state for context preservation after memory compaction.

OUTPUT FORMAT (use exactly these headers):
## Current Focus
[What is being actively worked on - 1-2 sentences]

## Key Decisions This Session
[Bullet list of decisions made, with brief rationale]

## Important Context
[Critical facts, constraints, or requirements established]

## Working State
[Files being edited, services involved, any blockers]

RULES:
1. Be CONCISE - must fit in limited context
2. Focus on ACTIONABLE context - what's needed to continue work
3. Skip greetings, chitchat, completed items
4. If exploratory with no clear state, say "Exploratory session - no active working state"

CONVERSATION:
{conversation_text}

SESSION STATE:"""


def check_timeout() -> bool:
    if _start_time is None or _max_time is None:
        return False
    return time.time() - _start_time >= _max_time


def time_remaining() -> Optional[int]:
    if _start_time is None or _max_time is None:
        return None
    return max(0, int(_max_time - (time.time() - _start_time)))


def get_project_context() -> str:
    """Detect current project from git or directory name."""
    try:
        cwd = Path.cwd()
        if cwd == Path.home() / "Claude":
            return "global"

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=1, cwd=cwd
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).name

        return cwd.name if cwd.name else "global"
    except Exception:
        return "global"


def ensure_ollama_running() -> bool:
    """Check if Ollama is available."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def call_ollama(prompt: str, timeout: int = 60) -> str:
    """Call Ollama to process text."""
    try:
        if check_timeout():
            return ""

        if not ensure_ollama_running():
            print("Ollama not available", file=sys.stderr)
            return ""

        remaining = time_remaining()
        if remaining is not None:
            timeout = min(timeout, remaining - 5)
            if timeout <= 0:
                return ""

        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return ""

        output = result.stdout.decode('utf-8', errors='ignore')

        # Strip ANSI codes
        output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)

        # Strip thinking tokens
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)

        return output.strip()

    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def parse_llm_output(output: str) -> Dict[str, List[str]]:
    """Parse LLM output into categorized learnings."""
    learnings = {cat: [] for cat in ["decisions", "conventions", "tech_stack", "gotchas", "solutions"]}

    if not output or output.upper().strip() == "NONE":
        return learnings

    for line in output.split('\n'):
        line = line.strip().lstrip('-•* ').strip()
        if not line or len(line) < 20:
            continue

        for prefix, category in CATEGORY_MAP.items():
            if line.startswith(f"{prefix}:"):
                content = line[len(prefix)+1:].strip()
                if len(content) >= 20:
                    learnings[category].append(content)
                break

    return learnings


def store_learning(content: str, category: str, project: str, context: str = "") -> bool:
    """Store a learning in SQLite."""
    try:
        MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(MEMORY_DB, timeout=10.0)
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                relevance_score REAL DEFAULT 1.0,
                project TEXT DEFAULT 'global'
            )
        """)

        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()

        cursor.execute(
            "INSERT INTO memories (id, timestamp, type, content, context, relevance_score, project) VALUES (?, ?, ?, ?, ?, 1.0, ?)",
            (memory_id, datetime.now().isoformat(), category, content, context, project)
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing: {e}", file=sys.stderr)
        return False


def load_processed_tracker() -> set:
    try:
        if PROCESSED_TRACKER.exists():
            with open(PROCESSED_TRACKER) as f:
                return set(json.load(f).get("processed_ids", []))
    except Exception:
        pass
    return set()


def save_processed_tracker(processed_ids: set):
    try:
        PROCESSED_TRACKER.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_TRACKER, "w") as f:
            json.dump({"processed_ids": list(processed_ids)}, f)
    except Exception:
        pass


def clear_processed_tracker():
    try:
        if PROCESSED_TRACKER.exists():
            PROCESSED_TRACKER.unlink()
    except Exception:
        pass


def save_to_queue(project: str, messages: List[Dict], reason: str):
    """Queue messages for later processing."""
    try:
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

        queue = []
        if QUEUE_FILE.exists():
            try:
                with open(QUEUE_FILE) as f:
                    queue = json.load(f)
            except Exception:
                pass

        queue.append({
            "project": project,
            "messages": [{"content": m["content"], "_msg_id": m.get("_msg_id")} for m in messages],
            "reason": reason,
            "queued_at": datetime.now().isoformat(),
        })

        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

        print(f"  Queued {len(messages)} messages ({reason})", file=sys.stderr)
    except Exception:
        pass


def find_session_file() -> Optional[Path]:
    """Find the most recent conversation file."""
    if not CLAUDE_PROJECTS_DIR.exists():
        return None

    session_files = []
    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if project_dir.is_dir():
            for f in project_dir.glob("*.jsonl"):
                if not f.name.startswith("agent-"):
                    session_files.append(f)

    if session_files:
        return max(session_files, key=lambda p: p.stat().st_mtime)
    return None


def extract_text_from_content(content_blocks: list) -> str:
    """Extract text from Claude's response content blocks."""
    parts = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)


def get_session_messages(only_new: bool = False) -> List[Dict]:
    """Get messages from the most recent session."""
    try:
        session_file = find_session_file()
        if not session_file:
            return []

        print(f"Reading: {session_file.name}", file=sys.stderr)

        messages = []
        processed_ids = load_processed_tracker() if only_new else set()

        with open(session_file) as f:
            for line_num, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    msg_data = entry.get("message", {})

                    if msg_data.get("role") == "assistant":
                        text = extract_text_from_content(msg_data.get("content", []))
                        if not text:
                            continue

                        session_id = entry.get("sessionId", "unknown")
                        msg_id = f"{session_id}:{line_num}"

                        if only_new and msg_id in processed_ids:
                            continue

                        messages.append({
                            "content": text,
                            "_msg_id": msg_id,
                        })
                except json.JSONDecodeError:
                    continue

        print(f"Found {len(messages)} messages", file=sys.stderr)
        return messages

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return []


def write_session_state(project: str, state_content: str):
    """Write session state for recovery after compaction."""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        session_file = SESSIONS_DIR / f"{project}.md"

        content = f"""# Session State: {project}
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{state_content}

---
*Auto-generated for compaction recovery*
"""
        session_file.write_text(content)
        print(f"  Session state saved", file=sys.stderr)
    except Exception as e:
        print(f"Error writing session state: {e}", file=sys.stderr)


def process_session(incremental: bool = False, final: bool = False):
    """Process session and extract learnings."""
    project = get_project_context()
    mode = "Incremental" if incremental else "Final" if final else "Full"
    print(f"Auto-learning ({mode}) for: {project}", file=sys.stderr)

    if check_timeout():
        print("Timed out, skipping", file=sys.stderr)
        return

    messages = get_session_messages(only_new=(incremental or final))
    if not messages:
        print("No messages to process", file=sys.stderr)
        if final:
            clear_processed_tracker()
        return

    print(f"Processing {len(messages)} messages with {OLLAMA_MODEL}...", file=sys.stderr)

    BATCH_SIZE = 5
    total_learnings = 0
    processed_ids = load_processed_tracker()
    timed_out = False

    for i in range(0, len(messages), BATCH_SIZE):
        if check_timeout():
            save_to_queue(project, messages[i:], "timeout")
            timed_out = True
            break

        batch = messages[i:i+BATCH_SIZE]
        combined = "\n\n---\n\n".join([m["content"] for m in batch])

        # Truncate if too long
        if len(combined) > 8000:
            combined = combined[:4000] + "\n\n[...]\n\n" + combined[-4000:]

        prompt = EXTRACTION_PROMPT.format(conversation_text=combined)
        output = call_ollama(prompt)

        if output:
            learnings = parse_llm_output(output)
            for category, items in learnings.items():
                for learning in items:
                    context = f"Auto-extracted via {OLLAMA_MODEL}"
                    if store_learning(learning, category, project, context):
                        total_learnings += 1
                        print(f"  + {category}: {learning[:60]}...", file=sys.stderr)

        for msg in batch:
            if msg.get("_msg_id"):
                processed_ids.add(msg["_msg_id"])

    if incremental:
        save_processed_tracker(processed_ids)

        # Extract session state for compaction recovery
        if not timed_out and not check_timeout():
            all_text = "\n\n---\n\n".join([m["content"] for m in messages])
            if len(all_text) > 12000:
                all_text = all_text[-12000:]

            prompt = SESSION_STATE_PROMPT.format(conversation_text=all_text)
            state = call_ollama(prompt)
            if state:
                write_session_state(project, state)

    elif final:
        clear_processed_tracker()

    print(f"Done: {total_learnings} learnings stored", file=sys.stderr)


def process_queue():
    """Process queued extractions from previous timeouts."""
    try:
        if not QUEUE_FILE.exists():
            return

        with open(QUEUE_FILE) as f:
            queue = json.load(f)

        if not queue:
            return

        print(f"Processing {len(queue)} queued items...", file=sys.stderr)

        for item in queue:
            if check_timeout():
                break

            project = item.get("project", "unknown")
            messages = item.get("messages", [])

            for i in range(0, len(messages), 5):
                batch = messages[i:i+5]
                combined = "\n\n---\n\n".join([m["content"] for m in batch])

                prompt = EXTRACTION_PROMPT.format(conversation_text=combined)
                output = call_ollama(prompt)

                if output:
                    for category, items in parse_llm_output(output).items():
                        for learning in items:
                            store_learning(learning, category, project, "From queue")

        QUEUE_FILE.unlink()
        print("Queue processed", file=sys.stderr)

    except Exception as e:
        print(f"Queue error: {e}", file=sys.stderr)


def main():
    global _start_time, _max_time
    import argparse

    parser = argparse.ArgumentParser(description="Auto-learning extraction via Ollama")
    parser.add_argument("--incremental", action="store_true", help="Process only new messages")
    parser.add_argument("--final", action="store_true", help="Final extraction, clear tracker")
    parser.add_argument("--process-queue", action="store_true", help="Process queued items")
    parser.add_argument("--max-time", type=int, help="Max seconds before queuing remainder")
    args = parser.parse_args()

    if args.max_time:
        _start_time = time.time()
        _max_time = args.max_time

    try:
        if args.process_queue:
            process_queue()
        else:
            process_session(incremental=args.incremental, final=args.final)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
