#!/usr/bin/env python3
"""UserPromptSubmit hook: search memory for relevant context based on user's prompt."""

import json
import os
import sys
from pathlib import Path

# Allow importing the server package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from claude_memory.server.storage.session_db import SessionDB


def main():
    # Read the user prompt from stdin (hook provides it as JSON)
    try:
        hook_input = json.load(sys.stdin)
        prompt = hook_input.get("prompt", "")
    except (json.JSONDecodeError, TypeError):
        prompt = ""

    if not prompt or len(prompt) < 10:
        return

    namespace = Path(os.environ.get("CLAUDE_CWD", os.getcwd())).name.lower().replace(" ", "-")
    db = SessionDB()

    # Quick FTS search for relevant memories
    results = db.fts_search(query=prompt[:200], namespace=namespace, limit=5)

    if results:
        print("<relevant-memories>")
        for r in results:
            tags = ", ".join(r.get("tags", []))
            tag_str = f"[{tags}]" if tags else ""
            cat = r.get("category", "").upper()
            print(f"- [{cat}]{tag_str} {r['content']}")
        print("</relevant-memories>")

    db.close()


if __name__ == "__main__":
    main()
