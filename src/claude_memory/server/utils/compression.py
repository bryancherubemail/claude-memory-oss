"""Token-efficient output formatting for memory results."""


def format_results_compact(results: list[dict], max_tokens: int = 500) -> str:
    """Format results as compact markdown, respecting token budget."""
    lines = []
    tokens_used = 0

    for r in results:
        # ~4 chars per token estimate
        content = r.get("content", "")
        category = r.get("category", "")
        tags = ", ".join(r.get("tags", []))
        age = r.get("created_at", "")[:10]

        line = f"- [{category}] {content}"
        if tags:
            line += f" ({tags})"
        if age:
            line += f" [{age}]"

        line_tokens = len(line) // 4
        if tokens_used + line_tokens > max_tokens:
            lines.append(f"... ({len(results) - len(lines)} more)")
            break

        lines.append(line)
        tokens_used += line_tokens

    return "\n".join(lines)


def format_session_context(
    open_threads: list[dict],
    recent_decisions: list[dict],
    active_facts: list[dict],
    namespace: str,
) -> str:
    """Format session context for injection after compaction."""
    sections = [f"## Memory Context ({namespace})\n"]

    if open_threads:
        sections.append("### Open Threads")
        for t in open_threads[:5]:
            sections.append(f"- {t['content']}")
        sections.append("")

    if recent_decisions:
        sections.append("### Recent Decisions")
        for d in recent_decisions[:5]:
            sections.append(f"- {d['content']}")
        sections.append("")

    if active_facts:
        sections.append("### Active Context")
        for f in active_facts[:5]:
            sections.append(f"- {f['content']}")

    return "\n".join(sections)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4
