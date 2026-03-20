"""Predictive pre-fetch: load relevant memories into hot cache based on recent file activity."""

import subprocess
from pathlib import Path

FILE_SIGNALS = {
    ("dockerfile", "docker-compose", ".dockerignore"):
        ["docker", "containers", "deployment", "images"],
    ("test", "spec", "pytest", "jest", "vitest"):
        ["testing", "fixtures", "mocks", "assertions"],
    (".env", "config", "settings"):
        ["config", "environment", "variables"],
    ("api", "routes", "endpoints", "handlers"):
        ["api", "endpoints", "auth", "requests"],
    ("migrations", "models", "schema", "seeds"):
        ["database", "migrations", "schema", "queries"],
    (".github/workflows", "gitlab-ci", "jenkins"):
        ["ci", "cd", "pipeline", "automation"],
    ("terraform", "pulumi", "kubernetes", "k8s"):
        ["infrastructure", "iac", "deployment", "cloud"],
    ("nginx", "caddy", "proxy"):
        ["proxy", "nginx", "ssl", "routing"],
    ("package.json", "go.mod", "requirements.txt", "cargo.toml"):
        ["dependencies", "packages", "versions"],
}


def analyze_recent_activity(cwd: str, minutes: int = 60) -> list[str]:
    """Analyze recently modified files to predict relevant memory tags."""
    try:
        result = subprocess.run(
            ["find", cwd, "-mmin", f"-{minutes}", "-type", "f",
             "-not", "-path", "*/.*", "-not", "-path", "*/node_modules/*",
             "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/vendor/*"],
            capture_output=True, text=True, timeout=5,
        )
        recent_files = result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    predicted_tags = set()

    for filepath in recent_files:
        if not filepath:
            continue
        filepath_lower = filepath.lower()

        for patterns, tags in FILE_SIGNALS.items():
            for pattern in patterns:
                if pattern in filepath_lower:
                    predicted_tags.update(tags)
                    break

    return list(predicted_tags)


def prefetch_context(
    namespace: str,
    cwd: str,
    session_db,
    hot_cache,
) -> dict:
    """Pre-load relevant memories into hot cache based on predicted activity."""
    predicted_tags = analyze_recent_activity(cwd)

    if not predicted_tags:
        return {"prefetched": 0, "tags": []}

    query = " ".join(predicted_tags)
    memories = session_db.fts_search(query=query, namespace=namespace, limit=20)

    for memory in memories:
        hot_cache.put(memory["id"], memory)

    return {"prefetched": len(memories), "tags": predicted_tags}
