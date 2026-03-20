"""Allow running as `python -m claude_memory.server`."""
import asyncio
from .main import main

asyncio.run(main())
