"""engram — agent-loop layer for persistent memory, built on top of vstash.

See ``CONSTITUTION.md`` for what engram is, what it isn't, and the principles
that should outlive any specific implementation.
"""

from engram.memory import Memory

__version__ = "0.1.0"
__all__ = ["Memory", "__version__"]
