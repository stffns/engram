"""engram — agent-loop layer for persistent memory, built on top of vstash.

See ``CONSTITUTION.md`` for what engram is, what it isn't, and the principles
that should outlive any specific implementation.
"""

from engram.consolidation import ConsolidationResult, Fact
from engram.memory import Memory, RememberResult
from engram.policies import (
    AlwaysWrite,
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    Decision,
    Event,
    HeuristicWriteDecider,
    NeverConsolidate,
    PeriodicConsolidator,
    WriteContext,
    WriteDecider,
)

__version__ = "0.1.0"
__all__ = [
    "AlwaysWrite",
    "ConsolidateContext",
    "ConsolidateDecider",
    "ConsolidationDecision",
    "ConsolidationResult",
    "Decision",
    "Event",
    "Fact",
    "HeuristicWriteDecider",
    "Memory",
    "NeverConsolidate",
    "PeriodicConsolidator",
    "RememberResult",
    "WriteContext",
    "WriteDecider",
    "__version__",
]
