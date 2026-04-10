"""engram — agent-loop layer for persistent memory, built on top of vstash.

See ``CONSTITUTION.md`` for what engram is, what it isn't, and the principles
that should outlive any specific implementation.
"""

from engram.consolidation import ConsolidationResult, Fact
from engram.memory import ForgetResult, Memory, RememberResult
from engram.policies import (
    AlwaysWrite,
    ContentTypePriorDecider,
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    Decision,
    Event,
    ForgetConsolidated,
    ForgetContext,
    ForgetDecider,
    ForgetDecision,
    HeuristicWriteDecider,
    LayerRequest,
    LayeredRecaller,
    NeverConsolidate,
    NeverForget,
    PeriodicConsolidator,
    RecallContext,
    RecallDecider,
    RecallPlan,
    SemanticOnlyRecaller,
    WriteContext,
    WriteDecider,
)

__version__ = "0.1.0"
__all__ = [
    "AlwaysWrite",
    "ContentTypePriorDecider",
    "ConsolidateContext",
    "ConsolidateDecider",
    "ConsolidationDecision",
    "ConsolidationResult",
    "Decision",
    "Event",
    "Fact",
    "ForgetConsolidated",
    "ForgetContext",
    "ForgetDecider",
    "ForgetDecision",
    "ForgetResult",
    "HeuristicWriteDecider",
    "LayerRequest",
    "LayeredRecaller",
    "Memory",
    "NeverConsolidate",
    "NeverForget",
    "PeriodicConsolidator",
    "RecallContext",
    "RecallDecider",
    "RecallPlan",
    "RememberResult",
    "SemanticOnlyRecaller",
    "WriteContext",
    "WriteDecider",
    "__version__",
]
