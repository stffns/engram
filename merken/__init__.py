"""merken — agent-loop layer for persistent memory, built on top of vstash.

See ``CONSTITUTION.md`` for what merken is, what it isn't, and the principles
that should outlive any specific implementation.
"""

from merken.consolidation import ConsolidationResult, Fact
from merken.memory import ForgetResult, Memory, RememberResult
from merken.policies import (
    AlwaysWrite,
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    ContentTypePriorDecider,
    Decision,
    Event,
    ForgetConsolidated,
    ForgetConsolidatedOrSuperseded,
    ForgetContext,
    ForgetDecider,
    ForgetDecision,
    ForgetSuperseded,
    HeuristicWriteDecider,
    LayeredRecaller,
    LayerRequest,
    NeverConsolidate,
    NeverForget,
    PeriodicConsolidator,
    RecallContext,
    RecallDecider,
    RecallPlan,
    SemanticOnlyRecaller,
    ShadowWriteDecider,
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
    "ForgetConsolidatedOrSuperseded",
    "ForgetContext",
    "ForgetDecider",
    "ForgetDecision",
    "ForgetResult",
    "ForgetSuperseded",
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
    "ShadowWriteDecider",
    "WriteContext",
    "WriteDecider",
    "__version__",
]
