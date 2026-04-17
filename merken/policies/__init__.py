"""Decision policies for the merken loop.

Each policy is a plain Python callable (Protocol-typed) with explicit inputs
and outputs that you can test, override, or replace. CONSTITUTION §4.7 — no
premature abstraction; if you can't name three concrete callers, you can't
ship the helper.
"""

from merken.policies.should_consolidate import (
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    NeverConsolidate,
    PeriodicConsolidator,
)
from merken.policies.should_forget import (
    ForgetConsolidated,
    ForgetConsolidatedOrSuperseded,
    ForgetContext,
    ForgetDecider,
    ForgetDecision,
    ForgetSuperseded,
    NeverForget,
)
from merken.policies.should_recall import (
    LayeredRecaller,
    LayerRequest,
    RecallContext,
    RecallDecider,
    RecallPlan,
    SemanticOnlyRecaller,
)
from merken.policies.should_remember import (
    AlwaysWrite,
    ChainedWriteDecider,
    ContentTypePriorDecider,
    HeuristicWriteDecider,
    ShadowWriteDecider,
)
from merken.policies.types import Decision, Event, WriteContext, WriteDecider

__all__ = [
    "AlwaysWrite",
    "ChainedWriteDecider",
    "ContentTypePriorDecider",
    "ConsolidateContext",
    "ConsolidateDecider",
    "ConsolidationDecision",
    "Decision",
    "Event",
    "ForgetConsolidated",
    "ForgetConsolidatedOrSuperseded",
    "ForgetContext",
    "ForgetDecider",
    "ForgetDecision",
    "ForgetSuperseded",
    "HeuristicWriteDecider",
    "LayerRequest",
    "LayeredRecaller",
    "NeverConsolidate",
    "NeverForget",
    "PeriodicConsolidator",
    "RecallContext",
    "RecallDecider",
    "RecallPlan",
    "SemanticOnlyRecaller",
    "ShadowWriteDecider",
    "WriteContext",
    "WriteDecider",
]
