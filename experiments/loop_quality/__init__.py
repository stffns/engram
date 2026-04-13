"""Loop-quality benchmarks.

These answer: *does merken's decision loop add value over raw vstash?*

Unlike ``experiments/retrieval/``, these benchmarks are scenario-driven
and simulate the stream an agent actually lives through — multiple
events per topic, paraphrased natural language, meta-discussion —
rather than academic chat-replay.

If a scenario here shows merken beating raw vstash, the loop is
earning its keep. If a scenario shows them tied, the policy on trial
moves to ``merken.policies.experimental`` until a different scenario
revives it.
"""
