# Draft: vstash issue — `Memory.search()` does not auto-scope to instance collection

> **Status:** local draft. Post to `stffns/vstash` when convenient. Found
> while building engram Phase 1 (`feature/should-remember`).

## Title

`Memory.search()` ignores the instance's `collection` when the `collection` argument is omitted

## Body

### Summary

When you construct a `vstash.Memory(project=..., collection="my_coll")` and then call `search(query)` without passing `collection=`, the search returns hits from **all** collections in the database, not from `my_coll`. The instance-level collection is honored on writes (`add` / `remember`) but silently ignored on reads.

This is surprising because:

1. The same omitted-argument pattern *does* scope writes to the instance collection.
2. The docstring for `search` says `Override the default collection filter. Pass None for no filter.` — which reads as "the instance default applies unless you explicitly opt out with `None`," but the actual behavior is "no filter applies unless you explicitly pass a string."
3. The `<object>` sentinel default is opaque; users can't tell from the signature whether it means "use instance default" or "no filter."

### Reproduction

```python
import tempfile, vstash
from pathlib import Path

with tempfile.TemporaryDirectory() as td:
    db = Path(td) / "t.db"
    m = vstash.Memory(project="x", db=db, collection="my_coll")

    # Write one row to the instance collection (default behavior — works as expected)
    m.remember("a fact about wombats", title="t1")

    # Write one row to a *different* collection
    m.remember("audit row mentioning wombats elsewhere",
               title="t2", collection="other_coll", layer="audit")

    print("--- search() with no collection arg ---")
    for r in m.search("wombats", top_k=5):
        print(repr((r.text or "")[:60]))
    # Returns BOTH rows. Expected: only t1.

    print("--- search(collection='my_coll') ---")
    for r in m.search("wombats", top_k=5, collection="my_coll"):
        print(repr((r.text or "")[:60]))
    # Returns only t1, as expected.
```

Tested against `vstash==0.26.0`, Python 3.12, macOS arm64.

### Expected behavior

Either of the following would resolve the surprise:

- **(A) Honor the instance collection on read.** When `collection` is omitted from `search`, scope to the instance collection — symmetric with `add` / `remember`.
- **(B) Make the asymmetry explicit in the docstring.** Document that `search` without `collection=` searches all collections, and that you must pass `collection=self.collection` to scope it. Less ergonomic but at least no longer surprising.

(A) seems strictly better — it matches user intuition and the write/read paths become consistent.

### Workaround (engram-side)

In engram we now store the collection on our wrapper and pass it explicitly on every `search` call:

```python
self._vstash.search(query, top_k=top_k, collection=self.collection, layer=layer)
```

This was caught by a leak test (`test_memory_recall_does_not_leak_audit_rows`) where audit rows written to a separate collection were leaking back into normal recall. The test now passes with the explicit-collection workaround.

### Suggested fix sketch

In `vstash/memory.py` (assuming `_DEFAULT_SENTINEL` is the existing sentinel):

```python
def search(self, query, *, collection=_DEFAULT_SENTINEL, ...):
    if collection is _DEFAULT_SENTINEL:
        collection = self._collection  # honor instance default
    # None still means "no filter"
    ...
```

This makes (A) one line and preserves the `None` escape hatch.

### Why this matters for engram

engram is the first external consumer that maintains a *second* collection (the audit log) inside the same vstash instance. The asymmetry between write and read scoping silently broke isolation between the two collections, which is a meaningful glass-box violation: the audit log is supposed to be invisible to normal recall by construction, and instead it leaked. The workaround is fine for now, but downstream consumers that maintain multiple collections are likely to hit the same trap.

---

*Filed by: engram dev session, 2026-04-08, while implementing `Memory.audit` in `feature/should-remember`.*
