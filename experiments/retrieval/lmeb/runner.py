"""LMEB R@k runner for merken.

Evaluates merken (and raw vstash) against LMEB sub-datasets in the
standard IR format: corpus.jsonl, queries.jsonl, qrels.tsv, and
candidates.jsonl.

The job per scene:
1. Ingest all candidate docs for the scene into a fresh Memory.
2. Optionally consolidate.
3. For each query whose scene matches, recall top-k and check
   whether any hit's doc_id appears in the qrels for that query.

Reports R@k with bootstrap CI, same discipline as the LongMemEval
runner. Results are manual → RESULTS.md (CONSTITUTION §9).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import vstash

from merken import Memory, PeriodicConsolidator


# ----------------------------------------------------------------- data loading


@dataclass
class CorpusDoc:
    id: str
    text: str
    title: str


@dataclass
class Query:
    id: str
    text: str
    scene_id: str  # derived from id prefix


@dataclass
class SceneSpec:
    scene_id: str
    candidate_doc_ids: list[str]


def _scene_from_query_id(qid: str, scene_ids: set[str] | None = None) -> str:
    """Extract scene_id from query id.

    Handles two formats:
    - LoCoMo-style: `scene_0_q_82` → `scene_0`
    - DeepPlanning-style: `case_1` where query.id == scene_id
    """
    if "_q_" in qid:
        return qid.split("_q_")[0]
    if scene_ids is not None and qid in scene_ids:
        return qid
    return qid.rsplit("_", 1)[0]


def load_corpus(path: Path) -> dict[str, CorpusDoc]:
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs[row["id"]] = CorpusDoc(
                id=row["id"],
                text=row["text"],
                title=row.get("title", ""),
            )
    return docs


def load_queries(path: Path, scene_ids: set[str] | None = None) -> list[Query]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries.append(Query(
                id=row["id"],
                text=row["text"],
                scene_id=_scene_from_query_id(row["id"], scene_ids),
            ))
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    """Load qrels.tsv → {query_id: {relevant_doc_ids}}."""
    qrels: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, did = parts[0], parts[1]
                qrels[qid].add(did)
    return dict(qrels)


def load_candidates(path: Path) -> dict[str, SceneSpec]:
    specs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            specs[row["scene_id"]] = SceneSpec(
                scene_id=row["scene_id"],
                candidate_doc_ids=row["candidate_doc_ids"],
            )
    return specs


# ----------------------------------------------------------------- adapters


class _Adapter:
    name: str

    def remember(self, text: str, *, title: str, doc_id: str) -> bool:
        raise NotImplementedError

    def recall(self, query: str, *, top_k: int) -> list[Any]:
        raise NotImplementedError

    def consolidate(self) -> None:
        pass

    def close(self) -> None:
        pass


class _VstashAdapter(_Adapter):
    name = "vstash"

    def __init__(self, project: str, db: Path) -> None:
        self._m = vstash.Memory(project=project, db=db, collection="default")
        self._id_map: dict[str, str] = {}  # vstash_path → doc_id

    def remember(self, text: str, *, title: str, doc_id: str) -> bool:
        result = self._m.remember(text, title=title, collection="default")
        if hasattr(result, "source"):
            self._id_map[result.source] = doc_id
        return True

    def recall(self, query: str, *, top_k: int) -> list[tuple[str, Any]]:
        hits = self._m.search(query, top_k=top_k, collection="default")
        return [(self._id_map.get(getattr(h, "path", ""), ""), h) for h in hits]

    def close(self) -> None:
        self._m.close()


class _EngramAdapter(_Adapter):
    name = "merken-heuristic"

    def __init__(self, project: str, db: Path) -> None:
        self._m = Memory(project=project, db=db)
        self._id_map: dict[str, str] = {}

    def remember(self, text: str, *, title: str, doc_id: str) -> bool:
        result = self._m.remember(text, title=title)
        if result.written and result.ingest is not None:
            self._id_map[result.ingest.source] = doc_id
        return result.written

    def recall(self, query: str, *, top_k: int) -> list[tuple[str, Any]]:
        hits = self._m.recall(query, top_k=top_k)
        return [(self._id_map.get(getattr(h, "path", ""), ""), h) for h in hits]

    def consolidate(self) -> None:
        result = self._m.consolidate(
            method="embedding_v1",
            embedding_threshold=_CONSOLIDATE_THRESHOLD,
            embedding_linkage="complete",
        )
        # Map fact paths back to source doc_ids (first source)
        from merken.consolidation import fact_fingerprint

        for fact in result.facts:
            fp = f"text://fact_{fact_fingerprint(fact)}"
            for src in fact.derived_from:
                src_id = self._id_map.get(src)
                if src_id:
                    self._id_map[fp] = src_id
                    break

    def close(self) -> None:
        self._m.close()


_ADAPTERS: dict[str, type] = {
    "vstash": _VstashAdapter,
    "merken-heuristic": _EngramAdapter,
}

_CONSOLIDATE_THRESHOLD: float = 0.70


# ----------------------------------------------------------------- eval core


@dataclass
class SceneResult:
    scene_id: str
    n_docs: int
    n_queries: int
    hits: list[bool]
    elapsed_s: float


@dataclass
class TaskResult:
    task: str
    baseline: str
    n_scenes: int
    n_queries: int
    r_at_k: float
    ci_low: float
    ci_high: float
    total_elapsed_s: float
    per_scene: list[SceneResult]


def bootstrap_ci(
    values: list[bool],
    *,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_iter)
    )
    return (means[int((alpha / 2) * n_iter)], means[int((1 - alpha / 2) * n_iter) - 1])


def eval_scene(
    adapter: _Adapter,
    scene: SceneSpec,
    corpus: dict[str, CorpusDoc],
    queries: list[Query],
    qrels: dict[str, set[str]],
    *,
    top_k: int,
    consolidate: bool = False,
) -> SceneResult:
    t0 = time.perf_counter()

    # Ingest candidate docs for this scene
    for doc_id in scene.candidate_doc_ids:
        doc = corpus.get(doc_id)
        if doc:
            adapter.remember(doc.text, title=doc.title, doc_id=doc_id)

    if consolidate:
        adapter.consolidate()

    # Evaluate queries for this scene
    scene_queries = [q for q in queries if q.scene_id == scene.scene_id]
    hits: list[bool] = []

    for q in scene_queries:
        relevant = qrels.get(q.id, set())
        if not relevant:
            continue
        results = adapter.recall(q.text, top_k=top_k)
        hit = any(doc_id in relevant for doc_id, _ in results if doc_id)
        hits.append(hit)

    return SceneResult(
        scene_id=scene.scene_id,
        n_docs=len(scene.candidate_doc_ids),
        n_queries=len(scene_queries),
        hits=hits,
        elapsed_s=time.perf_counter() - t0,
    )


def eval_task(
    baseline: str,
    task_dir: Path,
    corpus: dict[str, CorpusDoc],
    candidates: dict[str, SceneSpec],
    *,
    top_k: int,
    db_dir: Path,
    max_scenes: int | None = None,
    consolidate: bool = False,
) -> TaskResult:
    queries = load_queries(task_dir / "queries.jsonl", set(candidates.keys()))
    qrels = load_qrels(task_dir / "qrels.tsv")

    scenes = sorted(candidates.values(), key=lambda s: s.scene_id)
    if max_scenes:
        scenes = scenes[:max_scenes]

    t0 = time.perf_counter()
    per_scene: list[SceneResult] = []
    all_hits: list[bool] = []

    for scene in scenes:
        db = db_dir / f"{baseline}_{task_dir.name}_{scene.scene_id}.db"
        adapter = _ADAPTERS[baseline](
            f"lmeb_{baseline}_{task_dir.name}_{scene.scene_id}", db
        )
        try:
            result = eval_scene(
                adapter, scene, corpus, queries, qrels,
                top_k=top_k, consolidate=consolidate,
            )
            per_scene.append(result)
            all_hits.extend(result.hits)
        finally:
            adapter.close()

    r_at_k = sum(all_hits) / len(all_hits) if all_hits else 0.0
    lo, hi = bootstrap_ci(all_hits)

    return TaskResult(
        task=task_dir.name,
        baseline=baseline,
        n_scenes=len(scenes),
        n_queries=sum(r.n_queries for r in per_scene),
        r_at_k=r_at_k,
        ci_low=lo,
        ci_high=hi,
        total_elapsed_s=time.perf_counter() - t0,
        per_scene=per_scene,
    )


def format_result(result: TaskResult) -> str:
    return (
        f"task={result.task:<25} "
        f"baseline={result.baseline:<20} "
        f"n={result.n_queries:<5} "
        f"R@5={result.r_at_k:.3f} "
        f"95% CI=[{result.ci_low:.3f}, {result.ci_high:.3f}] "
        f"elapsed={result.total_elapsed_s:.1f}s"
    )


# ----------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="experiments.retrieval.lmeb.runner",
        description="LMEB R@k runner for merken baselines.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to an LMEB sub-dataset dir (e.g. .cache/eval_data/Dialogue/LoCoMo)",
    )
    p.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task subdirectory to evaluate (e.g. single_hop). Repeat for multiple. Default: all.",
    )
    p.add_argument(
        "--baseline",
        action="append",
        choices=sorted(_ADAPTERS),
        help="Baseline to evaluate. Repeat for multiple. Default: all.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--consolidate", action="store_true", default=False)
    p.add_argument(
        "--consolidate-threshold",
        type=float,
        default=0.70,
        help="Embedding similarity threshold for consolidation (default 0.70)",
    )
    p.add_argument("--db-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ds_dir = args.dataset_dir

    global _CONSOLIDATE_THRESHOLD
    _CONSOLIDATE_THRESHOLD = args.consolidate_threshold

    corpus = load_corpus(ds_dir / "corpus.jsonl")
    candidates = load_candidates(ds_dir / "candidates.jsonl")

    # Discover tasks (subdirectories with queries.jsonl)
    if args.task:
        task_dirs = [ds_dir / t for t in args.task]
    else:
        task_dirs = sorted(
            d for d in ds_dir.iterdir()
            if d.is_dir() and (d / "queries.jsonl").exists()
        )

    baselines = args.baseline or sorted(_ADAPTERS)

    print(
        f"# LMEB — {ds_dir.parent.name}/{ds_dir.name}, "
        f"{len(corpus)} corpus docs, "
        f"{len(candidates)} scenes, "
        f"top_k={args.top_k}"
    )

    with tempfile.TemporaryDirectory(prefix="merken_lmeb_") as td:
        db_dir = args.db_dir or Path(td)
        db_dir.mkdir(parents=True, exist_ok=True)

        for task_dir in task_dirs:
            for baseline in baselines:
                result = eval_task(
                    baseline,
                    task_dir,
                    corpus,
                    candidates,
                    top_k=args.top_k,
                    db_dir=db_dir,
                    max_scenes=args.max_scenes,
                    consolidate=args.consolidate,
                )
                print(format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
